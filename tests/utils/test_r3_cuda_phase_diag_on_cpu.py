"""Execute diagnostic/step production fragments; CUDA/MCore boundaries are mocked."""
import ast
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def diagnostic(monkeypatch, enabled=True):
    monkeypatch.setenv('R3_CUDA_PHASE_DIAG', '1' if enabled else '0')
    cuda = NS(is_initialized=lambda: True, current_device=lambda: 0, get_device_properties=lambda _: NS(uuid='GPU-test'),
              memory_allocated=lambda: 10, memory_reserved=lambda: 30,
              max_memory_allocated=lambda: 40, max_memory_reserved=lambda: 50,
              mem_get_info=lambda: (60, 100),
              memory_stats=lambda: {'active_bytes.all.current': 10, 'inactive_split_bytes.all.current': 5})
    ns = dict(os=os, datetime=datetime, torch=NS(cuda=cuda, distributed=NS(
        is_initialized=lambda: True, get_rank=lambda: 3)), logger=NS(warning=Mock()),
        estimate_optimizer_phase_offload_memory=lambda _: {'optimizer_phase_cuda_state_bytes': 7}, _GIB=1024**3)
    tree = ast.parse((ROOT/'verl/utils/memory_utils.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'log_r3_cuda_phase')
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'memory_utils.py', 'exec'), ns)
    return ns, cuda


@pytest.mark.parametrize('enabled,r3', [(None, True), (False, True), (True, False)])
def test_disabled_diagnostic_never_reads_cuda(monkeypatch, enabled, r3):
    ns, cuda = diagnostic(monkeypatch, enabled)
    if enabled is None:
        monkeypatch.delenv("R3_CUDA_PHASE_DIAG", raising=False)
    cuda.current_device = Mock(side_effect=AssertionError('CUDA must not be read'))
    assert ns['log_r3_cuda_phase']('unused', r3_enabled=r3) is None
    cuda.current_device.assert_not_called()
    ns['logger'].warning.assert_not_called()


def test_diagnostic_identity_allocator_and_independent_api_failures(monkeypatch):
    ns, cuda = diagnostic(monkeypatch)
    def fail():
        raise RuntimeError('CUDA allocator failed')
    cuda.mem_get_info = fail
    result = ns['log_r3_cuda_phase']('before_optimizer_step', r3_enabled=True)
    assert result['global_rank'] == 3 and result['physical_gpu_uuid'] == 'GPU-test'
    assert result['pid'] == os.getpid() and result['timestamp']
    assert result['cuda_memory_allocated'] == 10 and result['cuda_memory_reserved'] == 30
    assert result['pytorch_allocator']['inactive_split_bytes.all.current'] == 5
    assert 'CUDA allocator failed' in result['errors']['cuda_mem_get_info']
    assert result['optimizer_phase']['optimizer_phase_cuda_state_bytes'] == 7
    ns['logger'].warning.side_effect = RuntimeError('logger unavailable')
    assert ns['log_r3_cuda_phase']('optimizer_step_exception', r3_enabled=True)


def step_fragment(ns, step):
    tree = ast.parse((ROOT/'verl/workers/actor/megatron_actor.py').read_text(encoding='utf-8'))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'update_policy')
    loop = next(n for n in method.body if isinstance(n, ast.For))
    # Locate the real try/except around optimizer.step, retaining both surrounding diagnostics.
    idx = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Try)
               and 'self.actor_optimizer.step()' in ast.unparse(n))
    block = loop.body[idx-1:idx+2]
    ns['self'] = NS(config=NS(router_replay=NS(mode='R3')), actor_optimizer=NS(step=step), actor_module=None)
    return compile(ast.Module(body=block, type_ignores=[]), 'actor_step.py', 'exec')


def test_optimizer_exception_identity_survives_cuda_and_logger_failures(monkeypatch):
    ns, cuda = diagnostic(monkeypatch)
    original = RuntimeError('ncclUnhandledCudaError')
    cuda.memory_allocated = Mock(side_effect=RuntimeError('poisoned CUDA'))
    ns['logger'].warning.side_effect = RuntimeError('log failure')
    step = Mock(side_effect=original)
    code = step_fragment(ns, step)
    with pytest.raises(RuntimeError) as caught:
        exec(code, ns)
    assert caught.value is original
    step.assert_called_once()


def test_two_optimizer_boundaries_preserve_order_and_results(monkeypatch):
    ns, _ = diagnostic(monkeypatch)
    import torch
    from tests.utils.test_r3_v17_workload_on_cpu import replay_api
    api = replay_api()
    replay = api['RouterReplay']()
    def native_step():
        assert not replay.replay_backward_list
        assert logits.grad is not None and logits.grad.abs().sum() > 0
        return True, 1.25, 0
    step = Mock(side_effect=native_step)
    code = step_fragment(ns, step)
    for _ in range(2):
        target = torch.tensor([[127, 3, 6, 8, 10, 12, 14, 16]])
        replay.set_target_indices(target)
        logits = torch.linspace(-1, 1, 128).reshape(1, 128).requires_grad_()
        for action in ('REPLAY_FORWARD', 'REPLAY_BACKWARD'):
            replay.set_router_replay_action(getattr(api['RouterReplayAction'], action))
            probs, _ = api['_patched_topk_routing_with_score_function'](
                logits, 8, False, None, None, 'softmax', None, False, replay, 1.0)
        (probs * torch.arange(128)).sum().backward()
        exec(code, ns)
        assert (ns['update_successful'], ns['grad_norm'], ns['num_zeros_in_grad']) == (True, 1.25, 0)
        api['RouterReplay'].clear_global_indices()
        api['RouterReplay'].clear_global_router_replay_action()
        assert replay.target_topk_idx is None
    assert step.call_count == 2
    stages = [call.args[1]['stage'] for call in ns['logger'].warning.call_args_list]
    assert stages == ['before_optimizer_step', 'after_optimizer_step'] * 2


def test_http_control_process_diagnostic_does_not_initialize_cuda(monkeypatch):
    ns, cuda = diagnostic(monkeypatch)
    cuda.is_initialized = lambda: False
    cuda.current_device = Mock(side_effect=AssertionError('must not initialize CUDA'))
    result = ns['log_r3_cuda_phase']('sglang_sleep_ack', r3_enabled=True)
    assert result['cuda_initialized'] is False
    cuda.current_device.assert_not_called()


def test_pp1_microbatch_sampling_is_low_frequency_and_forward_only_is_silent():
    tree = ast.parse((ROOT/'verl/workers/actor/megatron_actor.py').read_text(encoding='utf-8'))
    forward = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'forward_step')
    block = next(n for n in forward.body if isinstance(n, ast.If) and 'diag_microbatches' in ast.unparse(n))
    events = []
    ns = {'self': NS(config=NS(router_replay=NS(mode='R3')), actor_optimizer=None, actor_module=None),
          'mpu': NS(get_pipeline_model_parallel_world_size=lambda: 1),
          'diag_microbatches': 0, 'forward_only': False,
          'log_r3_cuda_phase': lambda stage, *a, **k: events.append(stage)}
    code = compile(ast.Module(body=[block], type_ignores=[]), 'microbatch_diag.py', 'exec')
    for _ in range(33):
        exec(code, ns)
    assert events == ['after_microbatch_backward_16', 'after_microbatch_backward_32']
    ns['forward_only'] = True
    for _ in range(33):
        exec(code, ns)
    assert len(events) == 2 and ns['diag_microbatches'] == 33


def test_uuid_read_failure_is_explicit_and_keeps_pid_ordinal(monkeypatch):
    ns, cuda = diagnostic(monkeypatch)
    cuda.get_device_properties = Mock(side_effect=RuntimeError('UUID unavailable'))
    result = ns['log_r3_cuda_phase']('before_optimizer_step', r3_enabled=True)
    assert 'UUID unavailable' in result['errors']['physical_gpu_uuid']
    assert result['pid'] == os.getpid() and result['cuda_device'] == 0
    assert result['cuda_memory_allocated'] == 10
