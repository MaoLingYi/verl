import ast
from pathlib import Path
from types import SimpleNamespace as NS
import weakref
import gc
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec', flags=__import__('__future__').annotations.compiler_flag), namespace)
    return namespace[name]


def replay_api():
    tree = ast.parse((ROOT / 'verl/utils/megatron/router_replay_patch.py').read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in {'RouterReplay', 'RouterReplayAction', '_patched_topk_routing_with_score_function'}]
    from enum import Enum
    ns = {'torch': torch, 'Enum': Enum}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'replay', 'exec', flags=__import__('__future__').annotations.compiler_flag), ns)
    return ns


def test_replay_indices_weights_and_router_gradient_then_release():
    api = replay_api()
    replay = api['RouterReplay']()
    target = torch.tensor([[127, 3, 6, 8, 10, 12, 14, 16]])
    ref = weakref.ref(target)
    replay.set_target_indices(target)
    replay.set_router_replay_action(api['RouterReplayAction'].REPLAY_FORWARD)
    logits = torch.linspace(-1, 1, 128).reshape(1,128).requires_grad_()
    probs, mask = api['_patched_topk_routing_with_score_function'](
        logits, 8, False, None, None, 'softmax', None, False, replay, 1.0)
    expected = torch.softmax(logits.gather(1, target), -1)
    torch.testing.assert_close(probs.gather(1,target), expected)
    assert torch.equal(mask.nonzero()[:,1].sort().values, target[0].sort().values)
    (probs * torch.arange(128)).sum().backward()
    assert logits.grad.abs().sum() > 0
    replay.set_router_replay_action(api['RouterReplayAction'].REPLAY_BACKWARD)
    api['_patched_topk_routing_with_score_function'](
        logits, 8, False, None, None, 'softmax', None, False, replay, 1.0)
    assert not replay.replay_backward_list
    del target, probs, expected, mask
    api['RouterReplay'].clear_global_indices()
    api['RouterReplay'].clear_global_router_replay_action()
    gc.collect()
    assert ref() is None
    assert replay.target_topk_idx is None


@pytest.mark.parametrize('length,extra', [(16384, 0), (16384, 3), (7, 0)])
def test_agent_route_padding_truncation_48_layers_top8(length,extra):
    # Execute the actual route-materialization block without importing Ray/SGLang.
    tree = ast.parse((ROOT / 'verl/experimental/agent_loop/agent_loop.py').read_text(encoding='utf-8'))
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == 'output.routed_experts is not None')
    routes = (torch.arange((length+extra)*48*8).reshape(length+extra,48,8) %128).to(torch.uint8)
    ns = {'torch':torch, 'np': __import__('numpy'), 'output':NS(routed_experts=routes,prompt_ids=[1,2]),
          'input_ids':torch.zeros(1,length+2), 'total_length':length+2, 'prompt_output':{'input_ids':torch.zeros(1,4)}}
    exec(compile(ast.Module(body=[block],type_ignores=[]),'materialize','exec'),ns)
    out=ns['routed_experts']
    assert out.dtype == torch.uint8 and out.shape==(1,length+2,48,8)
    assert out[:,:2].count_nonzero()==0
    torch.testing.assert_close(out[:,2:],routes[:length].unsqueeze(0))


@pytest.mark.parametrize('tp_rank',[0,1])
@pytest.mark.parametrize('retain',[True,False])
def test_packed_route_layer_token_mapping_tp2_pp1_ep8(tp_rank,retain):
    api = replay_api(); routers=[api['RouterReplay']() for _ in range(48)]
    mask=torch.tensor([[0,1,1,0],[1,1,0,0]],dtype=torch.bool)
    routes=(torch.arange(2*4*48*8).reshape(2,4,48,8)%128).to(torch.uint8)
    # Model the sequence-parallel packing contract; use actual replay installation.
    ns={'torch':torch,'preprocess_packed_seqs':function('verl/models/mcore/util.py','preprocess_packed_seqs', {
            'torch':torch, 'PackedSeqParams':lambda **k:NS(**k),
            'mpu':NS(get_tensor_model_parallel_world_size=lambda:2,
                     get_context_parallel_world_size=lambda:1,get_context_parallel_rank=lambda:0)}),
        'device_name':'cpu', 'scatter_to_sequence_parallel_region':lambda x:x.chunk(2)[tp_rank],
        'get_current_rank_layer_info':lambda *a:{'start':0,'end':48},
        'RouterReplayHelper':NS(get_micro_batch_router_list=lambda *a:routers),
        'is_moe_layer':lambda *a:True}
    install=function('verl/utils/megatron/router_replay_utils.py','set_router_replay_data',ns)
    install(routes,mask,NS(num_layers=48),retain_for_backward=retain)
    expected=routes[mask].chunk(2)[tp_rank]
    for layer,router in enumerate(routers):
        assert bool(router.replay_backward_list)==retain
        assert router.target_topk_idx.dtype==torch.int64
        torch.testing.assert_close(router.target_topk_idx,expected[:,layer].long())
        # Global ids (including expert127) must not be renumbered by EP ownership.
    api['RouterReplay'].clear_global_indices()
    assert all(not r.replay_backward_list and r.target_topk_idx is None for r in routers)


@pytest.mark.parametrize('mode,pp,expected',[('R3',1,['cpu']),('R3',4,['cuda','broadcast','cpu']),
                                           ('disabled',1,['cuda','broadcast','cpu'])])
def test_batch_staging_r3_pp1_is_microbatch_only(mode,pp,expected):
    tree=ast.parse((ROOT/'verl/workers/actor/megatron_actor.py').read_text(encoding='utf-8'))
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='forward_backward_batch')
    # Execute actual staging prefix, before microbatch split and model forward.
    prefix=fn.body[1:next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
                         and ast.unparse(n.targets[0])=='mini_batch.batch[\'attention_mask\']')]
    events=[]
    class Batch:
        def contiguous(self): return self
    class Data:
        batch=Batch()
        def to(self,device): events.append(device);return self
    ns={'self':NS(config=NS(router_replay=NS(mode=mode))), 'data':Data(),
        'mpu':NS(get_pipeline_model_parallel_world_size=lambda:pp,
                 get_pipeline_model_parallel_last_rank=lambda:0,get_pipeline_model_parallel_group=lambda:None),
        'get_device_id':lambda:'cuda','broadcast_dict_tensor':lambda *a,**k:events.append('broadcast')}
    exec(compile(ast.Module(body=prefix,type_ignores=[]),'staging','exec'),ns)
    assert events==expected


def test_r3_worker_cleanup_finally_on_error_and_disabled_unchanged():
    tree=ast.parse((ROOT/'verl/workers/megatron_workers.py').read_text(encoding='utf-8'))
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='update_actor')
    final=next(n.finalbody for n in fn.body if isinstance(n,ast.Try))
    for mode in ['R3','disabled']:
        events=[]
        ns={'self':NS(config=NS(actor=NS(router_replay=NS(mode=mode)))), 'v13_route_diagnostic':None,
            'RouterReplay':NS(clear_global_indices=lambda:events.append('indices'),
                              clear_global_router_replay_action=lambda:events.append('action'))}
        try:
            raise RuntimeError('primary')
        except RuntimeError:
            exec(compile(ast.Module(body=final,type_ignores=[]),'cleanup','exec'),ns)
        assert events==(['indices','action'] if mode=='R3' else [])

@pytest.mark.parametrize('mode,eu,offload,expected',[
    ('R3',False,True,1),('disabled',False,True,0),('disabled',True,True,1),('R3',False,False,0)])
def test_backing_release_is_scoped_and_before_actor_compute(mode,eu,offload,expected):
    tree=ast.parse((ROOT/'verl/workers/megatron_workers.py').read_text(encoding='utf-8'))
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='update_actor')
    block=next(n for n in fn.body if isinstance(n,ast.If)
               and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name)
                       and x.func.id=='release_megatron_model_cpu_data' for x in ast.walk(n)))
    events=[]
    model=object()
    ns={'self':NS(config=NS(actor=NS(router_replay=NS(mode=mode))),_is_offload_param=offload,
                  actor_module=model,actor_optimizer=object()),'eu_enabled':eu,'_GIB':2**30,
        'log_eu_derpo_memory':lambda *a,**k:None,'megatron_model_cpu_data_bytes':lambda m:0,
        'release_megatron_model_cpu_data':lambda m:events.append(('release',m)) or 123,
        '_best_effort_release_host_allocator_after_copy_param_restore':lambda:events.append('reclaim') or {},
        'torch':torch}
    exec(compile(ast.Module(body=[block],type_ignores=[]),'release','exec'),ns)
    assert events==([('release',model),'reclaim'] if expected else [])
