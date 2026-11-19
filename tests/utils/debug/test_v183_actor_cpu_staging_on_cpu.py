"""Production actor staging on CPU: optional GPU/MCore/containers are fixtures, not eight-card validation."""
import ast
from functools import partial
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch

import test_eu_derpo_observer_on_cpu as harness

ROOT = Path(__file__).parents[3]
ACTOR = ROOT / "verl/workers/actor/megatron_actor.py"
BASELINE = "610e4d7bb39f5b7c9c4d6bc4bb28d6b4c9e5b6f6"


class Config(SimpleNamespace):
    def get(self, name, default=None):
        return getattr(self, name, default)


class Batch(dict):
    """CPU transport instrument: each virtual device transfer clones actual tensor leaves."""
    def __init__(self, values, *, device="cpu", ledger=None, batch_size=None, names=None):
        super().__init__(values)
        self.device, self.ledger, self.names = device, ledger if ledger is not None else [], names
        self.batch_size = torch.Size([len(next(iter(values.values())))]) if batch_size is None else batch_size

    def to(self, device):
        device = str(device)
        if device == self.device:
            return self
        size = sum(value.numel() * value.element_size() for value in self.values())
        self.ledger.append((self.device, device, len(next(iter(self.values()))), size))
        return Batch({key: value.clone() for key, value in self.items()}, device=device, ledger=self.ledger)

    def contiguous(self):
        return Batch({key: value.contiguous() for key, value in self.items()}, device=self.device, ledger=self.ledger)

    def split(self, size):
        length = len(next(iter(self.values())))
        return [Batch({key: value[i:i+size] for key, value in self.items()},
                      device=self.device, ledger=self.ledger) for i in range(0, length, size)]


class Proto:
    def __init__(self, batch, meta_info=None, non_tensor_batch=None):
        self.batch = batch
        self.meta_info = meta_info if meta_info is not None else {}
        self.non_tensor_batch = non_tensor_batch if non_tensor_batch is not None else {}


def compile_node(node, namespace, source):
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace[node.name]


# Execute the actual DataProto.to contract; only TensorDict's unavailable transport is a fixture.
proto_tree = ast.parse((ROOT / "verl/protocol.py").read_text(encoding="utf-8"))
proto_class = next(n for n in proto_tree.body if isinstance(n, ast.ClassDef) and n.name == "DataProto")
Proto.to = compile_node(next(n for n in proto_class.body if isinstance(n, ast.FunctionDef) and n.name == "to"), {},
                        ROOT / "verl/protocol.py")


def actor_source(old=False):
    if not old:
        return ACTOR.read_text(encoding="utf-8")
    return subprocess.run(["git", "-c", "safe.directory=" + str(ROOT), "-C", str(ROOT), "show",
                           BASELINE + ":verl/workers/actor/megatron_actor.py"],
                          check=True, capture_output=True, text=True, encoding="utf-8").stdout


def actor_node(old=False):
    tree = ast.parse(actor_source(old))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "forward_backward_batch")


def load_math(name):
    full = "verl.trainer.ppo." + name
    spec = importlib.util.spec_from_file_location(full, ROOT / ("verl/trainer/ppo/" + name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


def packed(records, input_ids, attention, response_length):
    full = torch.zeros((*input_ids.shape, *records.shape[2:]), dtype=records.dtype)
    full[:, -response_length-1:-1] = records
    return full[attention.bool()]


def unpacked(records, input_ids, attention):
    values = torch.stack(records, dim=1)
    full = torch.zeros((*input_ids.shape, *values.shape[1:]), dtype=values.dtype)
    full[attention.bool()] = values
    return full


def make_inputs():
    source = torch.arange(42).reshape(3, 14) % 16
    # Non-contiguous data, deliberately non-sorted sample IDs and unequal valid lengths.
    ids = source[:, ::2]
    attention = torch.tensor([[0,1,1,1,1,1,1], [1,1,1,1,1,0,0], [1,1,1,1,1,1,0]], dtype=torch.int64)
    response = torch.tensor([[1,1,1,1], [1,1,0,0], [1,1,1,0]], dtype=torch.bool)
    return dict(input_ids=ids, attention_mask=attention, position_ids=torch.arange(14).expand(3,-1)[:,::2],
                responses=ids[:, -4:], response_mask=response,
                advantages=torch.tensor([[1.],[-1.],[0.]]).expand(-1,4),
                rollout_log_probs=torch.full((3,4), -3.), eu_derpo_sample_ids=torch.tensor([23,17,31]),
                eu_derpo_prompt_group=torch.tensor([4,4,4]))


def execute(old, lambda_u, *, fail=False):
    case = harness.TestEUDERPOObserver()
    case.setUp()
    saved = {}
    def install(name, module):
        saved.setdefault(name, sys.modules.get(name))
        sys.modules[name] = module
    inputs = make_inputs()
    ledger, seen, reduced, broadcasts = [], [], [], []
    # The fixture represents one of four identical DP owners (nine local valid tokens each).
    batch = Proto(Batch(inputs, ledger=ledger), {"temperature": 1., "v183_global_tokens": 36})
    original = {key: value.clone() for key, value in inputs.items()}
    generator = torch.Generator().manual_seed(77)
    case.model.embedding = torch.nn.Embedding.from_pretrained(torch.randn(16,4,generator=generator), freeze=False)
    case.model.expert_values = torch.nn.Parameter(torch.randn(128,16,generator=generator) * .2)
    case.model.dense = torch.nn.Linear(4,4, bias=False)
    with torch.no_grad():
        case.model.dense.weight.copy_(torch.eye(4) * .3)
    try:
        for name in ("eu_derpo_v18", "eu_derpo_v182", "eu_derpo_v183"):
            saved["verl.trainer.ppo." + name] = sys.modules.get("verl.trainer.ppo." + name)
        v18, v182, v183 = [load_math(name) for name in ("eu_derpo_v18", "eu_derpo_v182", "eu_derpo_v183")]
        eu = sys.modules["verl.trainer.ppo.eu_derpo"]
        cfg = Config(policy_loss=Config(loss_mode="vanilla"),
                     eu_derpo=Config(enabled=True, version="1.8.3",
                                     expert_cluster_dppo=Config(delta_e=.012, diagnostics_only=False),
                                     behavior_expert_is=Config(behavior_source="rollout")),
                     loss_agg_mode="token-mean", entropy_coeff=0., use_kl_loss=False,
                     clip_ratio=.2, router_shift_weighting=Config(enabled=False),
                     router_replay=Config(mode="disabled"), megatron=Config(use_remove_padding=True))
        actor = Config(config=cfg, actor_module=[case.model], tf_config=Config(virtual_pipeline_model_parallel_size=None),
                       router_shift_observer=None, eu_derpo_observer=case.observer,
                       mctr_controller=object(), _eu_derpo_delta_current=.012,
                       use_fused_kernels=False, mtp_config=None, hf_config=Config())
        def forward_fn(*, model, input_ids, attention_mask, position_ids, logits_processor,
                       logits_processor_args, data_format, **kwargs):
            seen.append(tuple(x.clone() for x in (input_ids, attention_mask, position_ids,
                                                  logits_processor_args["label"],
                                                  logits_processor_args["label_mask"])))
            assert data_format == "thd"
            if fail and len(seen) == 2:
                raise RuntimeError("injected model failure")
            hidden = model.dense(model.embedding(input_ids[attention_mask]))
            with torch.no_grad():
                model(hidden)
            outputs = model(hidden)  # CPU fixture for actual F/recompute R observer hooks.
            probabilities = torch.stack([p for p, _ in outputs]).mean(0)
            logits = probabilities @ model.expert_values
            full = torch.zeros((*input_ids.shape, 16), dtype=logits.dtype)
            full[attention_mask] = logits
            return logits_processor(full, **logits_processor_args)

        mcore = ModuleType("verl.models.mcore")
        mcore.get_mcore_forward_fn = lambda config: forward_fn
        mcore.get_mcore_forward_fused_fn = lambda config: None
        install(mcore.__name__, mcore)
        corr = ModuleType("verl.trainer.ppo.rollout_corr_helper")
        corr.compute_rollout_corr_metrics_from_logprobs = lambda **kwargs: {}
        install(corr.__name__, corr)

        def schedule(**kwargs):
            assert kwargs["micro_batch_size"] == 1 and kwargs["num_microbatches"] == 3
            for _ in range(kwargs["num_microbatches"]):
                output, closure = kwargs["forward_step_func"](kwargs["data_iterator"], kwargs["model"][0])
                loss, metrics = closure(output)
                reduced.append((output["log_probs"].detach().clone(), loss.detach().clone()))
                (loss / kwargs["num_microbatches"]).backward()
            return [metrics]

        replay = Config(is_replay_backward_action=lambda *a: False, is_replay_forward_action=lambda *a: False,
                        is_r2_record_action=lambda *a: False)
        namespace = dict(torch=torch, time=time, partial=partial, TensorDict=Batch, DataProto=Proto,
                         get_device_id=lambda: "virtual_cuda", mpu=case.parallel_state,
                         broadcast_dict_tensor=lambda *a, **k: broadcasts.append(True),
                         get_forward_backward_func=lambda: schedule, unwrap_model=lambda model: model,
                         make_batch_generator=lambda batches, vpp_size: iter(batches),
                         RouterReplayHelper=replay, eu_derpo_v18=v18, eu_derpo_v182=v182, eu_derpo_v183=v183,
                         append_to_dict=lambda dst, src: dst.update(src),
                         vocab_parallel_log_probs_from_logits=lambda logits, label:
                            logits.log_softmax(-1).gather(-1, label.unsqueeze(-1)).squeeze(-1))
        case.parallel_state.get_pipeline_model_parallel_last_rank = lambda: 0
        method = compile_node(actor_node(old), namespace, ACTOR)
        # Production update_policy error decorator is exercised around the same forward/backward call.
        method = case.observer_module.clear_eu_derpo_on_error(method)
        with mock.patch.object(harness.FakeRouterShiftObserver, "_pack_sequence_parallel", side_effect=packed), \
                mock.patch.object(harness.FakeRouterShiftObserver, "_unpack_sequence_parallel", side_effect=unpacked):
            case.observer.start_main_batch(inputs["eu_derpo_sample_ids"], inputs["eu_derpo_prompt_group"],
                                           inputs["attention_mask"], inputs["response_mask"], 4, 0)
            if fail:
                with pytest.raises(RuntimeError, match="injected model failure"):
                    method(actor, batch, micro_batch_size=1)
                assert case.observer._hidden_cache is None and not case.observer.route_cache
                assert all(torch.equal(value, original[key]) for key, value in inputs.items())
                return None
            method(actor, batch, micro_batch_size=1)
            native = {name: parameter.grad.clone() for name, parameter in case.model.named_parameters()}
            for router in case.model.routers:
                router.weight.main_grad = router.weight.grad.clone()
            case.observer.wrap_native_finalize(lambda: None)()
            teacher, squares, counts, capture = case.observer.finish_main_batch()
            rows = list(case.observer.consume_post_backward_rows(inputs["eu_derpo_sample_ids"]))
            stats = []
            coefficients, objective = [], []
            for row, (routes, current) in enumerate(rows):
                mask = inputs["response_mask"][row:row+1]
                st = v18.cluster_statistics(current[None], inputs["rollout_log_probs"][row:row+1],
                                            inputs["advantages"][row:row+1], mask, routes[None], 128, .012)
                stats.append(st)
                coefficients.append(v183.schedule_coefficients(st, routes[None], mask, 36, 3, 4))
                objective.append(v183.objective_numerator(st, routes[None], mask))
            normalized, _ = eu.normalize_group_utility(teacher, counts, inputs["eu_derpo_prompt_group"], 1e-6, 2, 5e-7)
            masks = torch.cat([st.mask for st in stats])
            population = torch.cat([st.divergence[st.directional_outward] for st in stats])
            controller = eu.P95MCTRController(.012,.0027,.012,.018)
            auxiliary = case.observer.run_router_only_step_e(inputs["eu_derpo_sample_ids"],
                Config(count=counts, mask=masks), normalized, (counts > 0).sum((1,2)), lambda_u, 0)
            case.observer.validate_before_optimizer_step(0)
            controller.update(torch.quantile(population, .95).item() if population.numel() else None,
                              population.numel())
            final = {str(i): router.weight.main_grad.clone() for i, router in enumerate(case.model.routers)}
            assert auxiliary["aux_reduce_count_by_layer"] == auxiliary["main_grad_add_count_by_layer"] == [1]*48
            assert native["embedding.weight"].abs().sum() > 0 and native["expert_values"].abs().sum() > 0
            assert all(torch.equal(value, original[key]) for key, value in inputs.items())
            assert batch.meta_info["v183_global_tokens"] == 36
            return dict(ledger=ledger, broadcasts=len(broadcasts), seen=seen, reduced=reduced, native=native,
                        final=final, teacher=teacher, squares=squares, counts=counts, masks=masks,
                        population=population, coefficient=torch.cat(coefficients), objective=torch.stack(objective),
                        next_delta=controller.delta_current, hooks=capture["gradient_hook_count"])
    finally:
        case.tearDown()
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


@pytest.mark.parametrize("lambda_u", [0., .2])
def test_actual_actor_cpu_staging_full_path_equal(lambda_u):
    old, new = execute(True, lambda_u), execute(False, lambda_u)
    for key in ("teacher", "squares", "counts", "masks", "population", "coefficient", "objective"):
        assert torch.equal(old[key], new[key]), key
    for key in ("native", "final"):
        assert old[key].keys() == new[key].keys()
        assert all(torch.equal(old[key][name], new[key][name]) for name in old[key]), key
    for key in ("seen", "reduced"):
        assert all(torch.equal(left, right) for a, b in zip(old[key], new[key]) for left, right in zip(a,b)), key
    assert old["next_delta"] == new["next_delta"] and old["hooks"] == new["hooks"] == 144
    assert old["broadcasts"] == 1 and new["broadcasts"] == 0
    assert [n for _, _, n, _ in old["ledger"]] == [3,3,1,1,1]
    assert [n for _, _, n, _ in new["ledger"]] == [1,1,1]
    assert sum(event[3] for event in new["ledger"]) < sum(event[3] for event in old["ledger"])


@pytest.mark.parametrize("old", [True, False])
def test_actor_error_retains_input_and_clears_observer(old):
    execute(old, .2, fail=True)


@pytest.mark.parametrize("version,loss,pp,forward,dynamic,mb,cpu,expected", [
    ("1.8.3","vanilla",1,False,False,1,True,True),
    ("1.8.2","vanilla",1,False,False,1,True,False),
    ("1.8","vanilla",1,False,False,1,True,False),
    ("1.7","vanilla",1,False,False,1,True,False),
    ("1.8.3","vanilla",2,False,False,1,True,False),
    ("1.8.3","vanilla",1,True,False,1,True,False),
    ("1.8.3","vanilla",1,False,True,1,True,False),
    ("1.8.3","vanilla",1,False,False,2,True,False),
    ("1.8.3","vanilla",1,False,False,1,False,False),
    ("1.8.3","dppo_tv",1,True,True,2,True,True),
    ("1.8.3","dppo_kl",1,False,False,1,True,True),
])
def test_scope_preserves_legacy_staging(version,loss,pp,forward,dynamic,mb,cpu,expected):
    node = next(n for n in actor_node().body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id == "cpu_microbatch_staging" for t in n.targets))
    namespace = dict(torch=torch, self=Config(config=Config(policy_loss=Config(loss_mode=loss),
        eu_derpo=Config(enabled=True,version=version))),
        mpu=Config(get_pipeline_model_parallel_world_size=lambda: pp), forward_only=forward,
        use_dynamic_bsz=dynamic, micro_batch_size=mb,
        data=Proto(Batch({"x": torch.empty(1,device="cpu" if cpu else "meta")})))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(ACTOR),"exec"),namespace)
    assert namespace["cpu_microbatch_staging"] is expected

def copy_profile(old):
    """Run actual production staging AST with virtual device copies; not GPU timing."""
    import psutil
    from test_v183_hidden_store_on_cpu import memory_sample
    torch.set_num_threads(1)
    batch_size, sequence, response = 128, 17408, 16384
    inputs = {
        "input_ids": torch.ones((batch_size, sequence), dtype=torch.int64),
        "attention_mask": torch.ones((batch_size, sequence), dtype=torch.int64),
        "position_ids": torch.zeros((batch_size, sequence), dtype=torch.int64),
        "responses": torch.ones((batch_size, response), dtype=torch.int64),
        "response_mask": torch.ones((batch_size, response), dtype=torch.bool),
        "advantages": torch.ones((batch_size, response)),
        "rollout_log_probs": torch.zeros((batch_size, response)),
        "eu_derpo_sample_ids": torch.arange(batch_size),
        "eu_derpo_prompt_group": torch.arange(batch_size) // 8,
    }
    ledger = []
    data = Proto(Batch(inputs, ledger=ledger), {"temperature": 1., "v183_global_tokens": batch_size * response})
    observer = Config()
    actor = Config(config=Config(policy_loss=Config(loss_mode="vanilla"),
                                  eu_derpo=Config(enabled=True, version="1.8.3")),
                   eu_derpo_observer=observer, router_shift_observer=None)
    namespace = dict(torch=torch, self=actor, data=data, DataProto=Proto, TensorDict=Batch,
                     forward_only=False, use_dynamic_bsz=False, micro_batch_size=1,
                     max_token_len=None, get_device_id=lambda: "virtual_cuda",
                     mpu=Config(get_pipeline_model_parallel_world_size=lambda:1,
                                get_pipeline_model_parallel_last_rank=lambda:0,
                                get_pipeline_model_parallel_group=lambda:None),
                     broadcast_dict_tensor=lambda *a, **k: None, get_forward_backward_func=lambda:None)
    node = actor_node(old)
    prefix = node.body[:next(i for i,n in enumerate(node.body) if isinstance(n, ast.FunctionDef))]
    forward = next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name == "forward_step")
    stop = next(i for i,n in enumerate(forward.body) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id == "position_ids" for t in n.targets))
    forward.body = forward.body[:stop+1] + [ast.Return(value=ast.Name(id="batch",ctx=ast.Load()))]
    micro_stage = compile_node(forward, namespace, ACTOR)
    process = psutil.Process()
    before = memory_sample()
    peak = before["rss"]
    started = time.perf_counter()
    exec(compile(ast.fix_missing_locations(ast.Module(body=prefix,type_ignores=[])),str(ACTOR),"exec"), namespace)
    whole_seconds = time.perf_counter() - started
    peak = max(peak, process.memory_info().rss)
    iterator = iter(namespace["micro_batches"])
    started = time.perf_counter()
    for i in range(batch_size):
        staged = micro_stage(iterator, None)
        assert staged["eu_derpo_sample_ids"].item() == i
        assert staged["response_mask"].sum() == response
        peak = max(peak, process.memory_info().rss)
        del staged
    micro_seconds = time.perf_counter() - started
    info = process.memory_info()
    return dict(implementation="old" if old else "cpu_staging",
                scope="production staging AST; CPU clones emulate CUDA transfers; no GPU/Ray/MCore",
                shape=[batch_size,sequence], response_length=response, attention_dtype="int64",
                input_payload_bytes=sum(t.numel()*t.element_size() for t in inputs.values()),
                transfers=[dict(source=a,destination=b,rows=n,bytes=size) for a,b,n,size in ledger],
                transferred_bytes=sum(e[3] for e in ledger), calls=len(ledger),
                whole_batch_seconds=whole_seconds,microbatch_seconds=micro_seconds,
                total_seconds=whole_seconds+micro_seconds, memory_before=before,
                memory_after=memory_sample(),sampled_peak_rss=peak,
                process_peak_rss=getattr(info,"peak_wset",None),
                gpu_allocated=None,gpu_reserved=None, bit_exact=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one", choices=["old","new"])
    parser.add_argument("--benchmark-root",type=Path)
    parser.add_argument("--repetitions",type=int,default=3)
    args = parser.parse_args()
    if args.one:
        print(json.dumps(copy_profile(args.one == "old")))
    else:
        if args.benchmark_root is None:
            parser.error("--benchmark-root is required")
        args.benchmark_root.mkdir(parents=True,exist_ok=True)
        measurements = []
        for _ in range(args.repetitions):
            for variant in ("old","new"):
                result = subprocess.run([sys.executable,__file__,"--one",variant],
                                        check=True,capture_output=True,text=True,encoding="utf-8")
                measurements.append(json.loads(result.stdout.splitlines()[-1]))
        destination = args.benchmark_root / "results.json"
        destination.write_text(json.dumps(measurements,indent=2)+"\n",encoding="utf-8")
        print(destination)
