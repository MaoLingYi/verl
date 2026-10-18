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


def materialize_routes(routes, prompt_length, response_length, prompt_width=None):
    # Execute production CPU tensor code; this does not execute SGLang or Ray.
    tree = ast.parse((ROOT / 'verl/experimental/agent_loop/agent_loop.py').read_text(encoding='utf-8'))
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == 'output.routed_experts is not None')
    width = prompt_width or prompt_length
    ns = {'self':NS(model_config=NS(hf_config=NS(num_hidden_layers=48,num_experts_per_tok=8,num_experts=128))),
          'torch':torch, 'np': __import__('numpy'),
          'output':NS(routed_experts=routes,prompt_ids=[1]*prompt_length,response_ids=[2]*response_length),
          'input_ids':torch.zeros(1,width+response_length),
          'prompt_output':{'input_ids':torch.zeros(1,width)}}
    exec(compile(ast.Module(body=[block],type_ignores=[]),'materialize','exec'),ns)
    return ns['routed_experts']


@pytest.mark.parametrize('prompt_length,response_length,tail', [(2,1,-1),(7,4,-1),(2,16384,-1),(3,4,0)])
def test_agent_route_causal_coverage_48_layers_top8(prompt_length,response_length,tail):
    # v0.5.9 SGLang returns P+R-1 routes; a fully captured terminal input is harmless too.
    length=prompt_length+response_length+tail
    routes=(torch.arange(length*48*8).reshape(length,48,8)%128).to(torch.uint8)
    width=prompt_length+5
    out=materialize_routes(routes,prompt_length,response_length,width)
    assert out.dtype==torch.uint8 and out.shape==(1,width+response_length,48,8)
    assert out[:,:5].count_nonzero()==0
    torch.testing.assert_close(out[:,5:5+length],routes.unsqueeze(0))
    # Shifted logits predicting each response (including EOS/hard-cap last token) are all covered.
    positions=torch.arange(width-1,width+response_length-1)
    torch.testing.assert_close(out[0,positions],routes[positions-5])
    if tail==-1:
        assert out[0,-1].count_nonzero()==0  # No later loss depends on this input's forward.


@pytest.mark.parametrize('response_length', [1,7,16384])
@pytest.mark.parametrize('delta', [-2,1,3])
def test_agent_rejects_missing_causal_routes_and_unexplained_overlong_payload(response_length,delta):
    routes=torch.zeros(2+response_length+delta,48,8,dtype=torch.uint8)
    with pytest.raises(ValueError,match='causal replay inputs'):
        materialize_routes(routes,2,response_length,4)


@pytest.mark.parametrize('present,required,valid', [
    ([True,True],True,True),([True,False],True,False),([False,True],False,False),
    ([False,False],True,False),([False,False],False,True)])
def test_batch_route_completeness_and_validation_without_capture(present,required,valid):
    tree=ast.parse((ROOT/'verl/experimental/agent_loop/agent_loop.py').read_text(encoding='utf-8'))
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_postprocess')
    start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
               and ast.unparse(n.targets[0])=='route_present')
    ns={'inputs':[NS(routed_experts=torch.zeros(1,5,48,8,dtype=torch.uint8) if p else None) for p in present],
        'require_routing_replay':required,'optional_outputs':{},'torch':torch}
    code=compile(ast.Module(body=fn.body[start:start+3],type_ignores=[]),'batch-routes','exec')
    if not valid:
        with pytest.raises(ValueError,match='every training trajectory'):exec(code,ns)
    else:
        exec(code,ns)
        assert ('routed_experts' in ns['optional_outputs'])==all(present)


@pytest.mark.parametrize('mode,validate,legacy,required',[
    ('R3',False,True,True),('R3',True,True,False),('disabled',False,True,False),
    ('R3',False,False,True),(None,False,False,False)])
def test_training_requires_routes_but_validation_does_not(mode,validate,legacy,required):
    from omegaconf import OmegaConf
    tree=ast.parse((ROOT/'verl/experimental/agent_loop/agent_loop.py').read_text(encoding='utf-8'))
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
              and n.func.attr=='_postprocess')
    condition=next(k.value for k in call.keywords if k.arg=='require_routing_replay')
    config={'actor':{'router_replay':{'mode':mode}}} if mode else {}
    if legacy:config={'actor_rollout_ref':config}
    ns={'self':NS(config=OmegaConf.create(config)),'OmegaConf':OmegaConf,
        'batch':NS(meta_info={'validate':validate})}
    assert eval(compile(ast.Expression(condition),'require-routes','eval'),ns)==required


def test_single_turn_does_not_silently_truncate_routes_before_coverage_check():
    from contextlib import nullcontext
    tree=ast.parse((ROOT/'verl/experimental/agent_loop/single_turn_agent_loop.py').read_text(encoding='utf-8'))
    run=next(n for n in ast.walk(tree) if isinstance(n,ast.AsyncFunctionDef) and n.name=='run')
    raw=torch.ones(20,48,8,dtype=torch.uint8)
    async def vision(*a,**k):return {}
    async def prompt(*a,**k):return [1,2]
    async def generate(**k):return NS(token_ids=[3,4],log_probs=None,routed_experts=raw,
                                     num_preempted=None,extra_fields={})
    owner=NS(process_vision_info=vision,apply_chat_template=prompt,
             server_manager=NS(generate=generate),response_length=2)
    ns={'AgentLoopOutput':lambda **k:NS(**k),'simple_timer':lambda *a:nullcontext(),
        'uuid4':lambda:NS(hex='request')}
    exec(compile(ast.Module(body=[run],type_ignores=[]),'single-turn','exec',
                 flags=__import__('__future__').annotations.compiler_flag),ns)
    coroutine=ns['run'](owner,{},raw_prompt=[])
    with pytest.raises(StopIteration) as finished:
        coroutine.send(None)  # All awaited stubs complete synchronously; no OS event loop needed.
    output=finished.value.value
    assert output.routed_experts is raw
    with pytest.raises(ValueError,match='causal replay inputs'):
        materialize_routes(output.routed_experts,2,2)


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

@pytest.mark.parametrize('aux,z,balance',[(0.,0.,'none'),(.01,0.,'none'),(0.,.01,'none'),(0.,0.,'aux_loss')])
def test_built_r3_transformer_config_scalar_overrides_are_checked(aux,z,balance):
    tree=ast.parse((ROOT/'verl/workers/megatron_workers.py').read_text(encoding='utf-8'))
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_build_model_optimizer')
    block=next(n for n in ast.walk(fn) if isinstance(n,ast.If)
               and ast.unparse(n.test)=='self.config.actor.router_replay.mode == \'R3\'')
    config=NS(num_layers=48,num_moe_experts=128,moe_router_topk=8,moe_router_pre_softmax=False,
              moe_router_score_function='softmax',moe_aux_loss_coeff=aux,moe_z_loss_coeff=z,
              moe_router_load_balancing_type=balance,enable_routing_replay=True)
    ns={'self':NS(config=NS(actor=NS(router_replay=NS(mode='R3'))),tf_config=config,rank=1,hf_config=NS(model_type='qwen3_moe')),
        'override_transformer_config':dict(moe_aux_loss_coeff=0.,moe_z_loss_coeff=0.,moe_router_load_balancing_type='none')}
    code=compile(ast.Module(body=[block],type_ignores=[]),'built-config','exec')
    if (aux,z,balance)==(0.,0.,'none'):
        exec(code,ns)
        assert ns['r3_config']['moe_router_pre_softmax'] is False
        assert ns['r3_config']['enable_routing_replay'] is True
    else:
        with pytest.raises(RuntimeError,match='not applied'):exec(code,ns)


def test_r3_generic_phase_load_remains_native_hdo_and_not_selective():
    tree=ast.parse((ROOT/'verl/workers/megatron_workers.py').read_text(encoding='utf-8'))
    methods={n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in {
        '_preserve_hdo_optimizer_residency','_selective_optimizer_copy_param_offload_enabled',
        '_load_actor_optimizer_for_update','_offload_actor_optimizer'}}
    for node in methods.values():node.decorator_list=[]
    calls=[]
    ns={'load_megatron_optimizer':lambda x:calls.append('native_onload'),
        'offload_megatron_optimizer':lambda x:calls.append('native_offload')}
    exec(compile(ast.Module(body=list(methods.values()),type_ignores=[]),'native-residency','exec'),ns)
    actor=NS(config=NS(actor=NS(router_replay=NS(mode='R3'),eu_derpo=NS(enabled=False))),
             _is_offload_optimizer=True,_hdo_optimizer_residency_preserved=False,
             _optimizer_copy_params_offloaded_for_rollout=False,actor_optimizer=object())
    for _ in range(2):
        assert not ns['_preserve_hdo_optimizer_residency'](actor)
        assert not ns['_selective_optimizer_copy_param_offload_enabled'](actor)
        assert ns['_load_actor_optimizer_for_update'](actor)
        ns['_offload_actor_optimizer'](actor)
    assert calls==['native_onload','native_offload']*2
    assert not actor._hdo_optimizer_residency_preserved

@pytest.mark.parametrize('shape,dtype,value',[( (3,47,8),torch.uint8,1),((3,48,7),torch.uint8,1),
                                             ((3,48,8),torch.float32,1),((3,48,8),torch.int64,-1),
                                             ((3,48,8),torch.uint8,128)])
def test_agent_rejects_invalid_shape_dtype_or_global_expert_id(shape,dtype,value):
    tree=ast.parse((ROOT/'verl/experimental/agent_loop/agent_loop.py').read_text(encoding='utf-8'))
    block=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
               and ast.unparse(n.test)=='output.routed_experts is not None')
    ns={'self':NS(model_config=NS(hf_config=NS(num_hidden_layers=48,num_experts_per_tok=8,num_experts=128))),
        'torch':torch,'np':__import__('numpy'),'input_ids':torch.zeros(1,5),
        'output':NS(routed_experts=torch.full(shape,value,dtype=dtype),prompt_ids=[1]),
        'prompt_output':{'input_ids':torch.zeros(1,1)}}
    with pytest.raises(ValueError):exec(compile(ast.Module(body=[block],type_ignores=[]),'validate-routes','exec'),ns)


@pytest.mark.parametrize('key,wrong', [
    ('num_layers',47),('num_moe_experts',64),('moe_router_topk',4),
    ('moe_router_pre_softmax',True),('moe_router_score_function','sigmoid'),
    ('enable_routing_replay',False)])
def test_built_qwen3_full_r3_rejects_wrong_geometry_or_replay(key,wrong):
    with pytest.raises(RuntimeError,match=key):
        execute_built_config(**{key:wrong})


def execute_built_config(mode='R3',model_type='qwen3_moe',**changes):
    tree=ast.parse((ROOT/'verl/workers/megatron_workers.py').read_text(encoding='utf-8'))
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_build_model_optimizer')
    block=next(n for n in ast.walk(fn) if isinstance(n,ast.If)
               and ast.unparse(n.test)=="self.config.actor.router_replay.mode == 'R3'")
    fields=dict(num_layers=48,num_moe_experts=128,moe_router_topk=8,moe_router_pre_softmax=False,
                moe_router_score_function='softmax',moe_aux_loss_coeff=0.,moe_z_loss_coeff=0.,
                moe_router_load_balancing_type='none',enable_routing_replay=True)
    fields.update(changes)
    config=NS(**fields)
    before=vars(config).copy()
    ns={'self':NS(config=NS(actor=NS(router_replay=NS(mode=mode))),tf_config=config,
                  hf_config=NS(model_type=model_type),rank=1),'override_transformer_config':{}}
    exec(compile(ast.Module(body=[block],type_ignores=[]),'built-config','exec'),ns)
    assert vars(config)==before  # Validate only; never mutate routing semantics.


def test_built_r3_accepts_enum_and_does_not_impose_qwen3_on_other_models_or_modes():
    from enum import Enum
    class Score(Enum):softmax='softmax'
    class Balance(Enum):none='none'
    execute_built_config(moe_router_score_function=Score.softmax,moe_router_load_balancing_type=Balance.none)
    execute_built_config(mode='disabled',num_layers=2,enable_routing_replay=False)
    execute_built_config(model_type='other_moe',num_layers=2,num_moe_experts=16,moe_router_topk=2)
