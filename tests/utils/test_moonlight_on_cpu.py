import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as N
import pytest
import torch
from tests.utils.test_r3_v17_workload_on_cpu import function, replay_api

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('moonlight_helpers', ROOT/'verl/utils/moonlight.py')
helpers=importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)


def test_bias_values_remain_exact_through_float16module_cast_and_device_moves():
    router=torch.nn.Module(); router.register_buffer('expert_bias', torch.tensor([.123456789, -.987654321]))
    router.weight=torch.nn.Parameter(torch.ones(2,2))
    module=torch.nn.Sequential(router)
    original=router.expert_bias.clone()
    helpers.preserve_expert_bias(module)
    module.bfloat16().to('cpu')
    assert router.weight.dtype==torch.bfloat16
    assert router.expert_bias.dtype==torch.float32 and torch.equal(router.expert_bias, original)
    assert torch.equal(module.state_dict()['0.expert_bias'],original)


@pytest.mark.parametrize('bad',['range','duplicate','dtype','layer'])
def test_moonlight_route_capture_rejects_wrong_expert_or_layout(bad):
    routes=torch.arange(6).repeat(3,27,1)
    if bad=='range': routes[0,1,0]=64
    elif bad=='duplicate': routes[0,1,0]=1
    elif bad=='dtype': routes=routes.float()
    else: routes=routes[:,1:]
    with pytest.raises(ValueError): helpers.validate_routes(routes)


@pytest.mark.parametrize('rank',[0,1])
@pytest.mark.parametrize('retain',[False,True])
def test_mixed_dense_moe_tp2_install_and_sigmoid_full_replay(rank,retain):
    api=replay_api(); routers=[api['RouterReplay']() for _ in range(26)]
    routes=torch.arange(6).repeat(1,4,27,1).to(torch.uint8)
    routes[:,:,0]=0
    mask=torch.ones(1,4,dtype=torch.bool)
    ns={'torch':torch, 'device_name':'cpu',
        'preprocess_packed_seqs':function('verl/models/mcore/util.py','preprocess_packed_seqs',{
            'torch':torch,'PackedSeqParams':lambda **k:N(**k),
            'mpu':N(get_tensor_model_parallel_world_size=lambda:2,get_context_parallel_world_size=lambda:1,
                    get_context_parallel_rank=lambda:0)}),
        'scatter_to_sequence_parallel_region':lambda x:x.chunk(2)[rank],
        'get_current_rank_layer_info':lambda *a:{'start':0,'end':27},
        'RouterReplayHelper':N(get_micro_batch_router_list=lambda *a:routers),
        'is_moe_layer':lambda config,index:index!=0}
    install=function('verl/utils/megatron/router_replay_utils.py','set_router_replay_data',ns)
    install(routes,mask,N(num_layers=27),retain_for_backward=retain)
    target=routes[mask].chunk(2)[rank]
    for index,router in enumerate(routers):
        assert torch.equal(router.target_topk_idx,target[:,index+1].long())
        assert len(router.replay_backward_list)==int(retain)
    router=routers[0]; router.set_router_replay_action(api['RouterReplayAction'].REPLAY_FORWARD)
    logits=torch.randn(2,64,requires_grad=True); bias=torch.linspace(-.1,.1,64)
    probs,mapping=api['_patched_topk_routing_with_score_function'](
        logits,6,True,None,None,'sigmoid',bias,False,router,2.446)
    scores=logits.sigmoid().gather(1,target[:,1].long())
    expected=scores/scores.sum(-1,keepdim=True)*2.446
    torch.testing.assert_close(probs.gather(1,target[:,1].long()),expected)
    assert torch.equal(mapping.nonzero()[:,1].reshape(2,6),target[:,1].long())
    (probs*torch.arange(64)).sum().backward()
    assert logits.grad.abs().sum()>0
    if retain:
        router.set_router_replay_action(api['RouterReplayAction'].REPLAY_BACKWARD)
        again,_=api['_patched_topk_routing_with_score_function'](
            logits,6,True,None,None,'sigmoid',bias,False,router,2.446)
        torch.testing.assert_close(again,probs)
        assert not router.replay_backward_list
    api['RouterReplay'].clear_global_indices()
    assert all(r.target_topk_idx is None for r in routers)


def test_record_uses_native_sigmoid_correction_bias_top6():
    api=replay_api(); router=api['RouterReplay'](); router.set_router_replay_action(api['RouterReplayAction'].RECORD)
    logits=torch.linspace(-1,1,64).repeat(2,1); bias=torch.arange(64).float().flip(0)/10
    _,mask=api['_patched_topk_routing_with_score_function'](logits,6,True,None,None,'sigmoid',bias,False,router,2.446)
    expected=(logits.sigmoid()+bias).topk(6,-1).indices
    assert torch.equal(router.recorded_topk_idx,expected)
    assert mask.sum()==12


@pytest.mark.parametrize('action',['REPLAY_FORWARD','REPLAY_BACKWARD'])
def test_moonlight_strict_replay_never_falls_back_when_target_missing(action):
    api=replay_api(); router=api['RouterReplay'](); router.strict=True
    router.set_router_replay_action(getattr(api['RouterReplayAction'],action))
    with pytest.raises(RuntimeError,match='target is missing'):
        api['_patched_topk_routing_with_score_function'](torch.randn(2,64),6,True,None,None,'sigmoid',torch.zeros(64),False,router,2.446)


def test_real_27_layer_converter_and_inverse_cover_dense_shared64_experts_and_fp32_bias(monkeypatch):
    def module(**values):
        m=torch.nn.Module()
        for key,value in values.items(): setattr(m,key,value)
        return m
    def weight(rows,cols):
        m=torch.nn.Module(); m.weight=torch.nn.Parameter(torch.randn(rows,cols,dtype=torch.bfloat16)); return m
    def norm(size):
        m=torch.nn.Module(); m.weight=torch.nn.Parameter(torch.randn(size,dtype=torch.bfloat16)); return m
    def mlp(width): return module(gate_proj=weight(width,4),up_proj=weight(width,4),down_proj=weight(4,width))
    def grouped(rows,cols):
        m=torch.nn.Module()
        for i in range(64): m.register_parameter('weight'+str(i),torch.nn.Parameter(torch.empty(rows,cols,dtype=torch.bfloat16)))
        return m
    hf_layers=[]; mg_layers=[]
    for layer in range(27):
        attn=module(q_proj=weight(6,4),kv_a_proj_with_mqa=weight(3,4),kv_a_layernorm=norm(2),kv_b_proj=weight(8,2),o_proj=weight(4,4))
        mgattn=module(linear_q_proj=weight(6,4),linear_kv_down_proj=weight(3,4),linear_kv_up_proj=weight(8,2),linear_proj=weight(4,4))
        mgattn.linear_kv_up_proj.layer_norm_weight=torch.nn.Parameter(torch.empty(2,dtype=torch.bfloat16))
        if layer==0:
            hfmlp=mlp(6); mgmlp=module(linear_fc1=weight(12,4),linear_fc2=weight(4,6))
            mgmlp.linear_fc1.layer_norm_weight=torch.nn.Parameter(torch.empty(4,dtype=torch.bfloat16))
        else:
            gate=weight(64,4); gate.e_score_correction_bias=torch.nn.Parameter(torch.randn(64,dtype=torch.float32))
            hfmlp=module(gate=gate,experts=torch.nn.ModuleList([mlp(2) for _ in range(64)]),shared_experts=mlp(4))
            router=weight(64,4); router.register_buffer('expert_bias',torch.empty(64,dtype=torch.float32))
            mgmlp=module(router=router,experts=module(linear_fc1=grouped(4,4),linear_fc2=grouped(4,2)),
                         shared_experts=module(linear_fc1=weight(8,4),linear_fc2=weight(4,4)))
        hf_layers.append(module(input_layernorm=norm(4),post_attention_layernorm=norm(4),self_attn=attn,mlp=hfmlp))
        mg_layers.append(module(input_layernorm=norm(4),pre_mlp_layernorm=norm(4),self_attention=mgattn,mlp=mgmlp))
    # Dense layer post-norm lives on its fused fc1; avoid an extra fake parameter.
    del mg_layers[0].pre_mlp_layernorm
    hf=module(model=module(embed_tokens=weight(8,4),norm=norm(4),layers=torch.nn.ModuleList(hf_layers)),lm_head=weight(8,4))
    mg=module(embedding=module(word_embeddings=weight(8,4)),decoder=module(layers=torch.nn.ModuleList(mg_layers),final_layernorm=norm(4)),output_layer=weight(8,4))
    ns={'torch':torch,'mpu':N(get_pipeline_model_parallel_rank=lambda:0,get_pipeline_model_parallel_world_size=lambda:1,
         get_expert_model_parallel_rank=lambda:0,get_expert_model_parallel_world_size=lambda:1),'warnings':__import__('warnings')}
    def strict_zip(a,b,strict=False):
        if strict and len(a)!=len(b): raise ValueError('zip length mismatch')
        return zip(a,b)
    ns['zip']=strict_zip  # Local Python3.9 lacks zip(strict); server uses3.12.
    ns['safe_copy']=function('scripts/converter_hf_to_mcore.py','safe_copy',{'torch':torch})
    convert=function('scripts/converter_hf_to_mcore.py','convert_checkpoint_from_transformers_to_megatron_dpskv3',ns)
    count=convert(hf,mg,N(q_lora_rank=None,tie_word_embeddings=False),N(moe_grouped_gemm=True))
    assert count==sum(t.numel() for t in hf.state_dict().values())
    tree=ast.parse((ROOT/'verl/models/mcore/weight_converter.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='McoreToHFWeightConverterDpskv3')
    cls.bases=[]
    scope={'torch':torch}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'real-inverse','exec',flags=__import__('__future__').annotations.compiler_flag),scope)
    inverse=scope[cls.name](); exported={}
    for name,tensor in mg.state_dict().items():
        pieces=list(tensor.chunk(2,0)) if 'linear_fc1.weight' in name else [tensor]
        names,values=inverse.convert_param(name,pieces)
        exported.update(zip(names,values))
    original=hf.state_dict()
    assert set(exported)==set(original)
    assert all(torch.equal(original[name],value) for name,value in exported.items())
    assert exported['model.layers.26.mlp.gate.e_score_correction_bias'].dtype==torch.float32
    check_online_export_tp2_ep8(mg, hf, inverse, strict_zip, monkeypatch)


def check_online_export_tp2_ep8(mg, hf, inverse, strict_zip, monkeypatch):
    """Use real exporter/concat/name mapping with CPU rank tensors at collective boundaries."""
    import sys

    def selected(name):
        return '.layers.' not in name or '.layers.0.' in name or '.layers.1.' in name

    source = {k: v for k, v in mg.state_dict().items() if selected(k)}
    expected = {k: v for k, v in hf.state_dict().items() if selected(k)}
    cfg = N(model_type='deepseek_v3', num_hidden_layers=27, n_routed_experts=64,
            kv_lora_rank=2, qk_rope_head_dim=1, hidden_size=4, tie_word_embeddings=False)
    inverse.hf_config = cfg
    inverse.mcore_config = N(num_moe_experts=64)
    mpu = N(get_pipeline_model_parallel_rank=lambda: 0, get_pipeline_model_parallel_world_size=lambda: 1,
            get_pipeline_model_parallel_group=lambda: 'pp', get_expert_model_parallel_world_size=lambda: 8,
            get_expert_model_parallel_group=lambda: 'ep', get_expert_tensor_parallel_world_size=lambda: 1,
            get_expert_tensor_parallel_group=lambda: 'etp', get_tensor_model_parallel_world_size=lambda: 2,
            get_tensor_model_parallel_group=lambda: 'tp')
    monkeypatch.setitem(sys.modules, 'megatron.core', N(mpu=mpu, parallel_state=mpu))
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda group: 2)
    monkeypatch.setattr(torch.distributed, 'all_gather_object',
                        lambda object_list, obj, group: object_list.__setitem__(0, obj))
    utils = 'verl/utils/megatron_utils.py'
    tp = 'verl/utils/megatron/tensor_parallel.py'
    tp_namespace = {}
    is_parallel = function(tp, 'is_tensor_parallel_param', tp_namespace)
    partition_dim = function(tp, 'get_tensor_parallel_partition_dim', tp_namespace)
    ns = dict(torch=torch, zip=strict_zip, is_moonlight=helpers.is_moonlight, unwrap_model=lambda m: m,
              normalize_model_name=lambda name, *args: name, get_device_id=lambda: 'cpu',
              broadcast_str_from_megatron_pp=lambda name: name, broadcast_from_megatron_pp=lambda t: t,
              tp_utils=N(is_tensor_parallel_param=is_parallel, get_tensor_parallel_partition_dim=partition_dim))
    ns['default_tp_concat_fn'] = function(utils, 'default_tp_concat_fn', ns)
    export = function(utils, 'per_tensor_generator', ns)
    mapping = dict(qkv_layer_name='self_attention.linear_qkv.', gate_proj_layer_name='linear_fc1.')
    for rank in range(2):
        local, groups = {}, {}
        for name, full in source.items():
            if '.mlp.experts.' in name:
                prefix, index = name.rsplit('weight', 1)
                if int(index) >= 8:
                    continue
                parts = [source[f'{prefix}weight{ep * 8 + int(index)}'] for ep in range(8)]
                local[name] = parts[rank]
                groups[id(local[name])] = ('ep', parts)
                continue

            if name.endswith('linear_kv_down_proj.weight'):
                # TE 2.12 sets is_parallel=True, dim=0 even for duplicated TELinear.
                parts, dim = [full, full], 0
            elif name.endswith('linear_fc1.weight'):
                gate, up = full.chunk(2)
                parts = [torch.cat(pair) for pair in zip(gate.chunk(2), up.chunk(2))]
                dim = 0
            elif name.endswith(('linear_proj.weight', 'linear_fc2.weight')):
                dim = 1
                parts = full.chunk(2, dim)
            elif name.endswith(('linear_q_proj.weight', 'linear_kv_up_proj.weight',
                                'word_embeddings.weight', 'output_layer.weight')):
                dim = 0
                parts = full.chunk(2, dim)
            else:
                local[name] = full
                continue
            local[name] = parts[rank]
            local[name].tensor_model_parallel = True
            local[name].partition_dim = dim
            groups[id(local[name])] = ('tp', parts)

        def gather(output, tensor, group):
            required_group, pieces = groups[id(tensor)]
            assert group == required_group and len(output) == len(pieces)
            for dest, src in zip(output, pieces):
                dest.copy_(src)

        monkeypatch.setattr(torch.distributed, 'all_gather', gather)
        model = N(named_parameters=lambda: ((k, v) for k, v in local.items() if not k.endswith('expert_bias')),
                  state_dict=lambda: local)
        actual = dict(export([model], cfg, inverse, None, mapping))
        assert actual.keys() == expected.keys()
        for name, weight in expected.items():
            assert actual[name].shape == weight.shape, name
            assert actual[name].dtype == weight.dtype and torch.equal(actual[name], weight), name


def test_source_bias_restore_replaces_hf_bf16_rounding_without_changing_source(tmp_path,monkeypatch):
    import json,sys
    original=torch.linspace(-.123456789,.987654321,64)
    before=original.clone(); calls=[]
    class Reader:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def get_tensor(self,key): calls.append(key); return original.clone()
    monkeypatch.setitem(sys.modules,'safetensors',N(safe_open=lambda *a,**kw:Reader()))
    keys={f'model.layers.{i}.mlp.gate.e_score_correction_bias':'source.safetensors' for i in range(1,27)}
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':keys}))
    layers=[None]+[N(mlp=N(gate=N(e_score_correction_bias=torch.nn.Parameter(original.bfloat16())))) for _ in range(26)]
    helpers.restore_hf_expert_bias(N(model=N(layers=layers)),tmp_path)
    assert len(calls)==26
    for layer in layers[1:]:
        bias=layer.mlp.gate.e_score_correction_bias
        assert bias.dtype==torch.float32 and torch.equal(bias,original)
    assert torch.equal(original,before)


@pytest.fixture
def moonlight_hf_config():
    class Config(dict): __getattr__=dict.__getitem__
    return Config(model_type='deepseek_v3',num_hidden_layers=27,n_routed_experts=64,first_k_dense_replace=1,num_nextn_predict_layers=0,
        q_lora_rank=None,kv_lora_rank=512,qk_nope_head_dim=128,qk_rope_head_dim=64,v_head_dim=128,
        rope_theta=50000.,max_position_embeddings=8192,moe_intermediate_size=1408,n_shared_experts=2,
        num_experts_per_tok=6,routed_scaling_factor=2.446,aux_loss_alpha=.001)


def test_moonlight_no_mtp_postprocess(moonlight_hf_config, monkeypatch):
    pytest.importorskip("megatron.core")
    from megatron.core import tensor_parallel
    from megatron.core.models.gpt.gpt_model import GPTModel
    from tests.utils.r3_locked_dataproto_cpu import function as nested_function
    from megatron.core.transformer import MLATransformerConfig
    from verl.models.mcore import config_converter
    from verl.models.mcore.model_initializer import BaseModelInitializer, DeepseekV3Model

    # Run the native MLA constructor without a distributed actor/process group.
    def base_config(hf_config, dtype, **kwargs):
        return dict(num_layers=hf_config.num_hidden_layers, hidden_size=2048, num_attention_heads=16,
                    gated_linear_unit=True, activation_func=torch.nn.functional.silu, **kwargs)

    monkeypatch.setattr(config_converter, '_get_base_transformer_config', base_config)
    cfg = config_converter.hf_to_mcore_config_dpskv3(moonlight_hf_config, torch.bfloat16)
    assert isinstance(cfg, MLATransformerConfig)

    def build_model(self, **kwargs):
        assert 'mtp_block_spec' not in kwargs
        return torch.nn.Module()

    monkeypatch.setattr(BaseModelInitializer, 'initialize', build_model)
    DeepseekV3Model(cfg, moonlight_hf_config).initialize(freeze_moe_router=False)

    # Small CPU output head; exercise MCore's unmodified labels=None and loss branches.
    hidden = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3).div(24).requires_grad_()
    weight = torch.arange(15, dtype=torch.float32).reshape(3, 5).div(15)
    model = N(config=cfg, share_embeddings_and_output_weights=False, post_process=True,
              output_layer=lambda x, **kwargs: (x @ weight, None))
    model.compute_language_model_loss = GPTModel.compute_language_model_loss.__get__(model)
    monkeypatch.setattr(tensor_parallel, 'vocab_parallel_cross_entropy', lambda x, y:
                        torch.nn.functional.cross_entropy(x.flatten(0, 1), y.flatten(), reduction='none').view_as(y))
    args = dict(hidden_states=hidden, input_ids=None, position_ids=None, rotary_pos_emb=None,
                rotary_pos_cos=None, rotary_pos_sin=None)
    logits = GPTModel._postprocess(model, labels=None, **args)
    assert cfg.mtp_num_layers is None
    expected = hidden.transpose(0, 1) @ weight
    torch.testing.assert_close(logits, expected)
    labels = torch.tensor([[1, 2, 3, 4], [4, 0, 2, 1]])
    loss = GPTModel._postprocess(model, labels=labels, **args)
    log_probs = expected.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(loss, -log_probs)
    slice_responses = nested_function('verl/workers/actor/megatron_actor.py', 'compute_logprobs_fn', {})
    response_ids = labels[:, 1:3]
    actual = slice_responses({'log_probs': logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)},
                             {'responses': response_ids})
    torch.testing.assert_close(actual['log_probs'], log_probs[:, 1:3])
    loss.mean().backward()
    assert hidden.grad.abs().sum() > 0


@pytest.mark.parametrize('enabled',[True,False,None])
def test_actual_mla_config_arguments_are_native8192_and_fp32_router(enabled,moonlight_hf_config):
    tree=ast.parse((ROOT/'verl/models/mcore/config_converter.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='hf_to_mcore_config_dpskv3')
    # Only unavailable import/constructor boundaries are supplied; execute the actual config body.
    fn.body=[n for n in fn.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    mla=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_get_mla_transformer_config')
    cfg=moonlight_hf_config
    ns={'set_experimental_flag':lambda *a:None,'AttnBackend':N(fused='fused'),'apply_patch':lambda:None,
        'is_moonlight':helpers.is_moonlight,
        '_get_base_transformer_config':lambda **kw:{k:v for k,v in kw.items() if k not in ('hf_config','dtype')}}
    exec(compile(ast.Module(body=[mla,fn],type_ignores=[]),'real-mla-config','exec',flags=__import__('__future__').annotations.compiler_flag),ns)
    # Exercise the actual field filter with a config class that has no legacy replay field.
    from dataclasses import dataclass
    from functools import wraps
    import inspect
    import warnings

    @dataclass
    class BaseConfig:
        moe_layer_freq: object = None

    @dataclass
    class MLAConfig(BaseConfig):
        rotary_base: float = 0
        rope_type: str = ''
        rotary_scaling_factor: float = 0
        q_lora_rank: object = None
        moe_router_dtype: str = ''
        moe_router_bias_update_rate: float = -1
        num_moe_experts: int = 0
        moe_router_topk: int = 0

    class Router(torch.nn.Module):
        def __init__(self,config):
            super().__init__()
            self.config=config

    api=replay_api()
    patch=function('verl/utils/megatron/router_replay_patch.py','apply_router_replay_patch',
        dict(TransformerConfig=BaseConfig,TopKRouter=Router,RouterReplay=api['RouterReplay'],
             MoEAlltoAllTokenDispatcher=None,patched_routing=None,inspect=inspect,wraps=wraps))
    patch()
    ns['MLATransformerConfig']=MLAConfig
    check=function('verl/models/mcore/config_converter.py','check_and_construct_configs',
                   dict(torch=torch,warnings=warnings,mapping_string_to_attn_backend=lambda args:args))
    def construct(args,cls):
        assert args['max_position_embeddings']==8192
        return check(args,cls)
    ns['check_and_construct_configs']=construct
    overrides={} if enabled is None else {'enable_routing_replay':enabled}
    result=ns[fn.name](cfg,torch.bfloat16,**overrides)
    assert result.mtp_num_layers is None
    assert getattr(result,'enable_routing_replay',False) is bool(enabled)
    assert result.rotary_base==50000. and result.rope_type=='rope' and result.rotary_scaling_factor==1
    assert result.q_lora_rank is None and result.moe_layer_freq==[0]+[1]*26
    assert result.moe_router_dtype=='fp32' and result.moe_router_bias_update_rate==0
    assert result.num_moe_experts==64 and result.moe_router_topk==6
    layers=torch.nn.ModuleList(Router(result) if moe else torch.nn.Linear(1,1) for moe in result.moe_layer_freq)
    registered=api['RouterReplay'].router_instances
    assert len(registered)==(26 if enabled else 0)
    assert not hasattr(layers[0],'router_replay')
    assert [r.router_replay for r in list(layers)[1:]]==(registered if enabled else [None]*26)


def test_real_activation_checkpoint_recompute_consumes_backward_replay_once():
    from torch.utils.checkpoint import checkpoint
    api=replay_api(); router=api['RouterReplay'](); router.strict=True
    target=torch.arange(6).repeat(2,1)
    router.set_target_indices(target,retain_for_backward=True)
    seen=[]
    def forward(logits):
        action='REPLAY_BACKWARD' if torch.is_grad_enabled() else 'REPLAY_FORWARD'
        router.set_router_replay_action(getattr(api['RouterReplayAction'],action))
        probs,_=api['_patched_topk_routing_with_score_function'](logits,6,True,None,None,'sigmoid',torch.zeros(64),False,router,2.446)
        seen.append((action,router.replay_backward_list.copy()))
        return (probs*torch.arange(64)).sum()
    logits=torch.randn(2,64,requires_grad=True)
    checkpoint(forward,logits,use_reentrant=True).backward()
    assert [action for action,_ in seen]==['REPLAY_FORWARD','REPLAY_BACKWARD']
    assert len(seen[0][1])==1 and seen[1][1]==[]
    assert logits.grad.abs().sum()>0 and torch.isfinite(logits.grad).all()
    api['RouterReplay'].clear_global_indices()


def test_actual_agent_loop_rejects_absent_moonlight_capture():
    tree=ast.parse((ROOT/'verl/experimental/agent_loop/agent_loop.py').read_text(encoding='utf-8'))
    guard=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and 'Moonlight Full R3' in ast.unparse(n))
    ns={'self':N(model_config=N(hf_config=N(model_type='deepseek_v3',num_hidden_layers=27,n_routed_experts=64)),
                 rollout_config=N(enable_rollout_routing_replay=True)),
        'output':N(routed_experts=None),'is_moonlight':helpers.is_moonlight}
    with pytest.raises(ValueError,match='Moonlight Full R3'):
        exec(compile(ast.Module(body=[guard],type_ignores=[]),'actual-agent-loop-guard','exec'),ns)
