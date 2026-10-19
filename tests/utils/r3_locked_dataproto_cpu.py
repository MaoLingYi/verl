"""CPU-only real TensorDict regression: python -B -m tests.utils.r3_locked_dataproto_cpu.

No pytest, Ray cluster, CUDA allocation or MCore model forward. Missing dependencies FAIL, never SKIP.
"""
import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec',
                 flags=__import__('__future__').annotations.compiler_flag), namespace)
    return namespace[name]


def metadata_array(items):
    # Use the actual AgentLoop producer: {} is not None and is retained as an object-array row.
    tree = ast.parse((ROOT/'verl/experimental/agent_loop/agent_loop.py').read_text(encoding='utf-8'))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_postprocess')
    start = next(i for i,n in enumerate(fn.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == 'multi_modal_inputs_list')
    ns = {'inputs':[NS(multi_modal_inputs=item) for item in items], 'np':np, 'non_tensor_batch':{}}
    exec(compile(ast.Module(body=fn.body[start:start+2], type_ignores=[]),'metadata-producer','exec'),ns)
    return ns['non_tensor_batch']['multi_modal_inputs']


def run_full_actor_batch_on_cpu(data, mode, pp, forward_only=True, *, work_batches=None):
    """Run the whole method with real containers and mocked model/schedule/replay installation.

    forward_only=False checks forward preparation/retention flags, not the loss closure or backward.
    """
    import sys
    import time
    from functools import partial
    from types import ModuleType
    from verl.utils.memory_utils import log_r3_cuda_phase
    from tensordict import TensorDict
    tree=ast.parse((ROOT/'verl/workers/actor/megatron_actor.py').read_text(encoding='utf-8'))
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='forward_backward_batch')
    fn.decorator_list=[]
    seen=[]; replay=[]; broadcast=[]
    def model_forward(**kwargs):
        assert kwargs['attention_mask'].dtype==torch.bool
        assert kwargs['position_ids'].ndim==2
        assert kwargs['logits_processor_args']['label_mask'].dtype==torch.bool
        seen.append(kwargs)
        return {'log_probs':torch.zeros_like(kwargs['input_ids'],dtype=torch.float32)}
    mcore=ModuleType('verl.models.mcore')
    mcore.get_mcore_forward_fn=lambda config:model_forward
    mcore.get_mcore_forward_fused_fn=lambda config:model_forward
    def schedule(**kwargs):
        outputs=[]
        for _ in range(kwargs['num_microbatches']):
            output,loss_fn=kwargs['forward_step_func'](kwargs['data_iterator'],kwargs['model'][0])
            if forward_only:
                _,stats=loss_fn(output)
                outputs.append(stats)
        return outputs
    def install(routes,mask,*a,**kwargs):
        assert mask.dtype==torch.bool
        replay.append((routes,mask,kwargs['retain_for_backward']))
    class Config(NS):
        def get(self,key,default=None):return getattr(self,key,default)
    owner=NS(config=Config(router_replay=NS(mode=mode),megatron=NS(use_remove_padding=True),
                           clip_ratio=.2,entropy_coeff=0.,clip_ratio_c=3.),
             router_shift_observer=None,eu_derpo_observer=None,tf_config=NS(),hf_config=NS(),
             actor_module=[NS()],actor_optimizer=None,use_fused_kernels=False,mtp_config=None)
    created=[]
    def local_data(**kwargs):
        local=type(data)(**kwargs)
        created.append(local)
        return local
    ns={'torch':torch,'time':time,'partial':partial,'get_device_id':lambda:'cpu',
        'DataProto':local_data,'TensorDict':TensorDict,
        'mpu':NS(get_pipeline_model_parallel_world_size=lambda:pp,
                 get_pipeline_model_parallel_last_rank=lambda:0,get_pipeline_model_parallel_group=lambda:None),
        'broadcast_dict_tensor':lambda batch,**kwargs:broadcast.append(batch),
        'get_forward_backward_func':lambda:schedule,'unwrap_model':lambda model:model,
        'make_batch_generator':function('verl/utils/megatron/pipeline_parallel.py','make_batch_generator',{}),
        'set_router_replay_data':install,
        'RouterReplayHelper':NS(is_replay_backward_action=lambda *a:False,
                               is_replay_forward_action=lambda *a:mode=='R3',
                               is_r2_record_action=lambda *a:False,get_micro_batch_router_list=lambda *a:[])}
    ns['log_r3_cuda_phase'] = log_r3_cuda_phase
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'full-actor-batch','exec',
                 flags=__import__('__future__').annotations.compiler_flag),ns)
    # Let the production import use the real model utilities and their complete module globals.
    with patch.dict(sys.modules,{'verl.models.mcore':mcore}):
        result=ns['forward_backward_batch'](owner,data,forward_only=forward_only,
                                           use_dynamic_bsz=False,micro_batch_size=1)
    if work_batches is not None:work_batches.extend(created)
    assert len(seen)==len(data)
    assert len(broadcast)==(0 if mode=='R3' and pp==1 else 1)
    return seen,replay,result


def real_actor_data(types, three_dim_positions=False):
    TensorDict,DataProto=types
    rows,seq_len=4,8
    mask=torch.tensor([[0,1,1,1,1,1,1,0],[1,1,1,1,1,0,0,0],
                       [0,0,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]],dtype=torch.int64)
    positions=mask.cumsum(-1)-1
    if three_dim_positions:positions=positions[:,None,:].expand(-1,3,-1).clone()
    input_ids=torch.arange(rows*seq_len).reshape(rows,seq_len)
    routes=(torch.arange(rows)[:,None,None,None]*17+torch.arange(seq_len)[None,:,None,None]*9+
            torch.arange(48)[None,None,:,None]*3+torch.arange(8)[None,None,None,:])%128
    batch=TensorDict({'input_ids':input_ids,
                     'responses':input_ids[:,-3:].clone(),
                     'attention_mask':mask,'position_ids':positions,
                     'routed_experts':routes.to(torch.uint8)},
                    batch_size=[rows],device='cpu',names=['sample'])
    return DataProto(batch=batch,meta_info={'temperature':1.})


class LockedMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import inspect
        import tensordict
        from verl.protocol import DataProto
        if tensordict.__version__ != '0.10.0':
            raise RuntimeError('This runtime gate requires TensorDict 0.10.0, got '+tensordict.__version__)
        if Path(inspect.getfile(DataProto)).resolve() != ROOT/'verl/protocol.py':
            raise RuntimeError('DataProto must come from this baseline worktree')
        cls.types=(tensordict.TensorDict,DataProto)
        print('REAL_TYPES: TensorDict='+tensordict.__version__+' Torch='+torch.__version__+
              ' DataProto='+inspect.getfile(DataProto),flush=True)

    def metadata(self,kind):
        if kind=='empty':return metadata_array([{} for _ in range(4)])
        if kind=='nonempty':
            return metadata_array([
                {'pixel_values':torch.tensor([[10.,11.]])},
                {},
                {'pixel_values':torch.tensor([[20.,21.],[22.,23.]])},
                {'pixel_values':torch.tensor([[30.,31.]])},
            ])
        return None

    def check_path(self,kind,representation='locked',forward_only=True,three_dim=False):
        import pickle
        data=real_actor_data(self.types,three_dim)
        data.non_tensor_batch['sample_marker']=np.arange(len(data))
        payload=self.metadata(kind)
        if payload is not None:data.non_tensor_batch['multi_modal_inputs']=payload
        if representation=='consolidated':data.batch=data.batch.consolidate()
        elif representation=='serialized':data=pickle.loads(pickle.dumps(data))
        data.batch.lock_()
        original=data.batch
        original_non_tensor=data.non_tensor_batch
        original_meta=data.meta_info.copy()
        payload=data.non_tensor_batch.get('multi_modal_inputs')
        routes=original['routed_experts'];route_values=routes.clone()
        mask=original['attention_mask'];mask_values=mask.clone()
        positions=original['position_ids'];position_values=positions.clone()
        for _ in range(2):
            work=[]
            seen,replay,_=run_full_actor_batch_on_cpu(data,'R3',1,forward_only=forward_only,work_batches=work)
            self.assertIs(data.batch,original)
            self.assertTrue(original.is_locked)
            self.assertIs(data.non_tensor_batch,original_non_tensor)
            self.assertEqual(data.meta_info,original_meta)
            self.assertIs(data.non_tensor_batch.get('multi_modal_inputs'),payload)
            self.assertNotIn('multi_modal_inputs',original.keys())
            self.assertNotIn('multi_modal_inputs_idx',original.keys())
            self.assertIs(original['routed_experts'],routes)
            self.assertIs(original['attention_mask'],mask)
            torch.testing.assert_close(routes,route_values)
            torch.testing.assert_close(mask,mask_values)
            torch.testing.assert_close(positions,position_values)
            self.assertEqual(len(work),int(payload is not None or three_dim))
            if work:
                local=work[0]
                self.assertIsNot(local,data)
                self.assertFalse(local.batch.is_locked)
                self.assertEqual(local.batch.batch_size,original.batch_size)
                self.assertEqual(local.batch.device,original.device)
                self.assertEqual(local.batch.names,original.names)
                self.assertIs(local.meta_info,data.meta_info)
                self.assertIsNot(local.non_tensor_batch,original_non_tensor)
                self.assertIs(local.non_tensor_batch['sample_marker'],original_non_tensor['sample_marker'])
                self.assertNotIn('multi_modal_inputs',local.batch.keys())
                self.assertNotIn('multi_modal_inputs_idx',local.batch.keys())
                self.assertNotIn('multi_modal_inputs',local.non_tensor_batch)
                self.assertEqual(local.batch['routed_experts'].data_ptr(),routes.data_ptr())
                self.assertEqual(local.batch['routed_experts'].untyped_storage().data_ptr(),
                                 routes.untyped_storage().data_ptr())
            self.assertEqual(len(replay),len(data))
            for row,(kwargs,(micro_routes,replay_mask,retain)) in enumerate(zip(seen,replay)):
                expected={} if payload is None else payload[row]
                actual=kwargs['multi_modal_inputs']
                self.assertEqual(set(actual),set(expected))
                for key,value in expected.items():torch.testing.assert_close(actual[key],value)
                self.assertEqual(micro_routes.data_ptr(),routes[row:row+1].data_ptr())
                self.assertEqual(micro_routes.untyped_storage().data_ptr(),routes.untyped_storage().data_ptr())
                torch.testing.assert_close(micro_routes,routes[row:row+1])
                torch.testing.assert_close(kwargs['attention_mask'],mask[row:row+1].bool())
                torch.testing.assert_close(replay_mask,mask[row:row+1].bool())
                expected_pos=positions[row:row+1,0] if three_dim else positions[row:row+1]
                torch.testing.assert_close(kwargs['position_ids'],expected_pos)
                self.assertIs(retain,not forward_only)
                self.assertEqual(micro_routes.shape,(1,8,48,8))

    def test_original_metadata_injection_fails_on_locked_batch(self):
        data=real_actor_data(self.types)
        data.non_tensor_batch['multi_modal_inputs']=self.metadata('empty')
        data.batch.lock_()
        with self.assertRaisesRegex(RuntimeError,'[Ll]ocked'):
            data.batch['multi_modal_inputs']=data.non_tensor_batch['multi_modal_inputs']

    def test_empty_text_metadata_split_cleanup_and_repeat(self):
        for representation in ['locked','consolidated','serialized']:
            with self.subTest(representation=representation):self.check_path('empty',representation)

    def test_nonempty_metadata_preserves_global_sample_indices(self):
        for representation in ['locked','consolidated','serialized']:
            with self.subTest(representation=representation):self.check_path('nonempty',representation)
        from verl.utils.model import extract_multi_modal_inputs
        payload=self.metadata('nonempty')
        # Variable image counts, empty rows and reordered global sample indices use the real helper.
        selected=extract_multi_modal_inputs(payload,indices=[2,1,0,3])
        torch.testing.assert_close(selected['pixel_values'],
                                   torch.tensor([[20.,21.],[22.,23.],[10.,11.],[30.,31.]]))
        ragged=[{'pixel_values':torch.tensor([[1.,2.]]),'image_bound':torch.tensor([[0,1]])},
                {'pixel_values':torch.tensor([[3.,4.,5.]]),'image_bound':torch.tensor([[1,2]])}]
        # This fork retains image-bound payloads as lists rather than padding their last dimension.
        selected=extract_multi_modal_inputs(ragged,indices=[1,0])
        for key in ['pixel_values','image_bound']:
            self.assertIsInstance(selected[key],list)
            for actual,row in zip(selected[key],[1,0]):self.assertIs(actual,ragged[row][key])

    def test_training_mode_forward_preparation_only(self):
        # No policy loss closure, backward or optimizer step is run by the mocked schedule.
        self.check_path('nonempty',forward_only=False)

    def test_no_metadata_locked_path_still_shares_original(self):
        self.check_path(None)

    def test_three_dimensional_positions_are_local_not_a_vlm_forward_test(self):
        self.check_path('nonempty',three_dim=True)

    def test_legacy_writable_inputs_keep_existing_behavior(self):
        for mode,pp in [('disabled',1),('R3',4)]:
            for three_dim in [False,True]:
                with self.subTest(mode=mode,pp=pp,three_dim=three_dim):
                    data=real_actor_data(self.types,three_dim)
                    self.assertFalse(data.batch.is_locked)
                    seen,_,_=run_full_actor_batch_on_cpu(data,mode,pp)
                    self.assertEqual(len(seen),len(data))
                    self.assertEqual(data.batch['attention_mask'].dtype,torch.bool)
                    self.assertEqual(data.batch['position_ids'].ndim,2)


if __name__ == '__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(LockedMetadataTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print('REAL_TENSORDICT_METADATA_CPU='+('PASS' if result.wasSuccessful() else 'FAIL'),flush=True)
    sys.exit(0 if result.wasSuccessful() else 1)
