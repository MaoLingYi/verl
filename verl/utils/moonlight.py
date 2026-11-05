"""Moonlight's native MLA and routed-expert representation; no training lifecycle changes."""
import json
from pathlib import Path
import types
import torch


def is_moonlight(config):
    return (getattr(config, 'model_type', None) == 'deepseek_v3'
            and getattr(config, 'num_hidden_layers', None) == 27
            and getattr(config, 'n_routed_experts', None) == 64)


def _apply_with_fp32_bias(self, fn, recurse=True):
    bias = self.expert_bias
    result = torch.nn.Module._apply(self, fn, recurse=recurse)
    target = self.expert_bias
    # Float16Module must not round the correction bias before DCP load/save.
    self.expert_bias = (target.float() if bias.device.type == 'meta'
                        else bias.to(device=target.device, dtype=torch.float32))
    return result


def preserve_expert_bias(module):
    for child in module.modules():
        if getattr(child, 'expert_bias', None) is not None:
            child.expert_bias = child.expert_bias.float()
            child._apply = types.MethodType(_apply_with_fp32_bias, child)
            if getattr(child, 'router_replay', None) is not None:
                child.router_replay.strict = True


def restore_hf_expert_bias(model, source):
    from safetensors import safe_open
    source = Path(source)
    index = json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
    for layer in range(1, 27):
        key = f'model.layers.{layer}.mlp.gate.e_score_correction_bias'
        with safe_open(str(source/index[key]), framework='pt', device='cpu') as file:
            bias = file.get_tensor(key)
        if bias.shape != (64,) or bias.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError('invalid Moonlight correction bias: ' + key)
        parameter = model.model.layers[layer].mlp.gate.e_score_correction_bias
        parameter.data = bias.to(device=parameter.device, dtype=torch.float32)


def validate_routes(routes):
    if routes.ndim != 3 or tuple(routes.shape[1:]) != (27, 6):
        raise ValueError('Moonlight R3 requires [tokens, 27 layers, 6 routed experts]')
    active = routes[:, 1:, :]  # SGLang retains a zero placeholder for dense layer 0.
    if active.is_floating_point() or active.is_complex() or active.dtype == torch.bool:
        raise ValueError('Moonlight routes must be integer expert ids')
    if active.numel() and (active.min() < 0 or active.max() >= 64):
        raise ValueError('Moonlight routed expert id outside [0,64)')
    if active.numel() and torch.any(active.sort(dim=-1).values.diff(dim=-1) == 0):
        raise ValueError('Moonlight top6 routes contain duplicate experts')
