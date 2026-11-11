# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
from contextlib import nullcontext
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/converter_hf_to_mcore.py"


class Model(torch.nn.Linear):
    def sharded_state_dict(self):
        return self.state_dict()


@pytest.mark.parametrize("cpu_init", [True, False])
@pytest.mark.parametrize("fault", [None, "value", "shape", "dtype", "missing", "extra", "load_error"])
def test_converter_reload(tmp_path, capsys, cpu_init, fault):
    original = Model(2, 1, bias=False)
    saved = original.state_dict()
    if fault == "value":
        saved["weight"] = saved["weight"] + 1
    elif fault == "shape":
        saved["weight"] = saved["weight"].flatten()
    elif fault == "dtype":
        saved["weight"] = saved["weight"].double()
    elif fault == "missing":
        del saved["weight"]
    elif fault == "extra":
        saved["router.weight"] = torch.zeros(1)
    saved["decoder.mlp._extra_state127"] = None
    path = tmp_path / "checkpoint"
    torch.save(saved, path)
    cfg = SimpleNamespace(use_cpu_initialization=cpu_init)
    def build(**kwargs):
        assert kwargs["wrap_with_ddp"] is False and kwargs["transformer_config"] is cfg
        module = Model(2, 1, bias=False)
        assert module.weight.is_meta == cpu_init
        return [SimpleNamespace(module=module)]

    def load(request, location, strict):
        assert request["weight"].device.type == "cpu"
        if fault == "load_error":
            raise RuntimeError("DCP load failed")
        # Return independent storage, so ignoring the loader result cannot pass.
        return torch.load(location, weights_only=True)

    loader = Mock(side_effect=load)
    namespace = dict(torch=torch, re=re, get_model=build, dist_checkpointing=SimpleNamespace(load=loader),
                     init_empty_weights=lambda: torch.device("meta"), noop_context=nullcontext,
                     ModelType=SimpleNamespace(encoder_or_decoder=0),
                     StrictHandling=SimpleNamespace(ASSUME_OK_UNEXPECTED=0))
    tree = ast.parse(SCRIPT.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"_conversion_model","test_conversion"}]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    ctx = pytest.raises(RuntimeError, match="DCP load failed") if fault == "load_error" else (
        pytest.raises(ValueError, match="weight") if fault else nullcontext())
    with ctx:
        namespace["test_conversion"]("provider", cfg, path, [SimpleNamespace(module=original)])
    loader.assert_called_once()
    assert ("Conversion test passed!" in capsys.readouterr().out) == (fault is None)
