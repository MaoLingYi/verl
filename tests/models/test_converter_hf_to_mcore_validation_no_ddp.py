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
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "converter_hf_to_mcore.py"


class _Tensor:
    comparisons = 0

    def __init__(self, values, shape=(2,)):
        self.values = values
        self.shape = shape
        self.data = self

    def __eq__(self, other):
        type(self).comparisons += 1
        return SimpleNamespace(all=lambda: self.values == other.values)


class _Module:
    def __init__(self, state_dict, sharded_state_dict=None):
        self._state_dict = state_dict
        self._sharded_state_dict = sharded_state_dict

    def state_dict(self):
        return self._state_dict

    def sharded_state_dict(self):
        return self._sharded_state_dict


class _Float16ModuleLike:
    def __init__(self, module):
        self.module = module


def _function_node(name):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _load_test_conversion(get_model, dist_checkpointing):
    namespace = {
        "ModelType": SimpleNamespace(encoder_or_decoder=object()),
        "ShardedTensor": type("ShardedTensor", (), {}),
        "StrictHandling": SimpleNamespace(ASSUME_OK_UNEXPECTED=object()),
        "dist_checkpointing": dist_checkpointing,
        "get_model": get_model,
    }
    exec(compile(ast.Module(body=[_function_node("test_conversion")], type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace["test_conversion"], namespace


class TestConverterValidationWithoutDDP(unittest.TestCase):
    def _run(self, dut_values=(1, 2), ref_values=(1, 2)):
        ref_state_dict = {"weight": _Tensor(ref_values)}
        validation_model = _Float16ModuleLike(_Module({}, ref_state_dict))
        converted_model = _Float16ModuleLike(_Module({"weight": _Tensor(dut_values)}))
        get_model = Mock(return_value=[validation_model])
        dist_checkpointing = SimpleNamespace(load=Mock())
        test_conversion, namespace = _load_test_conversion(get_model, dist_checkpointing)
        _Tensor.comparisons = 0
        test_conversion("provider", "tfconfig", "output", [converted_model])
        return get_model, dist_checkpointing, ref_state_dict, namespace

    def test_validation_reloads_and_compares_without_ddp(self):
        get_model, dist_checkpointing, ref_state_dict, namespace = self._run()

        get_model.assert_called_once_with(
            model_provider_func="provider",
            model_type=namespace["ModelType"].encoder_or_decoder,
            wrap_with_ddp=False,
            transformer_config="tfconfig",
        )
        dist_checkpointing.load.assert_called_once_with(
            ref_state_dict,
            "output",
            strict=namespace["StrictHandling"].ASSUME_OK_UNEXPECTED,
        )
        self.assertGreaterEqual(_Tensor.comparisons, 2)

    def test_validation_still_rejects_different_tensor_values(self):
        with self.assertRaisesRegex(AssertionError, "weight is not equal"):
            self._run(ref_values=(3, 4))

    def test_main_conversion_model_remains_no_ddp(self):
        function = _function_node("convert_hf_to_mcore")
        get_model_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "get_model"
        ]
        self.assertEqual(len(get_model_calls), 1)
        wrap_with_ddp = next(keyword.value for keyword in get_model_calls[0].keywords if keyword.arg == "wrap_with_ddp")
        self.assertIs(wrap_with_ddp.value, False)


if __name__ == "__main__":
    unittest.main()
