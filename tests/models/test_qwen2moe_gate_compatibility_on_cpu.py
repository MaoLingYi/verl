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
import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def _method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def _base_config_keywords(function: ast.FunctionDef) -> dict[str, ast.expr]:
    call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_get_base_transformer_config"
    )
    return {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None}


class TestQwen2MoEGateCompatibility(unittest.TestCase):
    initializer_path = REPO_ROOT / "verl" / "models" / "mcore" / "model_initializer.py"
    converter_path = REPO_ROOT / "verl" / "models" / "mcore" / "config_converter.py"

    def test_qwen2moe_layer_spec_uses_config_with_legacy_fallback(self):
        method = _method(self.initializer_path, "Qwen2MoEModel", "get_transformer_layer_spec")
        fallback = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and isinstance(node.test.operand, ast.Call)
            and isinstance(node.test.operand.func, ast.Name)
            and node.test.operand.func.id == "hasattr"
            and isinstance(node.test.operand.args[1], ast.Constant)
            and node.test.operand.args[1].value == "moe_shared_expert_gate"
        )
        self.assertTrue(
            any(
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "params"
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == "gate"
                for node in ast.walk(fallback)
            )
        )

    def test_qwen2moe_config_enables_shared_expert_gate(self):
        keywords = _base_config_keywords(_function(self.converter_path, "hf_to_mcore_config_qwen2moe"))
        self.assertIsInstance(keywords["moe_shared_expert_gate"], ast.Constant)
        self.assertIs(keywords["moe_shared_expert_gate"].value, True)

    def test_qwen2moe_config_preserves_moe_contract(self):
        keywords = _base_config_keywords(_function(self.converter_path, "hf_to_mcore_config_qwen2moe"))
        expected_attributes = {
            "moe_shared_expert_intermediate_size": "shared_expert_intermediate_size",
            "num_moe_experts": "num_experts",
            "moe_router_topk": "num_experts_per_tok",
        }
        for keyword, attribute in expected_attributes.items():
            self.assertIsInstance(keywords[keyword], ast.Attribute)
            self.assertEqual(keywords[keyword].attr, attribute)
        self.assertIs(keywords["moe_shared_expert_overlap"].value, True)

    def test_other_moe_initializers_do_not_patch_shared_expert_gate(self):
        for class_name in ("MixtralModel", "Qwen3MoEModel"):
            method = _method(self.initializer_path, class_name, "get_transformer_layer_spec")
            self.assertNotIn("moe_shared_expert_gate", ast.unparse(method))
            self.assertNotIn('params["gate"]', ast.unparse(method))

    @unittest.skipUnless(importlib.util.find_spec("megatron") is not None, "Megatron-Core is not installed")
    def test_mcore_build_module_receives_gate_once(self):
        from megatron.core.transformer.spec_utils import ModuleSpec, build_module

        class SharedExpertProbe:
            def __init__(self, config, gate):
                self.config = config
                self.gate = gate

        config = object()
        module = build_module(ModuleSpec(module=SharedExpertProbe, params={}), config=config, gate=True)
        self.assertIs(module.config, config)
        self.assertIs(module.gate, True)


if __name__ == "__main__":
    unittest.main()
