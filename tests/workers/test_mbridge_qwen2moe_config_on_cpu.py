# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


WORKER = Path(__file__).resolve().parents[2] / "verl" / "workers" / "megatron_workers.py"


def _tree():
    return ast.parse(WORKER.read_text(encoding="utf-8"), filename=str(WORKER))


def _function(name):
    return next(node for node in _tree().body if isinstance(node, ast.FunctionDef) and node.name == name)


def _method(class_name, method_name):
    cls = next(node for node in _tree().body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def _normalizer():
    function = _function("_normalize_mbridge_qwen2moe_config")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(WORKER), "exec"), namespace)
    return namespace[function.name]


class TestMBridgeQwen2MoEConfig(unittest.TestCase):
    def test_qwen2moe_false_gate_is_normalized(self):
        config = SimpleNamespace(moe_shared_expert_gate=False)
        _normalizer()(SimpleNamespace(architectures=["Qwen2MoeForCausalLM"]), config)
        self.assertIs(config.moe_shared_expert_gate, True)

    def test_qwen2moe_requires_gate_support(self):
        with self.assertRaisesRegex(RuntimeError, "Qwen2MoE requires moe_shared_expert_gate support"):
            _normalizer()(SimpleNamespace(architectures=["Qwen2MoeForCausalLM"]), SimpleNamespace())

    def test_other_or_missing_architectures_are_unchanged(self):
        for architectures in (["Qwen3MoeForCausalLM"], ["MixtralForCausalLM"], ["Qwen2ForCausalLM"], None, []):
            with self.subTest(architectures=architectures):
                config = SimpleNamespace(moe_shared_expert_gate=False)
                _normalizer()(SimpleNamespace(architectures=architectures), config)
                self.assertIs(config.moe_shared_expert_gate, False)

    def test_normalization_follows_bridge_config_and_precedes_handoff(self):
        method = _method("MegatronWorker", "_init_hf_config_and_tf_config")
        bridge_config_line = next(
            node.lineno
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "tf_config" for target in node.targets)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "bridge"
            and node.value.attr == "config"
        )
        normalize_line = next(
            node.lineno
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_normalize_mbridge_qwen2moe_config"
        )
        handoff_line = next(
            node.lineno
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "tf_config"
                for target in node.targets
            )
        )
        self.assertLess(bridge_config_line, normalize_line)
        self.assertLess(normalize_line, handoff_line)


if __name__ == "__main__":
    unittest.main()
