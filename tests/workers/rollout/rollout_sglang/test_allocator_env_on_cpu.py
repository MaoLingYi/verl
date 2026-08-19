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


SERVER = (
    Path(__file__).resolve().parents[4]
    / "verl"
    / "workers"
    / "rollout"
    / "sglang_rollout"
    / "async_sglang_server.py"
)


def _server_env_vars():
    tree = ast.parse(SERVER.read_text(encoding="utf-8"), filename=str(SERVER))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_sglang_server_env_vars"
    )
    namespace = {"visible_devices_keyword": "CUDA_VISIBLE_DEVICES"}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SERVER), "exec"), namespace)
    return namespace["_sglang_server_env_vars"]()


class TestSGLangAllocatorEnv(unittest.TestCase):
    def test_child_override_disables_expandable_segments_and_preserves_parent_env(self):
        parent_env = {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "PATH": "/env/bin:/usr/bin",
            "LD_LIBRARY_PATH": "/env/lib",
            "CUDA_HOME": "/usr/local/cuda",
            "NINJA_STATUS": "[%f/%t] ",
            "NCCL_DEBUG": "WARN",
        }

        child_env = parent_env | _server_env_vars()

        self.assertEqual(parent_env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        self.assertEqual(parent_env["PYTORCH_ALLOC_CONF"], "expandable_segments:True")
        self.assertEqual(child_env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:False")
        self.assertEqual(child_env["PYTORCH_ALLOC_CONF"], "expandable_segments:False")
        for name in ("CUDA_VISIBLE_DEVICES", "PATH", "LD_LIBRARY_PATH", "CUDA_HOME", "NINJA_STATUS", "NCCL_DEBUG"):
            self.assertEqual(child_env[name], parent_env[name])
        self.assertEqual(child_env["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"], "1")

    def test_sglang_actor_creation_uses_isolated_env(self):
        source = SERVER.read_text(encoding="utf-8")

        self.assertIn('runtime_env={"env_vars": _sglang_server_env_vars()}', source)


if __name__ == "__main__":
    unittest.main()
