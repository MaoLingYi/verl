# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import ast
import os
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace


TRAINER = Path(__file__).resolve().parents[2] / "verl" / "trainer" / "ppo" / "ray_trainer.py"


class ConfigNode(dict):
    __getattr__ = dict.__getitem__


def _production_method(name, namespace, *, stop_after_first_step=False):
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"), filename=str(TRAINER))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    method.decorator_list = []
    if name == "fit":
        method.body = [node for node in method.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        if stop_after_first_step:
            first_step = next(
                index
                for index, node in enumerate(method.body)
                if isinstance(node, ast.AugAssign)
                and isinstance(node.target, ast.Attribute)
                and node.target.attr == "global_steps"
            )
            method.body = method.body[: first_step + 1]
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(TRAINER), "exec"), namespace)
    return namespace[name]


def _config(*, stop_after_first_step=False):
    rollout = ConfigNode(
        agent={"agent_loop_manager_class": "tests.FakeAgentLoopManager"},
        checkpoint_engine="checkpoint-engine-config",
    )
    trainer = ConfigNode(
        project_name="test",
        experiment_name="test",
        logger=["console"],
        val_before_train=not stop_after_first_step,
        val_only=not stop_after_first_step,
    )
    return ConfigNode(actor_rollout_ref=ConfigNode(rollout=rollout), trainer=trainer)


def _mock_trainer(events, *, restored_step=None, restore_error=None, stop_after_first_step=False):
    class AgentLoopManager:
        @staticmethod
        def create(**kwargs):
            events.append("agent_create")
            return SimpleNamespace(rollout_replicas="replicas")

    class CheckpointManager:
        def __init__(self, **kwargs):
            events.append("checkpoint_create")

        def sleep_replicas(self):
            events.append("sleep")

        def update_weights(self, step):
            events.append(("update", step))

    init_method = _production_method(
        "_init_async_rollout_and_checkpoint_manager",
        {
            "load_class_from_fqn": lambda *args: AgentLoopManager,
            "omega_conf_to_dataclass": lambda config: config,
            "CheckpointEngineManager": CheckpointManager,
        },
    )
    fit_method = _production_method(
        "fit",
        {
            "OmegaConf": SimpleNamespace(to_container=lambda *args, **kwargs: {}),
            "Tracking": lambda **kwargs: SimpleNamespace(log=lambda **kwargs: None),
            "pprint": lambda *args, **kwargs: None,
            "tqdm": lambda **kwargs: None,
        },
        stop_after_first_step=stop_after_first_step,
    )
    instance = SimpleNamespace(
        config=_config(stop_after_first_step=stop_after_first_step),
        use_rm=False,
        reward_loop_manager=SimpleNamespace(reward_loop_workers=["reward-worker"]),
        actor_rollout_wg="actor-worker-group",
        _actor_rollout_resource_pool="actor-resource-pool",
        checkpoint_manager=None,
        train_dataloader=[object()],
        total_training_steps=370,
        _validate=lambda: {"validation": 1},
    )

    def load_checkpoint():
        events.append("load")
        if restore_error is not None:
            raise restore_error
        if restored_step is not None:
            instance.global_steps = restored_step

    instance._load_checkpoint = load_checkpoint
    instance._init_async_rollout_and_checkpoint_manager = MethodType(init_method, instance)
    instance.fit = MethodType(fit_method, instance)
    return instance


class TestResumeRolloutInitOrder(unittest.TestCase):
    def test_disable_and_auto_without_checkpoint_keep_fresh_path(self):
        load_checkpoint = _production_method(
            "_load_checkpoint",
            {
                "os": os,
                "find_latest_ckpt_path": lambda path: None,
                "Role": SimpleNamespace(Critic="critic"),
                "torch": SimpleNamespace(load=lambda *args, **kwargs: None),
            },
        )

        for resume_mode in ("disable", "auto"):
            calls = []
            trainer = SimpleNamespace(
                config=ConfigNode(
                    trainer=ConfigNode(
                        resume_mode=resume_mode,
                        default_hdfs_dir=None,
                        default_local_dir="checkpoints",
                    )
                ),
                actor_rollout_wg=SimpleNamespace(load_checkpoint=lambda *args, **kwargs: calls.append(args)),
            )

            self.assertEqual(load_checkpoint(trainer), 0)
            self.assertEqual(calls, [])

    def test_only_explicit_resume_path_requests_staged_restore(self):
        fake_os = SimpleNamespace(
            getcwd=os.getcwd,
            path=SimpleNamespace(
                isabs=lambda path: True,
                join=os.path.join,
                exists=lambda path: path.endswith("data.pt"),
            ),
        )
        load_checkpoint = _production_method(
            "_load_checkpoint",
            {
                "os": fake_os,
                "find_latest_ckpt_path": lambda path: os.path.join(path, "global_step_100"),
                "Role": SimpleNamespace(Critic="critic"),
                "torch": SimpleNamespace(load=lambda *args, **kwargs: {"cursor": 100}),
            },
        )

        for resume_mode, expected_staged in (("auto", False), ("resume_path", True)):
            calls = []
            dataloader_states = []
            trainer_config = ConfigNode(
                resume_mode=resume_mode,
                default_hdfs_dir=None,
                default_local_dir="checkpoints",
                resume_from_path="checkpoints/global_step_100",
                del_local_ckpt_after_load=False,
            )
            trainer = SimpleNamespace(
                config=ConfigNode(trainer=trainer_config),
                actor_rollout_wg=SimpleNamespace(
                    load_checkpoint=lambda *args, **kwargs: calls.append((args, kwargs))
                ),
                use_critic=False,
                train_dataloader=SimpleNamespace(load_state_dict=dataloader_states.append),
            )

            load_checkpoint(trainer)

            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0][1]["staged_restore"], expected_staged)
            self.assertEqual(dataloader_states, [{"cursor": 100}])
            self.assertEqual(trainer.global_steps, 100)

    def test_fresh_run_keeps_rollout_initialization_before_checkpoint_load(self):
        events = ["actor_init"]
        trainer = _mock_trainer(events)

        trainer._init_async_rollout_and_checkpoint_manager(trainer._actor_rollout_resource_pool)
        trainer.fit()

        self.assertEqual(
            events,
            ["actor_init", "agent_create", "checkpoint_create", "sleep", "load", ("update", 0)],
        )
        self.assertEqual(events.count("agent_create"), 1)

    def test_resume_restores_before_rollout_and_updates_restored_step(self):
        events = ["actor_init"]
        trainer = _mock_trainer(events, restored_step=100, stop_after_first_step=True)

        trainer.fit()

        self.assertEqual(
            events,
            ["actor_init", "load", "agent_create", "checkpoint_create", "sleep", ("update", 100)],
        )
        self.assertEqual(events.count("agent_create"), 1)
        self.assertEqual(trainer.global_steps, 101)

    def test_restore_failure_does_not_create_rollout(self):
        events = ["actor_init"]
        trainer = _mock_trainer(events, restore_error=RuntimeError("checkpoint restore failed"))

        with self.assertRaisesRegex(RuntimeError, "checkpoint restore failed"):
            trainer.fit()

        self.assertEqual(events, ["actor_init", "load"])

    def test_init_workers_only_defers_explicit_resume_path(self):
        source = TRAINER.read_text(encoding="utf-8")
        init_workers = source[
            source.index("    def init_workers(self):") : source.index("    def _save_checkpoint(self):")
        ]

        self.assertLess(
            init_workers.index("self.actor_rollout_wg.init_model()"), init_workers.index("defer_rollout_init")
        )
        self.assertIn('self.config.trainer.resume_mode == "resume_path"', init_workers)
        self.assertIn("bool(self.config.trainer.resume_from_path)", init_workers)
        self.assertEqual(source.count("AgentLoopManager.create("), 1)


if __name__ == "__main__":
    unittest.main()
