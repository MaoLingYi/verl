from __future__ import annotations

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace


TRAINER = Path(__file__).resolve().parents[3] / "verl" / "trainer" / "ppo" / "ray_trainer.py"
WORKER = Path(__file__).resolve().parents[3] / "verl" / "workers" / "megatron_workers.py"


def _save_decision(global_steps: int, save_freq: int, *, last=False, esi=False):
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"), filename=str(TRAINER))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    method = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_should_save_checkpoint_this_step"
    )
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(TRAINER), "exec"), namespace)
    trainer = SimpleNamespace(
        global_steps=global_steps,
        config=SimpleNamespace(trainer=SimpleNamespace(save_freq=save_freq)),
    )
    decision = MethodType(namespace["_should_save_checkpoint_this_step"], trainer)
    return decision(last, esi)


def test_checkpoint_decision_covers_frequency_final_and_esi():
    assert not _save_decision(49, 50)
    assert _save_decision(50, 50)
    assert _save_decision(49, 50, last=True)
    assert _save_decision(49, 50, esi=True)


def test_checkpoint_hold_uses_the_same_decision_and_has_driver_cleanup():
    source = TRAINER.read_text(encoding="utf-8")
    fit = source[source.index("    def fit(self):") :]
    decision = fit.index("will_save_checkpoint = self._should_save_checkpoint_this_step(")
    defer = fit.index('batch.meta_info["defer_phase_offload_for_checkpoint"] = will_save_checkpoint')
    update = fit.index("actor_output = self._update_actor(batch)")
    save = fit.index("self._save_checkpoint()", update)
    cleanup = fit.index("self.actor_rollout_wg.release_checkpoint_residency_hold()", save)
    update_weights = fit.index("self.checkpoint_manager.update_weights(self.global_steps)", cleanup)
    assert decision < defer < update < save < cleanup < update_weights


def test_rollout_memory_stage_precedes_weight_update():
    source = WORKER.read_text(encoding="utf-8")
    start = source.index("    async def rollout_mode(self):")
    rollout = source[start:source.index("    @register", start)]
    assert rollout.index('"before_rollout_update_weights"') < rollout.index("await self.rollout.update_weights")
    assert '"before_rollout_wakeup"' in rollout
    assert "EU_DERPO_HDO_GPU_HEADROOM_INSUFFICIENT" not in rollout
