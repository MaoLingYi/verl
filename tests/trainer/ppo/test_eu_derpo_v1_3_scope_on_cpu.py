from pathlib import Path


ROOT = Path(__file__).parents[3]


def source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_partial_replay_is_old_only_and_full_r3_paths_are_unchanged():
    actor = source("verl/workers/actor/megatron_actor.py")
    worker = source("verl/workers/megatron_workers.py")
    assert 'mode in {"R3", "R3_OLD_ONLY"}' in worker
    assert 'mode in {"R2", "R3"}' in actor
    assert 'mode in ["R2", "R3"]' in actor
    assert 'response_replay_mask[:, -response_length - 1 : -1] = batch["response_mask"].bool()' in actor
    assert "replay_token_mask = response_replay_mask & replay_token_mask.bool()" in actor
    replay_utils = source("verl/utils/megatron/router_replay_utils.py")
    assert "retain_for_backward=token_mask_split is None" in replay_utils
    assert 'select_keys.append("routed_experts")' in actor


def test_replay_injects_only_ids_and_recomputes_megatron_scores():
    replay = source("verl/utils/megatron/router_replay_patch.py")
    assert "target_topk_idx" in replay
    assert "target_token_mask" in replay
    assert "probs = scores.gather(1, top_indices)" in replay
    assert "probs = torch.softmax(scores" in replay
    for forbidden in ("target_logits", "target_probs", "target_alpha", "target_hidden", "target_output"):
        assert forbidden not in replay


def test_v13_payload_is_compacted_after_old_and_never_selected_by_current_minibatch():
    trainer = source("verl/trainer/ppo/ray_trainer.py")
    worker = source("verl/workers/megatron_workers.py")
    actor = source("verl/workers/actor/megatron_actor.py")
    assert 'batch.batch.pop("routed_experts")' in trainer
    assert 'batch.batch["eu_derpo_rollout_routes"] = routes' in trainer
    assert 'data.batch.pop("eu_derpo_rollout_routes").cpu()' in worker
    iterator = actor[actor.index("def make_minibatch_iterator") : actor.index("def forward_backward_batch")]
    assert "eu_derpo_rollout_routes" not in iterator
