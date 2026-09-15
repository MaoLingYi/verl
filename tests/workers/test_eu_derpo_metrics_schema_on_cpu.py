from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
ACTOR = ROOT / "verl" / "workers" / "actor" / "megatron_actor.py"
WORKER = ROOT / "verl" / "workers" / "megatron_workers.py"
METRICS = ROOT / "verl" / "utils" / "metric" / "utils.py"
FUNCTIONAL = ROOT / "verl" / "utils" / "py_functional.py"


def _function(path, name, namespace=None):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {} if namespace is None else namespace
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def _reduce_metrics():
    class Metric:
        pass

    source = METRICS.read_text(encoding="utf-8")
    function = source[source.index("def reduce_metrics") : source.index("\n\nclass AggregationType")]
    namespace = {"np": np, "Metric": Metric}
    exec("from __future__ import annotations\n" + function, namespace)
    return namespace["reduce_metrics"]


def _real_observer_metrics():
    return {
        "route_mismatch_count": 0,
        "route_equal_fraction": 1.0,
        "layer_mismatch_count": [0, 0],
        "forward_recompute_mismatch_count": 0,
        "forward_recompute_equal_fraction": 1.0,
        "forward_recompute_layer_mismatch_count": [0, 0],
        "first_route_mismatch": {},
        "route_phase_execution": {
            "F": {"model_training": True, "grad_enabled": False},
            "R": {"model_training": True, "grad_enabled": True},
        },
        "gradient_hook_count": 2,
        "expected_gradient_hook_count": 2,
        "gradient_hook_count_by_layer": [1, 1],
        "utility_edge_count_by_layer": [8, 8],
        "weighted_center_max_abs": 0.0,
        "native_finalize_count": 1,
        "native_finalize_completed": 1.0,
        "planned_valid_rows": 16,
    }


def test_runtime_first_offender_is_first_route_mismatch():
    reduce_metrics = _reduce_metrics()
    worker_metrics = {}
    for key, value in _real_observer_metrics().items():
        if not isinstance(value, list):
            worker_metrics[f"actor/eu_derpo/{key}"] = [value]
    collect = _function(FUNCTIONAL, "list_of_dict_to_dict_of_list")
    collected = collect([worker_metrics, worker_metrics.copy()])
    offenders = [key for key, value in collected.items() if isinstance(value[0][0], dict)]
    assert offenders == [
        "actor/eu_derpo/first_route_mismatch",
        "actor/eu_derpo/route_phase_execution",
    ]
    with pytest.raises(TypeError, match="dict.*dict"):
        reduce_metrics(collected)


def test_update_actor_metrics_are_numeric_and_reducible():
    reduce_metrics = _reduce_metrics()
    append_metrics = _function(ACTOR, "_append_eu_derpo_scalar_metrics")
    validate = _function(WORKER, "_validate_actor_metrics_schema")
    metrics = {
        "actor/eu_derpo/objective_edppo": [0.25],
        "actor/eu_derpo/objective_utility": [0.5],
        "actor/eu_derpo/objective_total": [0.275],
    }
    append_metrics(
        metrics,
        _real_observer_metrics()
        | {"utility_objective": 0.5, "route_attribution": {"set_mismatch_by_layer": [0, 0]}},
    )
    metrics.update({"perf/mfu/actor": 0.4, "actor/lr": 1e-6})

    validate(metrics)
    assert not any(isinstance(value, dict) for value in metrics.values())
    assert not any(
        isinstance(item, dict)
        for value in metrics.values()
        if isinstance(value, list)
        for item in value
    )
    collect = _function(FUNCTIONAL, "list_of_dict_to_dict_of_list")
    collected = collect([metrics, metrics.copy()])
    for value in collected.values():
        np.mean(value)
    reduced = reduce_metrics(collected)
    assert reduced["actor/eu_derpo/objective_edppo"] == pytest.approx(0.25)
    assert reduced["actor/eu_derpo/route_equal_fraction"] == pytest.approx(1.0)
    assert reduced["actor/eu_derpo/gradient_hook_count_by_layer_1"] == pytest.approx(1.0)
    assert reduced["actor/eu_derpo/native_finalize_completed"] == pytest.approx(1.0)
    assert "actor/eu_derpo/first_route_mismatch" not in reduced
    assert "actor/eu_derpo/route_phase_execution" not in reduced
    assert "actor/eu_derpo/route_attribution" not in reduced


def test_scalar_filter_and_schema_guard_are_wired_to_update_actor_path():
    actor = ACTOR.read_text(encoding="utf-8")
    update_policy = actor[actor.index("    def update_policy(") :]
    worker = WORKER.read_text(encoding="utf-8")
    update_actor = worker[
        worker.index("    def update_actor(") : worker.index("    def _finish_actor_update_residency")
    ]
    assert "_append_eu_derpo_scalar_metrics(metrics, eu_metrics | auxiliary_metrics)" in update_policy
    assert "_validate_actor_metrics_schema(metrics)" in update_actor


def test_worker_schema_guard_reports_exact_offending_key():
    validate = _function(WORKER, "_validate_actor_metrics_schema")
    with pytest.raises(TypeError, match="actor/eu_derpo/first_route_mismatch.*element_type=dict"):
        validate({"actor/eu_derpo/first_route_mismatch": [{}]})
