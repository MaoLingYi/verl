"""Execute production startup control flow with mocked Ray boundaries, not native TLS/GPU."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

from omegaconf import OmegaConf
import pytest

SOURCE = Path(__file__).resolve().parents[2] / "verl/trainer/main_ppo.py"
EXPERIMENT = "grpo_r3_qwen3_30b_a3b_base_b64_n8_16k_pp1_tp2_ep8_etp1"


class ActorDeath(Exception):
    pass


def execute(*, ready_errors=(), run_error=None, mode="R3", experiment=EXPERIMENT,
            eu=False, custom=False, kill_error=None, nsight=False):
    events = []
    handles = []
    options = []
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_ppo")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TaskRunner")
    # Execute the real constructor and ready method; training resources are never constructed.
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in {"__init__", "startup_ready"}]

    class Factory:
        def options(self, **kwargs):
            options.append(kwargs)
            return NS(remote=self.remote)  # Ray's options wrapper does not support chained options().

        def remote(self):
            index = len(handles)
            events.append(("create", index))
            local = namespace["TaskRunner"]()

            def ready():
                events.append(("ready", index))
                assert local.role_worker_mapping == {} and local.mapping == {}
                assert local.startup_ready() is True
                error = ready_errors[index] if index < len(ready_errors) else None
                return NS(error=error)

            def train(config):
                events.append(("run", index))
                return NS(error=run_error)

            handle = NS(startup_ready=NS(remote=ready), run=NS(remote=train))
            handles.append(handle)
            return handle

    def get(ref):
        if ref.error is not None:
            raise ref.error
        return True

    def kill(handle, *, no_restart):
        assert no_restart is True
        events.append(("kill", handles.index(handle)))
        if kill_error is not None:
            raise kill_error

    factory = Factory()
    ray = NS(is_initialized=lambda: True, remote=lambda **kwargs: lambda cls: factory,
             get=get, kill=kill, exceptions=NS(RayActorError=ActorDeath))
    namespace = {"ray": ray, "OmegaConf": OmegaConf, "is_cuda_available": nsight}
    exec(compile(ast.Module(body=[cls, run], type_ignores=[]), str(SOURCE), "exec"), namespace)
    config = OmegaConf.create({
        "trainer": {"experiment_name": experiment},
        "actor_rollout_ref": {"actor": {"router_replay": {"mode": mode}, "eu_derpo": {"enabled": eu}}},
        "global_profiler": {"tool": "nsys" if nsight else None, "steps": [1],
                            "global_tool_config": {"nsys": {"controller_nsight_options": {"trace": "cuda"}}}},
        "ray_kwargs": {},
    })
    try:
        namespace["run_ppo"](config, task_runner_class=factory if custom else None)
    except Exception as exc:
        return events, options, exc
    return events, options, None


def test_first_startup_success_runs_once():
    events, options, error = execute()
    assert error is None
    assert events == [("create", 0), ("ready", 0), ("run", 0)]
    assert options == [{"max_restarts": 0, "max_task_retries": 0}]


@pytest.mark.parametrize("mode", ["R3", "disabled"])
def test_nsight_options_preserved_without_chained_actor_options(monkeypatch, mode):
    import sys
    monkeypatch.setitem(sys.modules, "verl.utils.import_utils", NS(is_nvtx_available=lambda: True))
    events, options, error = execute(mode=mode, nsight=True)
    assert error is None
    expected = {"runtime_env": {"nsight": {"trace": "cuda"}}}
    if mode == "R3":
        expected.update(max_restarts=0, max_task_retries=0)
    assert options == [expected]
    assert sum(event == "run" for event, _ in events) == 1


@pytest.mark.parametrize("kill_error", [None, ActorDeath("already dead")])
def test_first_ready_death_recreates_only_before_training(kill_error):
    events, _, error = execute(ready_errors=[ActorDeath("loader died")], kill_error=kill_error)
    assert error is None
    assert events == [("create", 0), ("ready", 0), ("kill", 0),
                      ("create", 1), ("ready", 1), ("run", 1)]


def test_two_startup_deaths_propagate_original_second_exception():
    second = ActorDeath("second loader failure")
    events, _, error = execute(ready_errors=[ActorDeath("first"), second])
    assert error is second
    assert events == [("create", 0), ("ready", 0), ("kill", 0), ("create", 1), ("ready", 1)]


@pytest.mark.parametrize("ready_errors", [(), (ActorDeath("startup"),)])
def test_training_actor_death_never_retries_run(ready_errors):
    failure = ActorDeath("may have updated optimizer already")
    events, _, error = execute(ready_errors=ready_errors, run_error=failure)
    assert error is failure
    assert sum(event == "run" for event, _ in events) == 1
    assert events[-1] == ("run", len(ready_errors))


def test_ready_application_error_is_not_worker_death_and_not_retried():
    failure = ValueError("not a worker death")
    events, _, error = execute(ready_errors=[failure])
    assert error is failure
    assert events == [("create", 0), ("ready", 0)]


@pytest.mark.parametrize("changes", [{"mode": "disabled"}, {"mode": "R3_OLD_ONLY"},
                                    {"experiment": "old_r3"}, {"eu": True}, {"custom": True}])
def test_other_modes_keep_original_no_ready_no_retry_path(changes):
    failure = ActorDeath("legacy failure")
    events, options, error = execute(run_error=failure, **changes)
    assert error is failure
    assert events == [("create", 0), ("run", 0)]
    assert options == []
