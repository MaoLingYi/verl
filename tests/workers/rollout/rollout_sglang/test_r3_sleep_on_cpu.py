"""Real async production control flow; SGLang RPCs and device memory are mocked."""
import ast
import asyncio
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[4]


def method(path, cls, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in owner.body if isinstance(n, ast.AsyncFunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec",
                 flags=__import__("__future__").annotations.compiler_flag), namespace)
    return namespace[name]


class Mode(Enum):
    HYBRID = 1
    COLOCATED = 2
    STANDALONE = 3


NS_GLOBALS = {"RolloutMode": Mode, "ReleaseMemoryOccupationReqInput": NS,
              "ResumeMemoryOccupationReqInput": NS, "asyncio": asyncio}
SERVER = "verl/workers/rollout/sglang_rollout/async_sglang_server.py"
WORKER = "verl/workers/rollout/sglang_rollout/sglang_rollout.py"
sleep = method(SERVER, "SGLangHttpServer", "sleep", dict(NS_GLOBALS))
wake = method(SERVER, "SGLangHttpServer", "wake_up", dict(NS_GLOBALS))
resume = method(WORKER, "ServerAdapter", "resume", {})
release = method(WORKER, "ServerAdapter", "release", {})
replica_sleep = method("verl/workers/rollout/replica.py", "RolloutReplica", "sleep", {"asyncio": asyncio})
all_sleep = method("verl/checkpoint_engine/base.py", "CheckpointEngineManager", "sleep_replicas",
                   {"asyncio": asyncio})


def objects(replay=True, mode=Mode.HYBRID, free=True, leader=0):
    events = []
    paused = set()
    config = NS(enable_rollout_routing_replay=replay, free_cache_engine=free)

    async def drop(tags):
        await asyncio.sleep(0)  # The acknowledgement is intentionally asynchronous.
        paused.update(tags)
        events.append(("sleep_done", tuple(tags)))

    async def recover(tags):
        assert len(tags) == len(set(tags))
        paused.difference_update(tags)
        events.append(("resume_done", tuple(tags)))

    async def drop_rpc(obj, _):
        await drop(obj.tags)

    async def recover_rpc(obj, _):
        await recover(obj.tags)

    async def noop():
        pass

    server = NS(node_rank=leader, config=config, rollout_mode=mode,
                tokenizer_manager=NS(release_memory_occupation=drop_rpc,
                                     resume_memory_occupation=recover_rpc, flush_cache=noop))
    worker = NS(config=config, _init_server_adapter=noop,
                device_mesh={"infer_tp": NS(get_local_rank=lambda: leader)},
                _engine=NS(release_memory_occupation=drop, resume_memory_occupation=recover))
    return server, worker, events, paused


@pytest.mark.parametrize("replay,free,mode,enabled", [
    (True, True, Mode.HYBRID, True),
    (True, True, Mode.COLOCATED, True),
    (False, True, Mode.HYBRID, False),
    (True, False, Mode.HYBRID, False),
    (True, True, Mode.STANDALONE, False),
])
def test_graph_memory_saver_enabled_before_server_subprocess_launch(replay, free, mode, enabled):
    # Execute the actual launch prefix, stopping before server/GPU initialization.
    tree = ast.parse((ROOT / SERVER).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SGLangHttpServer")
    launch = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "launch_server")
    boundary = next(i for i, n in enumerate(launch.body)
                    if isinstance(n, ast.If) and ast.unparse(n.test) == "self.nnodes > 1")
    launch.body = launch.body[:boundary]
    environment = {"SGLANG_MEMORY_SAVER_CUDA_GRAPH": "0"}
    namespace = {"os": NS(environ=environment), "RolloutMode": Mode}
    exec(compile(ast.Module(body=[launch], type_ignores=[]), SERVER, "exec"), namespace)
    server = NS(config=NS(enable_rollout_routing_replay=replay, free_cache_engine=free), rollout_mode=mode)
    asyncio.run(namespace["launch_server"](server))
    assert environment["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] == ("1" if enabled else "0")


@pytest.mark.parametrize("replay", [False, True])
def test_two_rollout_training_cycles_wait_for_sleep_and_pair_graph_resume(replay):
    async def run():
        server, worker, events, paused = objects(replay)
        replica = NS(servers=[NS(sleep=NS(remote=lambda: sleep(server)))])
        manager = NS(replicas=[NS(sleep=lambda: replica_sleep(replica))])
        for _ in range(2):
            # After all requests finish, the existing manager/replica chain must await the server ACK.
            await all_sleep(manager)
            assert events[-1][0] == "sleep_done"
            assert paused == ({"weights", "kv_cache", "cuda_graph"} if replay else {"weights", "kv_cache"})
            events.append(("actor_reload_and_update", ()))
            # The next weight sync does not need graph pools while actor weights are still resident.
            await resume(worker, ["weights"])
            assert ("cuda_graph" in paused) is replay
            events.append(("actor_offload", ()))
            tags = ["kv_cache"]
            await resume(worker, tags)
            assert tags == ["kv_cache"]  # Do not mutate a caller's tag list.
            assert paused == set()
            assert events[-1][1] == (("kv_cache", "cuda_graph") if replay else ("kv_cache",))
        assert sum(name == "actor_reload_and_update" for name, _ in events) == 2
    asyncio.run(run())


@pytest.mark.parametrize("mode", [Mode.COLOCATED, Mode.STANDALONE])
@pytest.mark.parametrize("replay", [False, True])
def test_server_sleep_wakeup_tag_pair_and_standalone_unchanged(mode, replay):
    async def run():
        server, _, events, paused = objects(replay, mode)
        await sleep(server)
        expected = {"kv_cache"} if mode == Mode.STANDALONE else {"weights", "kv_cache"}
        if replay and mode == Mode.COLOCATED:
            expected.add("cuda_graph")
        assert paused == expected
        await wake(server)
        assert paused == set()
        assert set(events[-1][1]) == expected
    asyncio.run(run())


@pytest.mark.parametrize("free,leader", [(False, 0), (True, 1)])
def test_disabled_or_nonleader_does_not_release_or_resume(free, leader):
    async def run():
        server, worker, events, _ = objects(free=free, leader=leader)
        await sleep(server)
        await release(worker)
        await resume(worker, ["kv_cache"])
        assert events == []
    asyncio.run(run())


def test_worker_release_pairs_resume_without_duplicate_graph_tag():
    async def run():
        _, worker, _, paused = objects()
        await release(worker)
        assert paused == {"weights", "kv_cache", "cuda_graph"}
        await resume(worker, ["weights", "kv_cache", "cuda_graph"])
        assert paused == set()
    asyncio.run(run())


def test_manager_waits_for_every_replica_acknowledgement():
    async def run():
        fast, _, _, _ = objects()
        slow, _, _, _ = objects()
        fast_done, slow_gate = asyncio.Event(), asyncio.Event()
        fast_release = fast.tokenizer_manager.release_memory_occupation
        slow_release = slow.tokenizer_manager.release_memory_occupation
        async def finish_fast(obj, unused):
            await fast_release(obj, unused)
            fast_done.set()
        async def finish_slow(obj, unused):
            await slow_gate.wait()
            await slow_release(obj, unused)
        fast.tokenizer_manager.release_memory_occupation = finish_fast
        slow.tokenizer_manager.release_memory_occupation = finish_slow
        replicas = [NS(servers=[NS(sleep=NS(remote=lambda server=server: sleep(server)))])
                    for server in (fast, slow)]
        manager = NS(replicas=[NS(sleep=lambda replica=replica: replica_sleep(replica))
                               for replica in replicas])
        pending = asyncio.create_task(all_sleep(manager))
        await asyncio.wait_for(fast_done.wait(), timeout=2)
        assert not pending.done()
        slow_gate.set()
        await pending
    asyncio.run(run())


def test_failed_sleep_propagates_before_actor_reload():
    async def run():
        server, _, _, _ = objects()
        failure = RuntimeError("scheduler refused memory release")
        async def fail(obj, _):
            raise failure
        server.tokenizer_manager.release_memory_occupation = fail
        replica = NS(servers=[NS(sleep=NS(remote=lambda: sleep(server)))])
        manager = NS(replicas=[NS(sleep=lambda: replica_sleep(replica))])
        with pytest.raises(RuntimeError) as caught:
            await all_sleep(manager)
            pytest.fail("actor must not reload before sleep completes")
        assert caught.value is failure
    asyncio.run(run())
