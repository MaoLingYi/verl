"""Node-wide quota and Linux accounting regressions using synthetic procfs/cgroupfs."""

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("eu_hidden_budget_test", ROOT / "verl/utils/debug/eu_hidden_budget.py")
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def test_node_pool_is_shared_by_actual_sp_owners_despite_uneven_demands():
    # Physical rank demand (rows) is 0, 2, 9, 18. All four see the same available pool.
    row = 4096
    demand = [n * row for n in (0, 2, 9, 18)]
    reserves = [row, 0, 2 * row, 3 * row]
    for cap_rows in range(33):
        caps = budget.hidden_ram_quotas(demand, cap_rows * row + 1, [50 * row, 45 * row, 25 * row, 50 * row],
                                       reserves, 3 * row, row)
        approved_rows = min(cap_rows, 25 - 6 - 3, 29)
        assert sum(caps) == approved_rows * row
        assert all(0 <= q <= d and q % row == 0 for q, d in zip(caps, demand))
        assert all(abs(q / row - approved_rows * d / sum(demand)) < 1 for q, d in zip(caps, demand))
    assert budget.hidden_ram_quotas([0, 0], 10, [100, 100], [1, 0], 2, 4) == (0, 0)
    assert budget.hidden_ram_quotas(demand, 10 * row, [50 * row, 50 * row, 0, 50 * row],
                                    reserves, 3 * row, row) == (0, 0, 0, 0)
    with pytest.raises(ValueError, match="whole"):
        budget.hidden_ram_quotas([10], 10, [100], [1], 2, 4)


def test_hdo_uses_each_owners_materialized_state_and_lazy_first_step():
    gib = 1024**3
    one = 10 * gib
    data = dict(hdo_cpu_param_numel=one // 4, hdo_cpu_master_bytes=one,
                hdo_cpu_grad_bytes=0, hdo_cpu_exp_avg_bytes=0, hdo_cpu_exp_avg_sq_bytes=0)
    assert budget.hdo_growth_reserve(data) == 3 * one
    data.update(hdo_cpu_exp_avg_bytes=one * 3, hdo_cpu_exp_avg_sq_bytes=one)
    assert budget.hdo_growth_reserve(data) == one  # oversized storage in a different field cannot offset grad.
    data.update(hdo_cpu_grad_bytes=one)
    assert budget.hdo_growth_reserve(data) == 0
    assert budget.hdo_growth_reserve({}) is None


KB = 1024
TOTAL = 1000 * KB
PHYSICAL_RAY = 950 * KB, TOTAL


def world(tmp_path, v1=False):
    proc = tmp_path / "proc"
    mount = tmp_path / "sys cgroup"
    (proc / "self").mkdir(parents=True)
    leaf = mount / "job" / "actor"
    leaf.mkdir(parents=True)
    (proc / "meminfo").write_text("MemTotal: 1000 kB\nMemAvailable: 750 kB\nSwapTotal: 0 kB\n")
    kind = "cgroup" if v1 else "cgroup2"
    options = "rw,memory" if v1 else "rw"
    prefix = "3:memory" if v1 else "0:"
    (proc / "self/cgroup").write_text(prefix + ":/job/actor\n")
    point = mount.as_posix().replace(" ", r"\040")
    (proc / "self/mountinfo").write_text(f"1 0 0:3 / {point} rw - {kind} mem {options}\n")
    for path, limit, usage in ((mount, TOTAL * 5, 6000), (mount / "job", 600 * KB, 450 * KB),
                               (leaf, 300 * KB, 5 * KB)):
        if v1:
            (path / "memory.limit_in_bytes").write_text(str(limit))
            (path / "memory.usage_in_bytes").write_text(str(usage))
            (path / "memory.use_hierarchy").write_text("1")
            (path / "memory.stat").write_text(f"hierarchical_memory_limit {limit}\n")
        elif path == mount:
            # The kernel's true v2 hierarchy root only exports controller capabilities.
            (path / "cgroup.controllers").write_text("cpu memory")
            (path / "cgroup.subtree_control").write_text("memory")
        else:
            (path / "memory.max").write_text(str(limit))
            (path / "memory.current").write_text(str(usage))
            (path / "memory.high").write_text("max")
    return proc, mount


@pytest.mark.parametrize("v1", [True, False])
def test_detectably_namespaced_root_cannot_prove_parent_headroom(tmp_path, v1):
    proc, mount = world(tmp_path, v1)
    (mount / ("memory.limit_in_bytes" if v1 else "memory.max")).write_text(str(900 * KB))
    sample = budget.HiddenMemoryHeadroom(PHYSICAL_RAY, proc)
    assert sample() == 0
    assert "outer ancestry hidden" in sample.unavailable_reason


def test_v1_subtree_requires_effective_ancestor_limits_and_exact_membership(tmp_path):
    proc, mount = world(tmp_path, v1=True)
    (proc / "self/mountinfo").write_text("1 0 0:3 /docker/container " + mount.as_posix().replace(" ", r"\040")
                                        + " rw - cgroup memory rw,memory\n")
    # Namespace-relative membership requires proof we reached this PID's actual group.
    members = mount / "job/actor/cgroup.procs"
    members.write_text("1\n")
    assert budget.HiddenMemoryHeadroom(PHYSICAL_RAY, proc)() == 0
    members.write_text(f"{os.getpid()}\n")
    sample = budget.HiddenMemoryHeadroom(PHYSICAL_RAY, proc)
    assert sample() == 120 * KB
    (mount / "memory.use_hierarchy").write_text("0")
    assert sample() == 0
    (mount / "memory.use_hierarchy").write_text("1")
    (mount / "memory.stat").write_text(f"hierarchical_memory_limit {750 * KB}\n")
    assert sample() == 0
    assert "outer v1" in sample.unavailable_reason
    (mount / "memory.memsw.limit_in_bytes").write_text(str(TOTAL * 5))
    (mount / "memory.memsw.usage_in_bytes").write_text("0")
    (mount / "memory.stat").write_text(f"hierarchical_memory_limit {5 * TOTAL}\n"
                                      f"hierarchical_memsw_limit {TOTAL + 50 * KB}\n")
    assert sample() == 120 * KB
    (proc / "meminfo").write_text("MemTotal: 1000 kB\nMemAvailable: 750 kB\nSwapTotal: 60 kB\n")
    assert sample() == 0  # hidden ancestor's combined limit can now bind


@pytest.mark.parametrize("v1", [True, False])
def test_host_ray_and_each_live_ancestor_limit(tmp_path, v1):
    proc, mount = world(tmp_path, v1)
    sample = budget.HiddenMemoryHeadroom(PHYSICAL_RAY, proc)
    assert sample() == 120 * KB  # job is tighter than the leaf and host: 600 * .95 - 450
    usage = "memory.usage_in_bytes" if v1 else "memory.current"
    (mount / "job" / usage).write_text(str(590 * KB))
    assert sample() == 0
    (mount / "job" / usage).write_text("0")
    if not v1:
        (mount / "job/memory.high").write_text(str(33 * KB))
        assert sample() == 33 * KB
    (proc / "meminfo").write_text("MemTotal: 1000 kB\nMemAvailable: 60 kB\n")
    assert sample() == 10 * KB
    (proc / "self/cgroup").write_text("0::/moved")
    assert sample() == 0


def test_unknown_ray_scope_and_inaccessible_bind_ancestors_spill(tmp_path, monkeypatch):
    proc, mount = world(tmp_path)
    (proc / "self/mountinfo").write_text("1 0 0:3 /job " + mount.as_posix().replace(" ", r"\040")
                                        + " rw - cgroup2 mem rw\n")
    assert budget.HiddenMemoryHeadroom(PHYSICAL_RAY, proc)() == 0
    monkeypatch.setattr(budget, "_raylet_cgroup_limit", lambda: (_ for _ in ()).throw(ValueError("unknown")))
    assert budget.HiddenMemoryHeadroom(None, proc)() == 0


def test_raylet_nonisolated_cgroup_limit_uses_real_actor_and_raylet_sibling_charge(tmp_path, monkeypatch):
    proc, mount = world(tmp_path)
    raylet = proc / "1177"
    raylet.mkdir()
    (raylet / "cgroup").write_text("0::/job/raylet\n")
    (raylet / "comm").write_text("raylet\n")
    for ns in ("mnt", "cgroup"):
        (proc / "self/ns").mkdir(exist_ok=True)
        (raylet / "ns").mkdir(exist_ok=True)
        (proc / "self/ns" / ns).write_text("namespace")
        os.link(proc / "self/ns" / ns, raylet / "ns" / ns)
    raygroup = mount / "job/raylet"
    raygroup.mkdir()
    (raygroup / "memory.max").write_text(str(200 * KB))
    (raygroup / "memory.high").write_text("max")
    (raygroup / "memory.current").write_text(str(170 * KB))
    (tmp_path / "logs").mkdir()
    log = tmp_path / "logs/raylet.out"
    startup = ("[2026-11-17 09:00:00,000 I 1177 1177] (raylet) memory_monitor.cc:50: "
               f"ThresholdMemoryMonitor initialized with usage threshold at {180 * KB} bytes "
               f"(90% of system memory), total system memory bytes: {200 * KB}, monitor interval: 250ms")
    log.write_text(startup)
    node = SimpleNamespace(get_session_dir_path=lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(
        _private=SimpleNamespace(worker=SimpleNamespace(global_worker=SimpleNamespace(node=node)))))
    sample = budget.HiddenMemoryHeadroom(None, proc)
    assert sample() == 10 * KB
    (raygroup / "memory.max").write_text(str(208 * KB))
    assert sample() == 0  # reported Ray total can no longer be tied to an observed scope
    log.write_text(startup + ", resource isolation enabled, monitoring only user slice memory usage")
    assert budget.HiddenMemoryHeadroom(None, proc)() == 0
