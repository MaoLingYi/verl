"""Node admission for optional V1.8.3 hidden payload RAM. Unknown headroom means spill."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re


def hidden_ram_quotas(planned_bytes_by_rank, node_cap, headroom_by_rank, reserve_by_rank, safety_bytes, row_bytes):
    """Proportionally divide one affordable pool amongst the physical SP row owners."""
    if not (planned_bytes_by_rank and len(planned_bytes_by_rank) == len(headroom_by_rank) == len(reserve_by_rank)):
        raise ValueError("Expected one memory plan, reserve and headroom observation per owner")
    if row_bytes <= 0 or min(node_cap, safety_bytes, *reserve_by_rank, *planned_bytes_by_rank) < 0:
        raise ValueError("Invalid hidden RAM budget")
    if any(n % row_bytes for n in planned_bytes_by_rank):
        raise ValueError("Hidden payload plan must contain whole BF16 rows")

    rows = [n // row_bytes for n in planned_bytes_by_rank]
    total = sum(rows)
    free = min(headroom_by_rank) - sum(reserve_by_rank) - safety_bytes
    approved = min(total, max(0, min(node_cap, free)) // row_bytes)
    if not approved:
        return (0,) * len(rows)
    shares = [divmod(approved * n, total) for n in rows]
    quotas = [whole for whole, _ in shares]
    # Largest remainders preserve the node total even when it is only one row.
    for rank in sorted(range(len(rows)), key=lambda i: shares[i][1], reverse=True)[:approved - sum(quotas)]:
        quotas[rank] += 1
    return tuple(n * row_bytes for n in quotas)


def hdo_growth_reserve(estimate):
    """Reserve missing FP32 master/grad/moments from the real partial-HDO group inventory."""
    numel = estimate.get("hdo_cpu_param_numel", 0)
    if numel <= 0:
        return None
    # Surplus storage in one category cannot pay for absent tensors in another.
    return sum(max(0, 4 * numel - estimate["hdo_cpu_" + field + "_bytes"])
               for field in ("master", "exp_avg", "exp_avg_sq", "grad"))


def _unescape(field):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), field)


def _cgroup_chain(membership, mountinfo, pid):
    groups = {}
    for line in membership.splitlines():
        hierarchy, controllers, name = line.split(":", 2)
        if "memory" in controllers.split(","):
            groups["cgroup"] = PurePosixPath(name)
        elif hierarchy == "0" and not controllers:
            groups["cgroup2"] = PurePosixPath(name)
    # On hybrid v1/v2 systems at most one hierarchy owns the memory controller.
    kind = "cgroup" if "cgroup" in groups else "cgroup2"
    member = groups[kind]
    if not member.is_absolute() or ".." in member.parts:
        raise ValueError("Unresolvable memory cgroup membership")
    mounts = []
    for line in mountinfo.splitlines():
        fields, filesystem = line.split(" - ", 1)
        fs, _, options = filesystem.split()[:3]
        if fs != kind or (fs == "cgroup" and "memory" not in options.split(",")):
            continue
        root, mount = fields.split()[3:5]
        root = PurePosixPath(_unescape(root))
        mount = Path(_unescape(mount))
        if root == member or root in member.parents:
            mounts.append((root, mount, member.relative_to(root)))
        elif kind == "cgroup":
            # A cgroup namespace may report /actor relative to a /docker/<id> bind root.
            relative = member.relative_to("/")
            try:
                with (mount / relative / "cgroup.procs").open() as members:
                    if any(line.strip() == str(pid) for line in members):
                        mounts.append((root, mount, relative))
            except FileNotFoundError:
                pass
    if not mounts:
        raise ValueError("No visible mount for memory cgroup")
    root, mount, relative = min(mounts, key=lambda entry: len(entry[0].parts))
    if root != PurePosixPath("/") and kind == "cgroup2":
        # Parent charge outside a bind-mounted subtree cannot be inferred from its child's limit.
        raise ValueError("Memory cgroup ancestors are hidden above the visible mount")
    path = mount
    chain = [(path, kind)]
    for part in relative.parts:
        path = path / part
        chain.append((path, kind))
    return tuple(chain), root != PurePosixPath("/")


def _raylet_cgroup_limit():
    """Identify the non-isolated raylet whose monitor selected a finite cgroup total."""
    import ray

    session = Path(ray._private.worker.global_worker.node.get_session_dir_path())
    with (session / "logs/raylet.out").open(encoding="utf-8", errors="replace") as stream:
        startup = stream.read(64 * 1024)
    match = re.search(
        r"^\[[^\]\n]*?\s[IWD]\s+(\d+)\s+\d+\]\s+\(raylet\)[^\n]*?"
        r"ThresholdMemoryMonitor initialized with usage threshold at (\d+) bytes "
        r"[^\n]*?total system memory bytes: (\d+), monitor interval: [^\n]*", startup, re.MULTILINE,
    )
    if match is None or "resource isolation enabled" in match[0]:
        raise ValueError("Unidentified Ray monitor accounting scope")
    return tuple(map(int, match.groups()))


def _limit(path):
    text = path.read_text().strip()
    return None if text == "max" else int(text)


class HiddenMemoryHeadroom:
    """Sample live host, Ray and visible ancestor headroom. Instantiate once per optimizer update."""

    def __init__(self, ray_limit, proc=Path("/proc")):
        self.proc = proc
        self.ray_limit = None
        self.unavailable_reason = None
        self.memberships = {}
        try:
            limit = ray_limit
            subjects = [proc / "self"]
            if limit is None:
                pid, threshold, total = _raylet_cgroup_limit()
                raylet = proc / str(pid)
                if (raylet / "comm").read_text().strip() != "raylet" or not all(
                    os.path.samefile(proc / "self/ns" / ns, raylet / "ns" / ns) for ns in ("mnt", "cgroup")
                ):
                    raise ValueError("Raylet accounting namespace is not visible")
                subjects.append(raylet)
                limit = threshold, total
            if not 0 < limit[0] <= limit[1]:
                raise ValueError("Invalid Ray monitor threshold")
            mountinfo = (proc / "self/mountinfo").read_text()
            scopes = set()
            self.roots = set()
            self.subtree_mounts = set()
            for subject in subjects:
                path = subject / "cgroup"
                self.memberships[path] = path.read_text()
                subject_pid = os.getpid() if subject.name == "self" else int(subject.name)
                chain, subtree = _cgroup_chain(self.memberships[path], mountinfo, subject_pid)
                scopes.update(chain)
                self.roots.add(chain[0])
                if subtree:
                    self.subtree_mounts.add(chain[0])
            self.scopes = sorted(scopes)
            self.ray_scopes = set(chain)  # last subject is the raylet for cgroup accounting
            self.ray_limit = limit
        except (ImportError, AttributeError, OSError, KeyError, ValueError, IndexError) as exc:
            self.unavailable_reason = str(exc)

    def __call__(self):
        if self.ray_limit is None:
            return 0
        try:
            if any(path.read_text() != old for path, old in self.memberships.items()):
                raise ValueError("Memory cgroup membership changed during the update")
            meminfo = {}
            for line in (self.proc / "meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                if key in ("MemAvailable", "MemTotal", "SwapTotal"):
                    meminfo[key] = int(value.split()[0]) * 1024
            physical = meminfo["MemTotal"]
            available = meminfo["MemAvailable"]
            threshold, ray_total = self.ray_limit
            if not 0 <= available <= physical or ray_total > physical:
                raise ValueError("Implausible host/Ray totals")
            headroom = available
            comparable = physical == ray_total
            if comparable:
                headroom -= physical - threshold
            for path, kind in self.scopes:
                if kind == "cgroup2":
                    # The real v2 root has no memory.max, as can a leaf with memory disabled.
                    try:
                        hard = _limit(path / "memory.max")
                    except FileNotFoundError:
                        (path / "cgroup.controllers").read_text()
                        if (path, kind) not in self.roots and "memory" in (
                            path.parent / "cgroup.subtree_control"
                        ).read_text().split():
                            raise ValueError("Expected memory.max for an enabled cgroup")
                        continue
                    if (path, kind) in self.roots:
                        raise ValueError("Non-root v2 memory cgroup presented as /; outer ancestry hidden")
                    high = _limit(path / "memory.high")
                    current = int((path / "memory.current").read_text())
                    if high is not None:
                        headroom = min(headroom, high - current)
                else:
                    hard = _limit(path / "memory.limit_in_bytes")
                    if (path, kind) in self.roots and hard is not None and hard <= physical:
                        raise ValueError("Finite v1 memory cgroup presented as /; outer ancestry hidden")
                    current = int((path / "memory.usage_in_bytes").read_text())
                    try:
                        memsw = _limit(path / "memory.memsw.limit_in_bytes")
                    except FileNotFoundError:
                        memsw = None
                    if memsw is not None:
                        headroom = min(headroom, memsw - int((path / "memory.memsw.usage_in_bytes").read_text()))
                    if (path, kind) in self.roots:
                        hierarchy = (path / "memory.use_hierarchy").read_text().strip()
                        if hierarchy == "1":
                            # v1 exports the minimum of ALL enclosing hard limits. Without
                            # visibility of outer charge it is usable only above physical capacity.
                            limits = dict(line.split() for line in (path / "memory.stat").read_text().splitlines())
                            if (int(limits["hierarchical_memory_limit"]) <= physical or
                                (memsw is not None and int(limits["hierarchical_memsw_limit"])
                                 <= physical + meminfo["SwapTotal"])):
                                raise ValueError("Effective outer v1 cgroup charge is not visible")
                        elif (path, kind) in self.subtree_mounts or hierarchy != "0":
                            raise ValueError("Cannot verify outer v1 cgroup hierarchy")
                if current < 0:
                    raise ValueError("Invalid cgroup memory charge")
                if hard is not None:
                    # Raw charge includes reclaimable file pages: conservative relative to working-set accounting.
                    headroom = min(headroom, hard * threshold // ray_total - current)
                    if (path, kind) in self.ray_scopes and hard == ray_total:
                        comparable = True
            if not comparable:
                raise ValueError("Raylet total differs from visible host and raylet cgroup limits")
            self.unavailable_reason = None
            return max(0, headroom)
        except (OSError, KeyError, ValueError, IndexError) as exc:
            self.unavailable_reason = str(exc)
            return 0
