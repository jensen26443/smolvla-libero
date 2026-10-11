"""One allocation of the unused PLAN-06 balance, with immutable parent records."""
from pathlib import Path

from ..assets import project_root
from ..latency_v2.integrity import file_sha256
from ..latency_v2.storage import Budget, canonical_hash, publish
from ..latency_v4 import layout as previous_layout
from ..latency_v4.layout import immutable_bytes, immutable_json, read

ROOT = project_root()
CHECKPOINT = previous_layout.CHECKPOINT
RUN_ID = "latency_v5_plan07"
RUN = ROOT / "runs" / RUN_ID
DOCS = ROOT / "docs/latency_v5"
CONFIG = ROOT / "configs/latency_v5/experiment_spec.json"
PARENT_LEDGER = ROOT / "runs/latency_v4_plan06/budget.json"
OLD_LEDGER = ROOT / "runs/latency_v2_budget.json"
PLAN = ROOT.parent / "plan/02/PLAN-07_控制线程去渲染与持续实时执行.md"
METHODS = ("RTC_STRICT", "RTC_CATCHUP")
RUNTIME = {"observation_protocol": "frozen_full_scene_rendered_in_worker", "delay_floor": "production_probe_p95",
           "visual_snapshot": "model_site_rgba_v3_signed_alpha",
           "window": 10, "percentile": 95, "tick_seconds": .05, "resolution": [256, 256],
           "max_ticks": 300, "sampling_steps": 5, "nominal_period": 25, "preview": "both_cameras_on_result"}


def budget():
    allocation = RUN / "allocation.json"
    if not allocation.exists():
        parent = read(PARENT_LEDGER)
        used = parent["initial_s0_and_profile_attempts"] + len(parent["attempts"])
        if parent["cap"] != 600 or used != 564:
            raise ValueError("PLAN-07 requires the recorded 36 remaining attempts; no renewed allocation")
        value = {"format": "latency-v5-budget-allocation-v1", "run_id": RUN_ID,
                 "authorization": "user: 继续下一轮，解决该问题", "cap": 36,
                 "source_authorized_cap": 600, "parent_used_at_allocation": used,
                 "parent_ledger_sha256": file_sha256(PARENT_LEDGER),
                 "old_ledger_sha256": file_sha256(OLD_LEDGER), "historical_used": 1770,
                 "historical_total_at_start": 2334,
                 "scope": "allocate the unused 36; parent balance is reserved, never a second budget"}
        immutable_json(allocation, value)
    value = read(allocation)
    if (value["cap"] != 36 or value["source_authorized_cap"] != 600
            or value["parent_used_at_allocation"] != 564
            or file_sha256(PARENT_LEDGER) != value["parent_ledger_sha256"]
            or file_sha256(OLD_LEDGER) != value["old_ledger_sha256"]):
        raise ValueError("parent accounting changed or the remaining allocation was renewed")
    result = Budget(RUN / "budget.json", cap=36, initial=0)
    ledger = read(result.path)
    if ledger["cap"] != 36 or ledger["initial_s0_and_profile_attempts"] != 0 or result.remaining() < 0:
        raise ValueError("PLAN-07 budget changed or exceeded its allocation")
    return result


def sources():
    old = read(ROOT / "runs/latency_v4_plan06_continue01/delivery/code_inventory.json")["files"]
    files = [ROOT / name for name in old]
    files += list((ROOT / "src/sim_vla/latency_v5").glob("*.py"))
    files += list((ROOT / "tests").glob("test_latency_v5*.py"))
    files += [ROOT / "scripts/run_latency_v5.sh"]
    result = {str(path.relative_to(ROOT)): file_sha256(path) for path in sorted(set(files)) if path.is_file()}
    if any(result.get(name) != digest for name, digest in old.items()):
        raise ValueError("protected PLAN-06 control, sampler, statistics or installed source changed")
    return result


def progress(stage, **fields):
    b = budget()
    value = {"run_id": RUN_ID, "stage": stage, "attempts_used": 36 - b.remaining(),
             "remaining": b.remaining(), "total_new_authorized_used": 564 + 36 - b.remaining(), **fields}
    publish(RUN / "progress.json", value)
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "PROGRESS.md").write_text("# PLAN-07 进度\n\n```json\n" +
        __import__("json").dumps(value, ensure_ascii=False, indent=2) + "\n```\n")
    print(__import__("json").dumps(value, ensure_ascii=False), flush=True)
    return value


def seed(task, index):
    return int(canonical_hash([RUN_ID, task, index, 20261006])[:8], 16) % (2**31 - 1)


def snapshot_sources(target):
    inventory = sources()
    for name in inventory:
        immutable_bytes(Path(target) / name, (ROOT / name).read_bytes())
    return inventory
