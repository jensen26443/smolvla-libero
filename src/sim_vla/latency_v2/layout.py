"""Resolve evidence locations from a frozen spec, with legacy read compatibility."""
from __future__ import annotations

import json
from pathlib import Path


def load_layout(root: Path, spec_path: Path | None = None):
    path = root / (spec_path or Path("runs/latency_v2_frozen/experiment_spec.json"))
    frozen = json.loads(path.read_text())
    runs = {name: root / frozen.get("formal_run_config_templates", {}).get(name, {}).get(
        "run_dir", "runs/latency_v2_formal_" + name) for name in ("core", "jitter", "async")}
    return path, frozen, runs


def completion_flags(directory: Path, rows: list[dict], expected: int) -> dict:
    complete = len(rows) == expected
    asynchronous = [row for row in rows if row.get("backend") == "actual_spawned_gpu_worker"]
    realtime = all(row.get("realtime_20hz_passed", False) and row.get("no_catchup_or_skipped_steps", False)
                   for row in asynchronous) if complete and asynchronous else None
    timing = bool(rows) and all(row.get("timing_provenance_complete", False) for row in rows)
    for row in rows:
        from .storage import trial_key

        fields = ("task", "init_index", "seed", "latency", "profile", "config")
        target = directory/"trials"/trial_key({key: row[key] for key in fields})
        attempt = json.loads((target/"attempt.json").read_text()) if (target/"attempt.json").exists() else {}
        timing = timing and attempt.get("end_time_status") == "known"
    return {"delivery_complete": complete, "realtime_20hz_passed": realtime if asynchronous else None,
            "timing_provenance_complete": timing if complete else None}


def write_progress(config: dict, directory: Path, rows: list[dict], expected: int, remaining: int):
    from ..assets import project_root
    from .storage import publish

    progress = {"completed": len(rows), "expected": expected, "remaining_budget": remaining,
                **completion_flags(directory, rows, expected)}
    publish(directory/"progress.json", progress)
    if not config.get("run_id"):
        return
    root = project_root()
    base = root/"runs"/config["run_id"]
    stages = {}
    for name in ("core", "jitter", "async"):
        path = base/("formal_"+name)/"progress.json"
        if path.exists():
            stages[name] = json.loads(path.read_text())
    matrix_complete = len(stages) == 3 and all(row["delivery_complete"] for row in stages.values())
    state = {"run_id": config["run_id"], "stage": config["stage"], "delivery_complete": False,
        "formal_completed": sum(row["completed"] for row in stages.values()), "formal_expected": 345,
        "realtime_20hz_passed": stages.get("async", {}).get("realtime_20hz_passed"),
        "timing_provenance_complete": all(row["timing_provenance_complete"] for row in stages.values()) if matrix_complete else None,
        "status": "matrices_complete_materials_pending" if matrix_complete else "running",
        "remaining_budget": remaining, "stages": stages}
    publish(base/"progress.json", state)
    docs = root/"docs/latency_v2"/config["run_id"].removeprefix("latency_v2_")
    docs.mkdir(parents=True, exist_ok=True)
    (docs/"PROGRESS.md").write_text(f"# PLAN-04 {config['run_id']}\n\n"
        f"阶段：{config['stage']}；正式 trial：{state['formal_completed']}/345；剩余预算：{remaining}。\n\n"
        f"- delivery_complete: {state['delivery_complete']}\n"
        f"- realtime_20hz_passed: {state['realtime_20hz_passed']}\n"
        f"- timing_provenance_complete: {state['timing_provenance_complete']}\n\n"
        "null 表示尚未完成该项验证。材料完整与逐回合实时验收分别判断。\n")
