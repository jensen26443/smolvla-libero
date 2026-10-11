#!/usr/bin/env python
"""Build compact, reproducible E3/E4 quantitative report artifacts.

The paired evaluator writes one official ``eval_info.json`` per condition and
an additional episode log containing the shared initialization index.  This
script only aggregates those files; it never reruns simulation or changes the
success denominator.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _task_rows(*, phase: str, condition: str, directory: Path) -> list[dict[str, Any]]:
    summary = _read(directory / "summary.json")
    protocol = _read(directory / "protocol.json")
    records_payload = _read(directory / "episode_records.json")
    by_task = summary.get("by_task")
    if not isinstance(by_task, dict):
        raise TypeError(f"summary missing by_task: {directory / 'summary.json'}")
    records = records_payload.get("records", [])
    if not isinstance(records, list):
        raise TypeError(f"episode_records missing records: {directory / 'episode_records.json'}")
    instructions = {str(key): str(value) for key, value in protocol.get("instructions", {}).items()}
    task_ids = [int(value) for value in protocol.get("task_ids", [])]
    task_keys = [str(value) for value in protocol.get("task_keys", [])]
    task_id_by_key = dict(zip(task_keys, task_ids, strict=True))
    rows: list[dict[str, Any]] = []
    for task_key, metrics in sorted(by_task.items()):
        if not isinstance(metrics, dict):
            raise TypeError(f"invalid task summary for {task_key}: {directory}")
        task_id = task_id_by_key.get(str(task_key))
        task_records = [record for record in records if str(record.get("task_key")) == str(task_key)]
        success_steps = [
            int(record["first_success_step"])
            for record in task_records
            if record.get("first_success_step") is not None and bool(record.get("success"))
        ]
        timeouts = sum(1 for record in task_records if int(record.get("steps", 0)) >= 300 and not record.get("success"))
        interval = metrics.get("wilson_95")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"missing Wilson interval for {task_key}: {directory}")
        rows.append(
            {
                "phase": phase,
                "condition": condition,
                "task_key": str(task_key),
                "task_id": task_id,
                "instruction": instructions.get(str(task_id), ""),
                "successes": int(metrics["successes"]),
                "trials": int(metrics["trials"]),
                "success_rate": float(metrics["success_rate"]),
                "wilson_low": float(interval[0]),
                "wilson_high": float(interval[1]),
                "mean_first_success_step": statistics.mean(success_steps) if success_steps else None,
                "median_first_success_step": statistics.median(success_steps) if success_steps else None,
                "timeouts": timeouts,
                "source": str((directory / "summary.json").relative_to(ROOT)),
                "shared_init_anchor": str(protocol.get("shared_init_anchor", "")),
                "shared_init_state_count": int(protocol.get("shared_init_state_count", 0)),
                "start_seed": int(protocol.get("start_seed", 0)),
                "start_initialization_index": int(protocol.get("start_initialization_index", 0)),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "phase",
        "condition",
        "task_key",
        "task_id",
        "instruction",
        "successes",
        "trials",
        "success_rate",
        "wilson_low",
        "wilson_high",
        "mean_first_success_step",
        "median_first_success_step",
        "timeouts",
        "shared_init_anchor",
        "shared_init_state_count",
        "start_seed",
        "start_initialization_index",
        "source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(path: Path, rows: list[dict[str, Any]]) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib unavailable; skipped figure"
    formal = [row for row in rows if row["phase"] == "formal"]
    if not formal:
        return "no formal rows; skipped figure"
    labels = [f"{row['condition']}\n{row['task_key'].replace('goal_bowl_', '')}" for row in formal]
    values = [row["success_rate"] * 100 for row in formal]
    lows = [value - row["wilson_low"] * 100 for value, row in zip(values, formal, strict=True)]
    highs = [row["wilson_high"] * 100 - value for value, row in zip(values, formal, strict=True)]
    fig, axis = plt.subplots(figsize=(max(8.0, 0.9 * len(labels)), 4.8), constrained_layout=True)
    axis.errorbar(range(len(labels)), values, yerr=[lows, highs], fmt="o", capsize=4)
    axis.set(
        title="E3/E4 paired-language formal evaluation",
        ylabel="Success rate (%)",
        ylim=(0, 105),
        xticks=range(len(labels)),
        xticklabels=labels,
    )
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return "figure written"


def build(*, strict: bool = True) -> dict[str, Any]:
    specs = [
        ("development", "E3", RESULTS / "e3_shared_dev"),
        ("development", "E4_empty", RESULTS / "e4_empty_dev"),
        ("formal", "E3", RESULTS / "e3_shared_formal"),
        ("formal", "E4_empty", RESULTS / "e4_empty_formal"),
    ]
    rows: list[dict[str, Any]] = []
    missing: list[Path] = []
    for phase, condition, directory in specs:
        if not (directory / "summary.json").is_file():
            missing.append(directory / "summary.json")
            continue
        rows.extend(_task_rows(phase=phase, condition=condition, directory=directory))
    if strict and missing:
        raise FileNotFoundError("missing E3/E4 report inputs: " + ", ".join(str(path.relative_to(ROOT)) for path in missing))
    if not rows:
        raise FileNotFoundError("no E3/E4 summaries found")

    status_path = RESULTS / "paired_language_status.json"
    status = _read(status_path) if status_path.is_file() else {}
    formal_presence = {
        condition: (directory / "summary.json").is_file()
        for phase, condition, directory in specs
        if phase == "formal"
    }
    if all(formal_presence.values()):
        report_status = "completed"
    elif any(formal_presence.values()):
        report_status = "formal_partial"
    elif rows:
        report_status = "development_only"
    else:
        report_status = str(status.get("status", "in_progress"))
    payload = {
        "protocol": "paired_shared_init_v1",
        "status": report_status,
        "formal_completion": formal_presence,
        "current_blocker": status.get("current_blocker"),
        "formal_required_trials": {"E3": 60, "E4_empty": 20},
        "rows": rows,
        "interpretation": {
            "E3": "native valid target instructions evaluated with the same frozen bowl-to-plate initialization array; each target keeps its native BDDL success predicate",
            "E4_empty": "empty-instruction input ablation on the bowl-to-plate anchor; the E3 valid-instruction trajectories provide the valid-language reference and are not counted as extra E4 trials",
            "scope": "custom paired protocol; not an official native per-task initialization benchmark",
        },
        "shared_reset_probes": status.get("shared_reset_probes", {}),
    }
    (RESULTS / "e3_e4_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(RESULTS / "e3_e4_results.csv", rows)
    figure_status = _write_figure(RESULTS / "e3_e4_success_rates.png", rows)
    payload["figure"] = figure_status
    (RESULTS / "e3_e4_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"rows": len(rows), "figure": figure_status, "output": str(RESULTS / "e3_e4_summary.json")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true", help="write available development rows without requiring formal inputs")
    args = parser.parse_args(argv)
    result = build(strict=not args.allow_incomplete)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
