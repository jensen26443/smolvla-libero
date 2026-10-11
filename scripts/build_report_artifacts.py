#!/usr/bin/env python
"""Build reproducible CSV/JSON/PNG report artifacts from formal eval results.

The script intentionally reads only the machine-readable summaries produced by
``sim-vla-summarize-lerobot``.  It never re-runs an evaluator and never
recomputes a denominator from raw video files.  E2-100% is represented by the
identical E1 result, as required by the experiment protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _summary_row(condition: str, budget: str, path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    overall = summary.get("overall", {}).get("overall")
    if not isinstance(overall, dict):
        raise TypeError(f"summary is missing overall.overall: {path}")
    interval = overall.get("wilson_95")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(f"summary is missing Wilson interval: {path}")
    return {
        "condition": condition,
        "budget": budget,
        "successes": int(overall["successes"]),
        "trials": int(overall["trials"]),
        "success_rate": float(overall["success_rate"]),
        "wilson_low": float(interval[0]),
        "wilson_high": float(interval[1]),
        "source": str(path.relative_to(ROOT)),
    }


def _conditions(strict: bool) -> list[tuple[str, str, Path, dict[str, Any]]]:
    # Keep this order stable so CSVs and figures are diff-friendly.
    specs: list[tuple[str, str, Path]] = [
        ("E0", "baseline", RESULTS / "e0_formal" / "summary.json"),
        ("E1", "100pct", RESULTS / "e1_formal" / "summary.json"),
        ("E2_020pct", "20pct", RESULTS / "e2_020pct_formal" / "summary.json"),
        ("E2_050pct", "50pct", RESULTS / "e2_050pct_formal" / "summary.json"),
        # S4.3: E2-100% is exactly the E1 checkpoint/evaluation, not a new trial.
        ("E2_100pct", "100pct", RESULTS / "e1_formal" / "summary.json"),
        ("E5", "paraphrase", RESULTS / "e5_formal" / "summary.json"),
    ]
    available: list[tuple[str, str, Path, dict[str, Any]]] = []
    missing: list[Path] = []
    for condition, budget, path in specs:
        if not path.is_file():
            missing.append(path)
            continue
        available.append((condition, budget, path, _read_json(path)))
    if strict and missing:
        missing_text = ", ".join(str(path.relative_to(ROOT)) for path in missing)
        raise FileNotFoundError(f"formal report inputs are incomplete: {missing_text}")
    if not available:
        raise FileNotFoundError("no formal summary inputs found")
    return available


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_tables(conditions: list[tuple[str, str, Path, dict[str, Any]]]) -> None:
    overall_rows = [_summary_row(c, b, p, s) for c, b, p, s in conditions]
    _write_csv(
        RESULTS / "formal_results.csv",
        overall_rows,
        [
            "condition",
            "budget",
            "successes",
            "trials",
            "success_rate",
            "wilson_low",
            "wilson_high",
            "source",
        ],
    )

    task_rows: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    for condition, budget, path, summary in conditions:
        by_task = summary.get("overall", {}).get("by_task", {})
        if not isinstance(by_task, dict):
            raise TypeError(f"summary is missing overall.by_task: {path}")
        overall = summary["overall"]["overall"]
        for failure_type, count in overall.get("failure_types", {}).items():
            failures[f"{condition}:{failure_type}"] += int(count)
        for task_key, task in sorted(by_task.items()):
            interval = task.get("wilson_95")
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(f"task summary is missing Wilson interval: {path} ({task_key})")
            task_rows.append(
                {
                    "condition": condition,
                    "budget": budget,
                    "task_key": task_key,
                    "successes": int(task["successes"]),
                    "trials": int(task["trials"]),
                    "success_rate": float(task["success_rate"]),
                    "wilson_low": float(interval[0]),
                    "wilson_high": float(interval[1]),
                    "source": str(path.relative_to(ROOT)),
                }
            )
    _write_csv(
        RESULTS / "task_results.csv",
        task_rows,
        [
            "condition",
            "budget",
            "task_key",
            "successes",
            "trials",
            "success_rate",
            "wilson_low",
            "wilson_high",
            "source",
        ],
    )
    (RESULTS / "failure_summary.json").write_text(
        json.dumps(
            {
                "protocol_version": "sim-vla-libero-v1",
                "sources": [str(path.relative_to(ROOT)) for _, _, path, _ in conditions],
                "by_condition_and_type": dict(sorted(failures.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _build_figures(conditions: list[tuple[str, str, Path, dict[str, Any]]]) -> str:
    """Write the two required figures; return a short status string."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib unavailable; CSV/JSON artifacts were written"

    rows = [_summary_row(c, b, p, s) for c, b, p, s in conditions]
    efficiency = [row for row in rows if row["condition"].startswith("E2_")]
    # E2-100% is the E1 result but remains an explicit point in this curve.
    x = {"20pct": 20, "50pct": 50, "100pct": 100}
    efficiency.sort(key=lambda row: x.get(row["budget"], 999))
    if efficiency:
        fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
        xs = [x[row["budget"]] for row in efficiency]
        ys = [row["success_rate"] * 100 for row in efficiency]
        lows = [ys_i - row["wilson_low"] * 100 for ys_i, row in zip(ys, efficiency, strict=True)]
        highs = [row["wilson_high"] * 100 - ys_i for ys_i, row in zip(ys, efficiency, strict=True)]
        ax.errorbar(xs, ys, yerr=[lows, highs], fmt="o-", capsize=4, linewidth=1.8)
        ax.set(xlabel="Training demonstrations (% of nested train pool)", ylabel="Success rate (%)")
        ax.set_ylim(0, 105)
        ax.set_xticks([20, 50, 100])
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(RESULTS / "data_efficiency.png", dpi=180)
        plt.close(fig)

    task_rows: list[dict[str, Any]] = []
    for condition, budget, path, summary in conditions:
        for task_key, task in sorted(summary["overall"]["by_task"].items()):
            task_rows.append(
                {
                    "condition": condition,
                    "budget": budget,
                    "task_key": task_key,
                    "success_rate": float(task["success_rate"]) * 100,
                }
            )
    if task_rows:
        tasks = sorted({row["task_key"] for row in task_rows})
        plot_conditions = [row["condition"] for row in rows]
        # One grouped panel per condition keeps the figure readable with six rows.
        fig, axes = plt.subplots(
            len(plot_conditions),
            1,
            figsize=(8.0, max(3.0, 1.15 * len(plot_conditions))),
            squeeze=False,
            constrained_layout=True,
        )
        for axis, condition in zip(axes[:, 0], plot_conditions, strict=True):
            values = {
                row["task_key"]: row["success_rate"]
                for row in task_rows
                if row["condition"] == condition
            }
            axis.bar(range(len(tasks)), [values.get(task, float("nan")) for task in tasks])
            axis.set_ylim(0, 105)
            axis.set_ylabel(condition)
            axis.set_xticks(range(len(tasks)), [task.replace("goal_bowl_", "") for task in tasks])
            axis.grid(axis="y", alpha=0.2)
        axes[-1, 0].set_xlabel("Task")
        fig.savefig(RESULTS / "task_success_rates.png", dpi=180)
        plt.close(fig)
    return "figures written"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="require every formal E0/E1/E2/E5 summary")
    args = parser.parse_args(argv)
    conditions = _conditions(args.strict)
    _build_tables(conditions)
    print(f"formal report tables written for {len(conditions)} conditions; {_build_figures(conditions)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
