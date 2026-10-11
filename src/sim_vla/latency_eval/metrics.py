"""Aggregation and statistics over completed trials of one experiment.

All outputs are regenerable from the published trial records alone (no live
state), which is the reproducibility requirement for the final report.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..protocol import wilson_interval


def load_trial_results(experiment_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(experiment_dir) / "trials"
    results: list[dict[str, Any]] = []
    if not root.is_dir():
        return results
    for trial_dir in sorted(root.iterdir()):
        if not trial_dir.is_dir():
            continue
        result_path = trial_dir / "result.json"
        if not result_path.is_file():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["_dir"] = str(trial_dir)
        results.append(payload)
    return results


def _group(results: list[dict[str, Any]], *keys: str) -> dict[tuple, list[dict[str, Any]]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[tuple(str(result.get(key)) for key in keys)].append(result)
    return dict(grouped)


def _success_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(1 for r in records if bool(r["success"]))
    trials = len(records)
    low, high = wilson_interval(successes, trials)
    reasons: dict[str, int] = defaultdict(int)
    for record in records:
        if not record["success"]:
            reasons[str(record.get("termination_reason", "unspecified"))] += 1
    steps = [float(r.get("steps", 0)) for r in records]
    success_steps = [float(r["steps"]) for r in records if r["success"]]
    requests = [float(r.get("n_requests", 0)) for r in records]
    sim_seconds = [float(r.get("steps", 0)) * float(r.get("tick_seconds", 0.05)) for r in records]
    inference = [float(r.get("inference_seconds_total", 0.0)) for r in records]
    total_sim = sum(sim_seconds)
    return {
        "successes": successes,
        "trials": trials,
        "success_rate": successes / trials if trials else None,
        "wilson_95": [low, high] if trials else None,
        "failure_types": dict(sorted(reasons.items())),
        "mean_steps": float(np.mean(steps)) if steps else None,
        "mean_success_steps": float(np.mean(success_steps)) if success_steps else None,
        "mean_requests_per_episode": float(np.mean(requests)) if requests else None,
        "requests_per_sim_second": (float(np.sum(requests)) / total_sim) if total_sim else None,
        "inference_seconds_total": float(np.sum(inference)),
        "wall_seconds_total": float(sum(float(r.get("wall_seconds", 0.0)) for r in records)),
        "timeout_fraction": (
            sum(1 for r in records if r.get("termination_reason") == "timeout") / trials
            if trials
            else None
        ),
    }


def summarize_experiment(experiment_dir: str | Path) -> dict[str, Any]:
    results = load_trial_results(experiment_dir)
    if not results:
        return {"experiment_dir": str(experiment_dir), "trials": 0}
    summary: dict[str, Any] = {
        "experiment_dir": str(experiment_dir),
        "trials": len(results),
        "overall": _success_summary(results),
        "by_task": {k[0]: _success_summary(v) for k, v in sorted(_group(results, "task").items())},
        "by_task_method_latency": {
            "/".join(k): _success_summary(v)
            for k, v in sorted(_group(results, "task", "method", "latency").items())
        },
        "by_method": {k[0]: _success_summary(v) for k, v in sorted(_group(results, "method").items())},
        "attempts_histogram": dict(
            sorted(
                {
                    str(a): sum(1 for r in results if int(r.get("attempt", 1)) == a)
                    for a in {int(r.get("attempt", 1)) for r in results}
                }.items()
            )
        ),
    }
    summary["paired_comparisons"] = paired_comparisons(results)
    summary["distributions"] = distribution_summary(results)
    return summary


def paired_comparisons(
    results: list[dict[str, Any]],
    *,
    n_bootstrap: int = 10000,
    seed: int = 7,
) -> dict[str, Any]:
    """Paired bootstrap success-rate differences over shared initializations.

    Pairing key: (task, latency, init_index). Methods being compared must share
    that key; unpaired leftovers are excluded and counted. Shared seeds do not
    make conditions independent samples — the bootstrap below resamples
    initializations, not frames.
    """

    cell: dict[tuple, dict[str, bool]] = defaultdict(dict)
    for result in results:
        key = (str(result["task"]), str(result["latency"]), int(result["init_index"]))
        cell[key][str(result["method"])] = bool(result["success"])
    methods = sorted({m for per_key in cell.values() for m in per_key})
    pairs: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    for i, method_a in enumerate(methods):
        for method_b in methods[i + 1 :]:
            shared = [
                (key, per_key[method_a], per_key[method_b])
                for key, per_key in cell.items()
                if method_a in per_key and method_b in per_key
            ]
            if len(shared) < 2:
                continue
            keys = np.array([f"{task}/{latency}/i{init:02d}" for task, latency, init in (k for k, _, _ in shared)])
            a = np.array([va for _, va, _ in shared], dtype=bool)
            b = np.array([vb for _, _, vb in shared], dtype=bool)
            diffs = []
            n = len(shared)
            for _ in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                diffs.append(float(a[idx].mean() - b[idx].mean()))
            diffs_array = np.asarray(diffs)
            pairs[f"{method_a}_vs_{method_b}"] = {
                "n_pairs": int(n),
                "paired_keys": sorted(keys.tolist()),
                "success_rate_a": float(a.mean()),
                "success_rate_b": float(b.mean()),
                "mean_difference": float(a.mean() - b.mean()),
                "difference_ci95_bootstrap": [
                    float(np.percentile(diffs_array, 2.5)),
                    float(np.percentile(diffs_array, 97.5)),
                ],
                "a_wins_on_pair": int((a & ~b).sum()),
                "b_wins_on_pair": int((b & ~a).sum()),
                "ties": int((a == b).sum()),
            }
    return pairs


def distribution_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    def _collect(field: str, sub: str | None = None) -> list[float]:
        values = []
        for result in results:
            stats = result.get("summary_stats") or {}
            if sub is None:
                value = stats.get(field)
            else:
                value = (stats.get(field) or {}).get(sub)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return values

    trigger_reasons: dict[str, int] = defaultdict(int)
    fallback_fraction = []
    obs_ages = []
    plan_ages = []
    action_changes = []
    for result in results:
        stats = result.get("summary_stats") or {}
        for reason, count in (stats.get("trigger_reasons") or {}).items():
            trigger_reasons[reason] += int(count)
        for field, bucket in (
            ("fallback_fraction", fallback_fraction),
            ("input_obs_age_ticks_mean", obs_ages),
            ("active_plan_age_ticks_mean", plan_ages),
            ("action_change_mean_per_tick", action_changes),
        ):
            value = stats.get(field)
            if isinstance(value, (int, float)):
                bucket.append(float(value))

    def _stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {}
        array = np.asarray(values)
        return {
            "mean": float(array.mean()),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "min": float(array.min()),
            "max": float(array.max()),
            "n": len(values),
        }

    return {
        "trigger_reasons_total": dict(sorted(trigger_reasons.items())),
        "fallback_fraction": _stats(fallback_fraction),
        "input_obs_age_ticks": _stats(obs_ages),
        "active_plan_age_ticks": _stats(plan_ages),
        "action_change_per_tick": _stats(action_changes),
    }


def write_summary(experiment_dir: str | Path, output: str | Path | None = None) -> Path:
    summary = summarize_experiment(experiment_dir)
    target = Path(output) if output else Path(experiment_dir) / "summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(target)
    return target


def write_results_csv(experiment_dir: str | Path, output: str | Path | None = None) -> Path:
    import csv

    results = load_trial_results(experiment_dir)
    target = Path(output) if output else Path(experiment_dir) / "results.csv"
    fields = [
        "trial_id",
        "task",
        "method",
        "latency",
        "obs_delay_ticks",
        "inference_delay_ticks",
        "init_index",
        "seed",
        "success",
        "termination_reason",
        "steps",
        "first_success_step",
        "n_requests",
        "requests_per_sim_second",
        "inference_seconds_total",
        "wall_seconds",
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for result in sorted(
            results, key=lambda r: (r["task"], r["method"], r["latency"], r["init_index"])
        ):
            row = dict(result)
            stats = result.get("summary_stats") or {}
            row["requests_per_sim_second"] = stats.get("requests_per_sim_second")
            writer.writerow(row)
    return target
