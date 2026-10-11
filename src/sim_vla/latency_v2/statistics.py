"""Fixed-task descriptive intervals and correctly paired high-latency inference."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..assets import project_root
from ..tasks import TASK_BY_KEY
from .experiments import completed_results
from .layout import completion_flags, load_layout
from .storage import canonical_hash, publish


def wilson(successes: int, count: int) -> list[float]:
    if not count:
        raise ValueError("zero denominator")
    z = 1.959963984540054
    p = successes/count
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    radius = z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [float(center-radius), float(center+radius)]


def paired_bootstrap(rows: list[dict], candidate: str, baseline: str, *, draws: int = 10000, seed=20261001) -> dict:
    indexed = {(row["task"], row["init_index"], row["latency"], row["config"]["method"]): row for row in rows}
    differences, costs = [], []
    unit_counts = {}
    for task in TASK_BY_KEY:
        inits = sorted({row["init_index"] for row in rows if row["task"] == task})
        task_difference, task_cost = [], []
        for index in inits:
            paired = [(indexed[(task, index, latency, candidate)], indexed[(task, index, latency, baseline)])
                      for latency in (10, 20)]
            task_difference.append(np.mean([int(c["success"])-int(b["success"]) for c, b in paired]))
            task_cost.append([np.mean([c["generation_seconds"] for c, _ in paired]),
                              np.mean([b["generation_seconds"] for _, b in paired])])
        differences.append(np.asarray(task_difference))
        costs.append(np.asarray(task_cost))
        unit_counts[task] = len(inits)
    generator = np.random.default_rng(seed)
    samples = np.zeros(draws)
    for difference in differences:
        sample_indices = generator.integers(0, len(difference), (draws, len(difference)))
        samples += np.mean(difference[sample_indices], axis=1)/len(TASK_BY_KEY)
    means = np.mean([np.mean(cost, axis=0) for cost in costs], axis=0)
    return {"candidate": candidate, "baseline": baseline,
            "gain": float(np.mean([np.mean(difference) for difference in differences])),
            "ci95": np.percentile(samples, [2.5, 97.5]).tolist(), "draws": draws, "seed": seed,
            "unit_counts": unit_counts, "units": "init within fixed task, both L10/L20 retained together",
            "candidate_mean_generation_s": float(means[0]), "baseline_mean_generation_s": float(means[1]),
            "generation_cost_ratio": float(means[0]/max(means[1], 1e-12))}


def cells(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["latency"], row.get("profile", "fixed"), row["config"]["method"])].append(row)
    output = []
    for (task, latency, profile, method), values in sorted(grouped.items()):
        n, successes = len(values), sum(value["success"] for value in values)
        output.append({"task": task, "latency": latency, "profile": profile, "method": method,
            "episodes": n, "successes": successes, "success_rate": successes/n, "wilson95": wilson(successes, n),
            "mean_requests": float(np.mean([row["n_requests"] for row in values])),
            "mean_requests_per_sim_second": float(np.mean([row["requests_per_sim_second"] for row in values])),
            "mean_generation_s": float(np.mean([row["generation_seconds"] for row in values])),
            "mean_generation_s_per_sim_second": float(np.mean([row["generation_seconds"] / max(.05*row["steps"], .05) for row in values])),
            "mean_wall_s": float(np.mean([row["wall_seconds"] for row in values]))})
    return output


def summarize(directory: Path | None = None, *, spec_path: Path | None = None) -> dict:
    root = project_root()
    _, frozen, run_dirs = load_layout(root, spec_path)
    directory = root/(directory or Path(frozen.get("documentation_dir", "docs/latency_v2")))
    all_rows = {name: completed_results(run_dirs[name]) for name in ("core", "jitter", "async")}
    for name, rows in all_rows.items():
        if len(rows) != frozen["counts"][name]:
            raise ValueError("formal denominator differs from the frozen matrix")
        fields = ("task", "init_index", "seed", "latency", "profile", "config")
        actual = sorted(canonical_hash({key: row[key] for key in fields}) for row in rows)
        if actual != sorted(canonical_hash(row) for row in frozen["formal_trial_matrices"][name]):
            raise ValueError("completed formal trial assignments differ from the global frozen matrix")
        # All methods must start from identical physics, excluding only later trajectories.
        grouped = defaultdict(set)
        for row in rows:
            grouped[(row["task"], row["init_index"])].add(row["initial_state_after_settle_sha256"])
        if any(len(digests) != 1 for digests in grouped.values()):
            raise ValueError("paired methods did not restore identical physical starts")
    main = paired_bootstrap(all_rows["core"], frozen["C_SELECTED"], frozen["B_SELECTED"])
    asynchronous = all_rows["async"]
    async_rates = {method: float(np.mean([np.mean([row["success"] for row in asynchronous if row["task"] == task
                         and row["config"]["method"] == method]) for task in TASK_BY_KEY]))
                   for method in ("B_RAW", "B_SHIFT", frozen["C_SELECTED"])}
    async_gain = async_rates[frozen["C_SELECTED"]]-async_rates[frozen["B_SELECTED"]]
    realtime_passed = bool(asynchronous) and all(row["fraction_cycles_under_55ms"] >= .95
        and row.get("no_catchup_or_skipped_steps", spec_path is None) for row in asynchronous)
    if (main["gain"] >= .10 and main["ci95"][0] > 0 and main["generation_cost_ratio"] <= 1.2
            and async_gain > 0 and realtime_passed):
        classification = "positive_within_fixed_tasks"
    elif main["gain"] > 0:
        classification = "limited_positive_not_meeting_full_threshold"
    else:
        classification = "negative_no_improvement_over_selected_strong_baseline"
    aggregate = []
    for latency in (2, 5, 10, 20):
        for method in ("B_RAW", "B_SHIFT", "C_PREFIX", "C_BLEND"):
            subset = [row for row in all_rows["core"] if row["latency"] == latency and row["config"]["method"] == method]
            aggregate.append({"latency": latency, "method": method,
                "success_rate_equal_task": float(np.mean([np.mean([row["success"] for row in subset
                                                        if row["task"] == task]) for task in TASK_BY_KEY])),
                "mean_requests": float(np.mean([row["n_requests"] for row in subset])),
                "mean_generation_s": float(np.mean([row["generation_seconds"] for row in subset])),
                "mean_generation_s_per_sim_second": float(np.mean([row["generation_seconds"] / max(.05*row["steps"], .05) for row in subset])),
                "mean_requests_per_sim_second": float(np.mean([row["requests_per_sim_second"] for row in subset]))})
    async_metrics = []
    for method in ("B_RAW", "B_SHIFT", frozen["C_SELECTED"]):
        subset = [row for row in asynchronous if row["config"]["method"] == method]
        async_metrics.append({"method": method, "success_rate": async_rates[method],
            "mean_actual_control_hz": float(np.mean([row["actual_control_hz"] for row in subset])),
            "mean_deadline_missed_fraction": float(np.mean([row["deadline_missed_fraction"] for row in subset])),
            "mean_fraction_cycles_under_55ms": float(np.mean([row["fraction_cycles_under_55ms"] for row in subset])),
            "mean_late_results": float(np.mean([row["late_results"] for row in subset])),
            "mean_fallback_fraction": float(np.mean([row["fallback_ticks"]/max(row["steps"], 1) for row in subset]))})
    report = {"formal_spec_sha256": frozen["spec_sha256"], "classification": classification,
        "main": main, "cells": {name: cells(rows) for name, rows in all_rows.items()},
        "core_aggregate": aggregate, "async_success_rates": async_rates, "async_gain": async_gain,
        "async_target_20hz_gate_passed": realtime_passed,
        "async_metrics": async_metrics, "counts": {name: len(rows) for name, rows in all_rows.items()},
        "fresh_physical_bank": True, "fixed_task_scope": True, "formal_scores_used_for_tuning": False}
    flags = [completion_flags(run_dirs[name], rows, frozen["counts"][name]) for name, rows in all_rows.items()]
    report.update(delivery_complete=all(flag["delivery_complete"] for flag in flags),
                  realtime_20hz_passed=realtime_passed,
                  timing_provenance_complete=all(flag["timing_provenance_complete"] for flag in flags))
    publish(directory/"summary.json", report)
    return report
