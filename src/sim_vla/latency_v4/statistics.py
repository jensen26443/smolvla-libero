"""Task-stratified physical-state pairing, never treating ticks as samples."""
from collections import defaultdict

import numpy as np

from ..latency_v2.statistics import wilson
from ..tasks import TASK_BY_KEY


def paired_gain(rows, candidate="RTC_CATCHUP", baseline="RTC_STRICT", *, draws=10000, seed=20261005):
    differences, unit_counts = [], {}
    for task in TASK_BY_KEY:
        subset = [row for row in rows if row["task"] == task and row["config"]["method"] in (candidate, baseline)]
        indexed = {}
        for row in subset:
            key = row["state_id"], row["config"]["method"]
            if key in indexed:
                raise ValueError("duplicate physical-state outcome; repeated stages cannot be pooled as independent samples")
            indexed[key] = row
        states = sorted({row["state_id"] for row in subset})
        if not states:
            raise ValueError("main paired comparison requires every fixed task")
        delta = []
        for state in states:
            try:
                c, b = indexed[(state, candidate)], indexed[(state, baseline)]
            except KeyError as failure:
                raise ValueError("missing paired method outcome") from failure
            if c["seed"] != b["seed"] or c["initial_state_after_settle_sha256"] != b["initial_state_after_settle_sha256"]:
                raise ValueError("paired methods did not share noise seed and physical start")
            delta.append(int(c["success"])-int(b["success"]))
        differences.append(np.asarray(delta, dtype=float))
        unit_counts[task] = len(states)
    generator = np.random.default_rng(seed)
    bootstrap = np.zeros(draws)
    for delta in differences:
        indices = generator.integers(0, len(delta), (draws, len(delta)))
        bootstrap += np.mean(delta[indices], axis=1)/len(TASK_BY_KEY)
    return {"candidate": candidate, "baseline": baseline,
        "gain": float(np.mean([np.mean(delta) for delta in differences])),
        "ci95": np.percentile(bootstrap, [2.5, 97.5]).tolist(), "draws": draws, "seed": seed,
        "unit_counts": unit_counts, "units": "physical initialization within each fixed task; tasks equally weighted"}


def cells(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["success"] is None:
            continue
        grouped[(row["task"], row["config"]["method"], row["config"]["sampling_steps"],
                 row["config"]["period"], row["latency"])].append(row)
    output = []
    for (task, method, steps, period, latency), subset in sorted(grouped.items()):
        n, successes = len(subset), sum(row["success"] for row in subset)
        wall = sum(row["wall_seconds"]+row["reset_wall_s"] for row in subset)
        output.append({"task": task, "method": method, "sampling_steps": steps, "period": period,
            "latency": latency, "n": n, "successes": successes, "success_rate": successes/n,
            "wilson95": wilson(successes, n), "total_task_wall_s": wall,
            "successful_tasks_per_wall_hour": 3600*successes/max(wall, 1e-9),
            "success_completion_wall_s": [row["wall_seconds"] for row in subset if row["success"]],
            "mean_generation_s": float(np.mean([row["generation_seconds"] for row in subset]))})
    return output


def classify(main, secondary, rows):
    activations = sum(row["catchup_accepted"] for row in rows if row["config"]["method"] == "RTC_CATCHUP")
    if activations == 0:
        return "mechanism_not_activated"
    if main["gain"] > 0 and main["ci95"][0] > 0:
        if all(item["gain"] > 0 and item["ci95"][0] > 0 for item in secondary):
            return "catchup_effect_supported_within_fixed_tasks"
        return "local_gain_over_strict_strong_baseline_gain_unproven"
    if main["gain"] < 0:
        return "negative_catchup_no_gain"
    return "uncertain_success_gain_no_equivalence_claim"
