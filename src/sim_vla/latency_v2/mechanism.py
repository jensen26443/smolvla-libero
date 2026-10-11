"""Offline provenance checks and action-switch diagnostics; never policy inputs."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .experiments import completed_results
from .storage import publish, trial_key


def trial_metrics(directory: Path) -> dict:
    rows = [json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines()]
    arrays = np.load(directory/"arrays.npz", allow_pickle=False)
    actions = arrays["actions"]
    requests = json.loads((directory/"requests.json").read_text())
    steps = [row for row in rows if row["event"] == "step"]
    violations = []
    for index, request in enumerate(requests):
        start, length = request["t_obs"], request["prefix_length"]
        available = min(length, len(actions)-start)
        if available > 0:
            violations.append(float(np.max(np.abs(actions[start:start+available]-arrays["prefixes"][index, :available]))))
    switch_ticks = [row["tick"] for row in rows if row["event"] == "activate" and 0 < row["tick"] < len(actions)]
    translation = [float(np.linalg.norm(actions[tick, :3]-actions[tick-1, :3])) for tick in switch_ticks]
    rotation = [float(np.linalg.norm(actions[tick, 3:6]-actions[tick-1, 3:6])) for tick in switch_ticks]
    # A copied commitment still executes the OLD model plan. Its pending new
    # snapshot is not the observation that originally generated these actions.
    request_observations = {row["request_id"]: row["t_obs"] for row in requests}
    active_obs, ages, pending_prefix_end = None, [], None
    neutral_causes = Counter()
    for row in rows:
        if row["event"] == "submit":
            pending_prefix_end = row["prefix_end"] if row["prefix_length"] else None
        elif row["event"] == "activate":
            active_obs = request_observations[row["request_id"]]
            pending_prefix_end = None
        elif row["event"] == "late_discard":
            active_obs = None
            pending_prefix_end = None
        elif row["event"] == "expired_discard":
            pending_prefix_end = None
        elif row["event"] == "step":
            if not row["neutral"] and active_obs is not None:
                ages.append(row["tick"]-active_obs)
            elif row["neutral"]:
                if active_obs is None:
                    cause = "without_active_model_plan"
                elif row["source"] == "committed":
                    cause = "committed_prefix_missing_old_buffer"
                elif pending_prefix_end is not None and row["tick"] >= pending_prefix_end:
                    cause = "waiting_after_prefix_lock"
                else:
                    cause = "active_plan_exhausted"
                neutral_causes[cause] += 1
    pending, violations_single = None, 0
    for row in rows:
        if row["event"] == "submit":
            violations_single += int(pending is not None)
            pending = row["request_id"]
        elif row["event"] in {"activate", "late_discard", "expired_discard"}:
            violations_single += int(pending != row["request_id"])
            pending = None
    return {"max_executed_prefix_violation": max(violations, default=0.),
        "mean_observation_age_ticks": float(np.mean(ages)) if ages else None,
        "activation_ticks": switch_ticks, "switch_translation_action_norms": translation,
        "switch_rotation_action_norms": rotation,
        "gripper_sign_change_count": int(np.count_nonzero(np.diff(np.sign(actions[:, -1])))) if len(actions) else 0,
        "fallback_fraction": sum(row["neutral"] for row in steps)/max(len(steps), 1),
        "neutral_causes_ticks": dict(neutral_causes),
        "committed_old_buffer_shortage_ticks": neutral_causes["committed_prefix_missing_old_buffer"],
        "committed_control_fraction": sum(row["source"] == "committed" for row in steps)/max(len(steps), 1),
        "late_results": sum(row["event"] == "late_discard" for row in rows),
        "expired_results": sum(row["event"] == "expired_discard" for row in rows),
        "single_flight_violations": violations_single}


def analyze(directory: Path, output: Path) -> dict:
    results = completed_results(directory)
    grouped = defaultdict(list)
    all_metrics = []
    for result in results:
        trial = {key: result[key] for key in ("task", "init_index", "seed", "latency", "profile", "config")}
        metrics = trial_metrics(directory/"trials"/trial_key(trial))
        item = {**trial, "success": result["success"], "failure_stage": result["failure_stage_heuristic"], **metrics}
        all_metrics.append(item)
        grouped[(result["latency"], result["config"]["method"])].append(item)
    aggregate = []
    for (latency, method), values in sorted(grouped.items()):
        translation = [n for row in values for n in row["switch_translation_action_norms"]]
        rotation = [n for row in values for n in row["switch_rotation_action_norms"]]
        ages = [row["mean_observation_age_ticks"] for row in values if row["mean_observation_age_ticks"] is not None]
        aggregate.append({"latency": latency, "method": method,
            "mean_observation_age_ticks": float(np.mean(ages)) if ages else 0.,
            "switch_translation_mean_action_norm": float(np.mean(translation)) if translation else 0.,
            "switch_rotation_mean_action_norm": float(np.mean(rotation)) if rotation else 0.,
            "mean_gripper_sign_changes": float(np.mean([row["gripper_sign_change_count"] for row in values])),
            "mean_fallback_fraction": float(np.mean([row["fallback_fraction"] for row in values])),
            "mean_committed_old_buffer_shortage_ticks": float(np.mean([row["committed_old_buffer_shortage_ticks"] for row in values])),
            "neutral_causes_ticks": dict(sum((Counter(row["neutral_causes_ticks"]) for row in values), Counter())),
            "prefix_violation_max": max(row["max_executed_prefix_violation"] for row in values),
            "mean_late_results": float(np.mean([row["late_results"] for row in values])),
            "failure_stage_counts_heuristic": dict(Counter(row["failure_stage"] for row in values))})
    output_data = {"run": str(directory), "trial_metrics": all_metrics, "aggregate": aggregate,
        "observation_age_scope": "age of active model-plan component for non-neutral actions; copied commitments retain old plan age; blended translation also contains an older component",
        "stage_label_scope": "offline object-height/eef-distance heuristic, not a ground-truth failure causal label"}
    if any(row["max_executed_prefix_violation"] > 1e-12 for row in all_metrics):
        raise ValueError("executed actions violated the committed sampler prefix")
    if any(row["single_flight_violations"] for row in all_metrics):
        raise ValueError("single-flight request/activation provenance violated")
    publish(output, output_data)
    return output_data
