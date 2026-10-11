"""Deterministic development matrices, selection, and resumable real runs."""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..assets import project_root
from ..tasks import TASK_BY_KEY
from .integrity import file_sha256
from .layout import completion_flags, write_progress
from .storage import Budget, canonical_hash, prepare_run, publish, trial_key

CHECKPOINT = "runs/train_100pct_formal/checkpoints/010000/pretrained_model"


def semantic_inventory(root: Path, *, asynchronous: bool = False) -> dict:
    names = ["schema.py", "alignment.py", "env.py", "model_poses.py", "initializations.py",
             "policy.py", "runner_sim.py", "storage.py", "experiments.py", "timing.py", "env_timing.py", "layout.py"]
    if asynchronous:
        names += ["runner_async.py", "worker.py"]
    paths = [root / "src/sim_vla/latency_v2" / name for name in names]
    paths += [root / "src/sim_vla/latency_eval" / name for name in ("policy_adapter.py", "env_adapter.py", "contracts.py")]
    paths += [root / "src/sim_vla" / name for name in ("assets.py", "runtime.py", "tasks.py")]
    return {str(path.relative_to(root)): file_sha256(path) for path in paths}


def base_trial(task: str, init: int, latency: int, config: dict, *, profile: str = "fixed") -> dict:
    # Same policy noise stream for paired methods, independent of simulator placement RNG.
    seed = int(canonical_hash([task, init, 42])[:8], 16) % (2**31-1)
    return {"task": task, "init_index": init, "seed": seed, "latency": latency,
            "profile": profile, "config": config}


def development_baselines() -> list[dict]:
    # First 18 are the predeclared cost pilot: every task/L and both methods, K=40/init=0.
    trials = []
    for period, init, task, latency, method in itertools.product(
            (40, 5, 10, 20, 25, 50), range(3), TASK_BY_KEY, (5, 10, 20), ("B_RAW", "B_SHIFT")):
        trials.append(base_trial(task, init, latency, {"method": method, "period": period}))
    assert len(trials) == 324
    return trials


def development_candidates(selection: dict) -> list[dict]:
    configs = [("C_PREFIX", 0, 0), ("C_PREFIX", 2, 0), ("C_BLEND", 2, 2), ("C_BLEND", 2, 5)]
    trials = []
    for task, latency, init, (method, margin, blend) in itertools.product(
            TASK_BY_KEY, (5, 10, 20), range(5), configs):
        period = selection["latencies"][str(latency)]["B_SELECTED"]["config"]["period"]
        trials.append(base_trial(task, init, latency,
                                 {"method": method, "period": period, "margin": margin, "blend": blend}))
    assert len(trials) == 180
    return trials


def completed_results(directory: Path, *, require_complete: bool = True) -> list[dict]:
    expected = json.loads((directory / "expected_trials.json").read_text())
    results = []
    for trial in expected:
        path = directory / "trials" / trial["key"] / "result.json"
        if path.exists():
            result = json.loads(path.read_text())
            for key, value in trial.items():
                if key != "key" and result[key] != value:
                    raise ValueError("trial metadata differs from expected manifest")
            results.append(result)
    if require_complete and len(results) != len(expected):
        raise ValueError(f"incomplete run: {len(results)}/{len(expected)}")
    return results


def score(rows: list[dict]) -> dict:
    task_rates = [np.mean([row["success"] for row in rows if row["task"] == task]) for task in TASK_BY_KEY]
    return {"success_rate_equal_task": float(np.mean(task_rates)),
            "mean_generation_seconds": float(np.mean([row["generation_seconds"] for row in rows])),
            "mean_requests": float(np.mean([row["n_requests"] for row in rows])),
            "mean_request_frequency": float(np.mean([row["requests_per_sim_second"] for row in rows])),
            "episodes": len(rows)}


def select_baselines(directory: Path) -> dict:
    rows = completed_results(directory)
    selection = {"development_run": str(directory), "rule": "equal-task success, lower mean generation time, smaller K",
                 "latencies": {}, "scan": []}
    for latency in (5, 10, 20):
        candidates = {}
        for method in ("B_RAW", "B_SHIFT"):
            options = []
            for period in (5, 10, 20, 25, 40, 50):
                subset = [row for row in rows if row["latency"] == latency and row["config"] == {
                    "method": method, "period": period}]
                item = {"latency": latency, "config": {"method": method, "period": period}, **score(subset)}
                options.append(item)
                selection["scan"].append(item)
            candidates[method] = min(options, key=lambda row: (-row["success_rate_equal_task"],
                                      row["mean_generation_seconds"], row["config"]["period"]))
        candidates["B_SELECTED"] = min([candidates["B_RAW"], candidates["B_SHIFT"]], key=lambda row: (
            -row["success_rate_equal_task"], row["mean_generation_seconds"], row["config"]["method"]))
        selection["latencies"][str(latency)] = candidates
    # One identity across all formal latencies; per-L parameters remain method-specific.
    overall = {}
    for method in ("B_RAW", "B_SHIFT"):
        chosen = [selection["latencies"][str(latency)][method] for latency in (5, 10, 20)]
        overall[method] = (np.mean([row["success_rate_equal_task"] for row in chosen]),
                           np.mean([row["mean_generation_seconds"] for row in chosen]))
    selection["B_SELECTED"] = min(overall, key=lambda method: (-overall[method][0], overall[method][1], method))
    # Candidate period follows the development-selected main baseline identity at each L.
    for latency in (5, 10, 20):
        selection["latencies"][str(latency)]["B_SELECTED"] = selection["latencies"][str(latency)][selection["B_SELECTED"]]
    selection["latencies"]["2"] = selection["latencies"]["5"]
    publish(directory / "baseline_selection.json", selection)
    return selection


def select_candidates(directory: Path) -> dict:
    rows = completed_results(directory)
    output = {"development_run": str(directory), "latencies": {}, "scan": [],
              "rule": "equal-task success, lower generation time, margin/blend ascending"}
    for latency in (5, 10, 20):
        choices = {}
        grouped = defaultdict(list)
        for row in rows:
            if row["latency"] == latency:
                grouped[canonical_hash(row["config"])].append(row)
        for subset in grouped.values():
            item = {"latency": latency, "config": subset[0]["config"], **score(subset)}
            output["scan"].append(item)
        for method in ("C_PREFIX", "C_BLEND"):
            options = [row for row in output["scan"] if row["latency"] == latency and row["config"]["method"] == method]
            choices[method] = min(options, key=lambda row: (-row["success_rate_equal_task"],
                row["mean_generation_seconds"], row["config"]["margin"], row["config"]["blend"]))
        output["latencies"][str(latency)] = choices
    output["C_SELECTED"] = min(("C_PREFIX", "C_BLEND"), key=lambda method: (
        -np.mean([output["latencies"][str(latency)][method]["success_rate_equal_task"] for latency in (5, 10, 20)]),
        np.mean([output["latencies"][str(latency)][method]["mean_generation_seconds"] for latency in (5, 10, 20)]), method))
    output["latencies"]["2"] = output["latencies"]["5"]
    publish(directory / "candidate_selection.json", output)
    return output


def run_matrix(config: dict, trials: list[dict], *, resume: bool = False, dry_run: bool = False) -> dict:
    root = project_root()
    directory = root / config["run_dir"]
    asynchronous = config.get("backend") == "async"
    code = semantic_inventory(root, asynchronous=asynchronous)
    spec = {"stage": config["stage"], "config": config, "semantic_code": code,
            "semantic_code_sha256": canonical_hash(code), "checkpoint": config.get("checkpoint", CHECKPOINT),
            "model_sha256": "e65416ea9ef2fc80f3e29aff5c70f972aa5009518c675d3cbaf4fca4db6892e8",
            "selection_rule": "development only", "max_ticks": config.get("max_ticks", 300)}
    if dry_run:
        return {"count": len(trials), "unique_keys": len({trial_key(row) for row in trials}),
                "run_dir": str(directory), "spec": spec}
    frozen = prepare_run(directory, spec, trials, resume=resume)
    budget = Budget(root / "runs/latency_v2_budget.json")
    states, poses = {}, {}
    if config.get("initialization_manifest"):
        from .initializations import check_formal_manifest

        manifest_path = root / config["initialization_manifest"]
        audit = json.loads((root / "runs/latency_v2_audit/audit.json").read_text())
        checked = check_formal_manifest(manifest_path, audit["initializations"])
        states = {task: np.load(row["path"], allow_pickle=False) for task, row in checked["files"].items()}
        from .model_poses import ModelPoseBank

        poses = {task: ModelPoseBank.load(Path(row["model_poses_path"]), expected_states=len(states[task]))
                 for task, row in checked["files"].items()}
    elif config["stage"].startswith("formal"):
        raise ValueError("formal stage requires certified physical-state manifest")
    if len(trials)-len(completed_results(directory, require_complete=False)) > budget.remaining():
        raise RuntimeError("remaining matrix exceeds available attempt budget")
    import torch

    from .env import ControlledEnv
    from .policy import PrefixPolicyAdapter
    from .runner_sim import run_episode

    if file_sha256(root/spec["checkpoint"]/"model.safetensors") != spec["model_sha256"]:
        raise ValueError("frozen E1 model file differs from S0 full-weight fingerprint")
    torch.set_num_threads(4)
    policy = PrefixPolicyAdapter(checkpoint=root/spec["checkpoint"], warmup_calls=3)
    policy.warmup()
    envs = {}
    try:
        for ordinal, trial in enumerate(trials):
            key = trial_key(trial)
            target = directory / "trials" / key
            if (target / "result.json").exists():
                continue
            budget.prepare_retry(target)
            with budget.track(trial_key(trial), config["stage"], target, trial,
                              run_id=config.get("run_id", directory.name)) as attempt:
                if trial["task"] not in envs:
                    envs[trial["task"]] = ControlledEnv(task_key=trial["task"],
                        observation_mode=config.get("observation_mode", "per_tick"),
                        profile_env=config.get("profile_env", False))
                    envs[trial["task"]].model_pose_bank = poses.get(trial["task"])
                sample = config["stage"].startswith("formal") or (trial["init_index"] == 0 and (
                    config["stage"] == "dev_candidates" or (trial["latency"] == 20 and trial["config"]["period"] == 40)))
                result = run_episode(envs[trial["task"]], policy, trial, target, states=states.get(trial["task"]),
                                     max_ticks=config.get("max_ticks", 300), video=sample,
                                     generation=attempt, version=spec["semantic_code_sha256"])
            write_progress(config, directory, completed_results(directory, require_complete=False), len(trials), budget.remaining())
            print(json.dumps({"stage": config["stage"], "completed": ordinal+1, "total": len(trials),
                              "trial": trial, "success": result["success"], "reason": result["termination_reason"],
                              "seconds": round(result["wall_seconds"], 2), "budget_remaining": budget.remaining()}), flush=True)
            if ordinal == 17 and config["stage"] == "dev_baselines":
                pilot = completed_results(directory, require_complete=False)
                seconds = [row["wall_seconds"] for row in pilot]
                publish(directory / "pilot_cost.json", {"episodes": len(pilot), "mean_wall_s": float(np.mean(seconds)),
                        "p95_wall_s": float(np.percentile(seconds, 95)), "default_total_1698_hours": float(np.mean(seconds)*1698/3600),
                        "note": "pilot episodes are part of the full baseline scan, no duplicated attempts"})
            if result["error"]:
                raise RuntimeError(f"method implementation failed: {result['error']}; stop scan, fix and start new semantics")
    finally:
        for env in envs.values():
            env.close()
    rows = completed_results(directory)
    publish(directory / "complete.json", {"completed": len(rows), "spec_sha256": frozen["spec_sha256"],
                                           "remaining_budget": budget.remaining(),
                                           **completion_flags(directory, rows, len(trials))})
    return {"completed": len(rows), "run_dir": str(directory)}
