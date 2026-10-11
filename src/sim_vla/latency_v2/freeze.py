"""Freeze all formal comparisons before reading the first formal policy outcome."""
from __future__ import annotations

import itertools
import json
import os
import shutil
from pathlib import Path

from ..assets import project_root
from ..tasks import TASK_BY_KEY
from .experiments import (
    CHECKPOINT,
    base_trial,
    development_baselines,
    development_candidates,
    semantic_inventory,
)
from .initializations import check_formal_manifest
from .integrity import file_sha256, inventory
from .storage import Budget, canonical_hash, publish, trial_key


def async_development_trials(baselines: dict, candidates: dict) -> list[dict]:
    selected = candidates["C_SELECTED"]
    trials = []
    for task in TASK_BY_KEY:
        for method in ("B_RAW", "B_SHIFT", selected):
            source = baselines if method.startswith("B_") else candidates
            config = source["latencies"]["20"][method]["config"]
            trials.append(base_trial(task, 0, 20, config))
    # The first three are plate development episodes; the same matrix covers all tasks/methods.
    return trials


def freeze_formal(baselines: dict, candidates: dict, manifest_path: Path, *, directory: Path) -> dict:
    root = project_root()
    if (directory/"experiment_spec.json").exists():
        raise FileExistsError("all formal specs already frozen; preserve them")
    audit = json.loads((root/"runs/latency_v2_audit/audit.json").read_text())
    checked = check_formal_manifest(manifest_path, audit["initializations"])
    bank = json.loads(manifest_path.read_text())
    for task, entry in bank["tasks"].items():
        if "model_poses_path" not in checked["files"][task]:
            raise ValueError(f"{task}: fixture placements are missing from physical initialization assets")
        validation = manifest_path.parent/entry["reset_validation_path"]
        if (not entry["compatibility_gate_passed"] or file_sha256(validation) != entry["reset_validation_sha256"]):
            raise ValueError(f"{task}: physical-state task/reset validity is not certified")
        rows = json.loads(validation.read_text())
        if (len(rows) != 20 or any(row["image_repeat_max_difference"] != 0 or row["initial_success"]
                                 or row["min_qpos_linf_difference_from_prior"] <= 1e-8
                                 or not row.get("cross_seed_scene_repeat_passed")
                                 or not row.get("model_fixture_poses_restored") for row in rows)):
            raise ValueError("physical-state reset validation failed")
    if file_sha256(root/CHECKPOINT/"model.safetensors") != "e65416ea9ef2fc80f3e29aff5c70f972aa5009518c675d3cbaf4fca4db6892e8":
        raise ValueError("E1 checkpoint changed after S0")
    if not (root/"runs/latency_v2_dev_async/complete.json").exists():
        raise ValueError("G3 requires completed real asynchronous development smoke")
    probe_path = root/"runs/latency_v2_conditioning_probe/probe.json"
    if not (probe_path.parent/"complete.json").exists():
        raise ValueError("G3 requires the fixed-observation/noise conditioning probe")
    probe = json.loads(probe_path.read_text())
    if not probe["private_wrapper_without_projection_bitwise_native_equal"]:
        raise ValueError("private sampler prefix/cache wrapper does not match the native sampler")
    resource_path = root/"runs/latency_v2_resource_probe/probe.json"
    if not (resource_path.parent/"complete.json").exists():
        raise ValueError("G3 requires actual native/projected GPU memory and timing measurements")
    resources = json.loads(resource_path.read_text())
    pilot = json.loads((root/"runs/latency_v2_dev_baselines/pilot_cost.json").read_text())
    # Predeclared development-only stopping criterion, never a formal score callback.
    main_baseline, main_candidate = baselines["B_SELECTED"], candidates["C_SELECTED"]
    gains = [candidates["latencies"][str(latency)][main_candidate]["success_rate_equal_task"]
             -baselines["latencies"][str(latency)][main_baseline]["success_rate_equal_task"] for latency in (10, 20)]
    clearly_ineffective = all(gain <= -.10 for gain in gains)
    # PLAN-04 permits a smaller matrix BEFORE formal evaluation for measured cost
    # or clear negative development evidence; the exact decision is frozen here.
    expensive = pilot["default_total_1698_hours"] > 24
    reduced = clearly_ineffective or expensive
    core_inits = 5 if reduced else 20
    supplemental_inits = 5 if reduced else 10
    budget = Budget(root/"runs/latency_v2_budget.json")
    budget_reduced = False
    if 48*core_inits+21*supplemental_inits+32 > budget.remaining():
        options = [size for size in (20, 15, 10, 5) if size <= core_inits
                   and 48*size+21*supplemental_inits+32 <= budget.remaining()]
        if not options:
            raise RuntimeError("remaining attempts cannot fund a minimum fair formal matrix and repair reserve")
        core_inits = max(options)
        budget_reduced = True
    core, jitter, asynchronous = [], [], []
    for task, latency, init, method in itertools.product(TASK_BY_KEY, (2, 5, 10, 20), range(core_inits),
                                                        ("B_RAW", "B_SHIFT", "C_PREFIX", "C_BLEND")):
        source = baselines if method.startswith("B_") else candidates
        config = source["latencies"][str(latency)][method]["config"]
        core.append(base_trial(task, init, latency, config))
    for task, profile, init, method in itertools.product(tuple(TASK_BY_KEY)[:2], ("J1", "J2"),
                                              range(supplemental_inits), ("B_RAW", "B_SHIFT", main_candidate)):
        latency = 5 if profile == "J1" else 20
        source = baselines if method.startswith("B_") else candidates
        jitter.append(base_trial(task, init, latency, source["latencies"][str(latency)][method]["config"], profile=profile))
    for task, init, method in itertools.product(TASK_BY_KEY, range(supplemental_inits), ("B_RAW", "B_SHIFT", main_candidate)):
        source = baselines if method.startswith("B_") else candidates
        asynchronous.append(base_trial(task, init, 20, source["latencies"]["20"][method]["config"]))
    budget = Budget(root/"runs/latency_v2_budget.json")
    if sum(map(len, (core, jitter, asynchronous))) > budget.remaining():
        raise RuntimeError("all frozen formal comparisons exceed remaining attempts")
    modules = root/"src/sim_vla/latency_v2"
    implementation = inventory(root, list(modules.glob("*.py")) + list((root/"tests").glob("test_latency_v2*.py"))
        + [root/"scripts"/name for name in ("probe_latency_v2_conditioning.py", "probe_latency_v2_resources.py",
                                            "analyze_latency_v2_development_pair.py", "run_latency_v2.sh",
                                            "queue_latency_v2.py")])
    matrices = {"core": core, "jitter": jitter, "async": asynchronous}
    templates = {name: {"stage": "formal_"+name, "run_dir": "runs/latency_v2_formal_"+name,
        "checkpoint": CHECKPOINT, "backend": "async" if name == "async" else "simulated", "max_ticks": 300,
        "renderer": "glx_d3d12_rtx3090", "initialization_manifest": str(manifest_path.relative_to(root)),
        "trials_path": str((directory/(name+"_trials.json")).relative_to(root)),
        "formal_spec_path": str((directory/"experiment_spec.json").relative_to(root))} for name in matrices}
    spec = {"format": "latency-v2-formal-spec-v1", "checkpoint": CHECKPOINT,
        "checkpoint_inventory": audit["checkpoint"], "model_sha256": "e65416ea9ef2fc80f3e29aff5c70f972aa5009518c675d3cbaf4fca4db6892e8",
        "vlm_revision": "7b375e1b73b11138ff12fe22c8f2822d8fe03467", "versions": audit["versions"],
        "assets_sha256": audit["libero_assets"]["sha256"], "physical_manifest": str(manifest_path.relative_to(root)),
        "physical_manifest_sha256": file_sha256(manifest_path), "physical_states": checked,
        "semantic_code": semantic_inventory(root, asynchronous=True), "implementation_inventory": implementation,
        "baseline_selection": baselines, "candidate_selection": candidates,
        "B_SELECTED": main_baseline, "C_SELECTED": main_candidate,
        "development_only_scale_decision": {"reduced": reduced or budget_reduced, "clearly_ineffective": clearly_ineffective,
            "budget_reduced_before_any_formal_score": budget_reduced, "repair_attempts_reserved": 32,
            "development_high_latency_gains": gains, "estimated_default_hours": pilot["default_total_1698_hours"],
            "measured_cost_over_24h": expensive, "core_inits": core_inits, "supplemental_inits": supplemental_inits},
        "sampler": {"steps": 10, "dt": -.1, "native_normalization": True, "valid_action_dims": 7,
                    "padded_action_dims": 32, "prefix_projection": "before and after every Euler step"},
        "timing": {"control_hz": 20, "max_ticks": 300, "fallback_limit": 60, "single_flight": True,
                   "late_prefix_policy": "discard, neutral fallback, urgent new request", "expired_shift_policy": "discard",
                   "blend_dimensions": [0, 1, 2], "rotation_and_gripper": "new action unchanged",
                   "async_estimator": {"initial_p95_seconds": 1.10448796484, "window": 10, "statistic": "completed endpoint p95"}},
        "jitter": {"J1": {"values": [3, 5, 7], "probabilities": [.25, .5, .25]},
                   "J2": {"values": [10, 20, 40], "probabilities": [.4, .4, .2]},
                   "seed": "sha256(task/init/profile/20261001)", "indexed_by": "control tick",
                   "candidate_estimator": "known mean initially, past 10 completed delays thereafter"},
        "statistics": {"main": "C_SELECTED versus B_SELECTED across L10 and L20, equal task weight",
            "bootstrap": "10000 paired samples, resample init within each task, retain both high latencies and all methods",
            "bootstrap_seed": 20261001, "cell_interval": "Wilson 95%", "claim_scope": "fixed three supported tasks",
            "positive_threshold": "gain >=10pp, paired CI lower >0, generation cost <=20% increase, same async direction, and >=95% cycles <=55ms per async episode"},
        "video_rule": {"raw": "all formal episodes from cached observations, no extra render or rollout",
                       "main_examples": "minimum bank index 0, L20, each task, regardless of success",
                       "failure_example": "first failed C_SELECTED core L20 in task order then init order; otherwise first failed B_SELECTED with the same order; illustration only, no score tuning"},
        "counts": {"core": len(core), "jitter": len(jitter), "async": len(asynchronous)},
        "formal_trial_matrices": matrices, "formal_run_config_templates": templates,
        "conditioning_probe": {"path": str(probe_path.relative_to(root)), "sha256": file_sha256(probe_path),
                               "native_wrapper_equivalent": True,
                               "normalized_suffix_linf_difference": probe["normalized_suffix_linf_difference"],
                               "prefix_coupling_detected": probe["prefix_coupling_detected"]},
        "resource_probe": {"path": str(resource_path.relative_to(root)), "sha256": file_sha256(resource_path),
                           "memory_scope": resources["memory_scope"],
                           "rows": [{key: value for key, value in row.items() if key != "samples"}
                                    for row in resources["rows"]]},
        "budget_before_formal": budget.remaining(), "formal_score_callbacks": False,
        "development_counts": {"baselines": len(development_baselines()), "candidates": len(development_candidates(baselines))}}
    spec["spec_sha256"] = canonical_hash(spec)
    directory.mkdir(parents=True, exist_ok=True)
    for name, trials in matrices.items():
        if len({trial_key(trial) for trial in trials}) != len(trials):
            raise ValueError("duplicate frozen formal trial")
        publish(directory/(name+"_trials.json"), trials)
        config = {**templates[name], "formal_spec_sha256": spec["spec_sha256"]}
        publish(directory/(name+"_config.json"), config)
    snapshot = directory/"code_snapshot"
    for relative in set(spec["semantic_code"]) | {row["path"] for row in implementation["files"]}:
        target = snapshot/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/relative, target)
    publish(directory/"experiment_spec.json", spec)
    return spec


def verify_frozen(config: dict) -> None:
    root = project_root()
    spec = json.loads((root/config["formal_spec_path"]).read_text())
    expected = spec["spec_sha256"]
    if config["formal_spec_sha256"] != expected or canonical_hash({k: v for k, v in spec.items() if k != "spec_sha256"}) != expected:
        raise ValueError("formal specification fingerprint mismatch")
    name = config["stage"].removeprefix("formal_")
    if name not in spec["formal_run_config_templates"]:
        raise ValueError("unexpected formal stage")
    if canonical_hash(config) != canonical_hash({**spec["formal_run_config_templates"][name], "formal_spec_sha256": expected}):
        raise ValueError("formal run configuration differs from the frozen specification")
    if canonical_hash(json.loads((root/config["trials_path"]).read_text())) != canonical_hash(spec["formal_trial_matrices"][name]):
        raise ValueError("formal trial matrix differs from the frozen specification")
    for relative, digest in spec["semantic_code"].items():
        if file_sha256(root/relative) != digest:
            raise ValueError(f"formal semantics changed: {relative}")
    for entry in spec["implementation_inventory"]["files"]:
        if file_sha256(root/entry["path"]) != entry["sha256"]:
            raise ValueError("formal implementation/test files changed after freeze")
    if file_sha256(root/spec["physical_manifest"]) != spec["physical_manifest_sha256"]:
        raise ValueError("formal physical manifest changed after freeze")
    for row in spec["physical_states"]["files"].values():
        if file_sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError("formal physical state array changed after freeze")
        if file_sha256(Path(row["model_poses_path"])) != row["model_poses_sha256"]:
            raise ValueError("formal fixture model transforms changed after freeze")
    for entry in spec["checkpoint_inventory"]["files"]:
        if file_sha256(root/spec["checkpoint"]/entry["path"]) != entry["sha256"]:
            raise ValueError("formal checkpoint/processor assets changed after S0")
    if os.environ.get("MUJOCO_GL") != "glx" or os.environ.get("GALLIUM_DRIVER") != "d3d12":
        raise ValueError("formal renderer differs from frozen glx/d3d12; use scripts/run_latency_v2.sh")
