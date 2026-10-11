"""Small predeclared development grids, frozen manifests and shared accounting."""
import itertools
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

from ..assets import project_root
from ..latency_v2.experiments import CHECKPOINT, base_trial
from ..latency_v2.integrity import file_sha256
from ..latency_v2.locking import run_lock
from ..latency_v2.model_poses import ModelPoseBank
from ..latency_v2.storage import Budget, canonical_hash, prepare_run, publish, trial_key
from ..tasks import TASK_BY_KEY
from .execution_settings import capture_execution_settings, save_probe_snapshot, verify_execution_settings
from .schema import METHODS

RUN_ID = os.environ.get("SIM_VLA_LATENCY_V3_RUN_ID", "latency_v3_plan05")
if not re.fullmatch(r"latency_v3_[A-Za-z0-9_-]+", RUN_ID):
    raise ValueError("SIM_VLA_LATENCY_V3_RUN_ID must be a simple latency_v3_ run name")
ROOT = project_root()
RUN = ROOT/"runs"/RUN_ID
DOCS = ROOT/"docs/latency_v3" if RUN_ID == "latency_v3_plan05" else ROOT/"docs/latency_v3"/RUN_ID
CONFIG = (ROOT/"configs/latency_v3/experiment_spec.json" if RUN_ID == "latency_v3_plan05"
          else ROOT/"configs/latency_v3"/RUN_ID/"experiment_spec.json")
BANK = ROOT/"runs/latency_v2_physical_bank/manifest.json"
MODEL_SHA = "e65416ea9ef2fc80f3e29aff5c70f972aa5009518c675d3cbaf4fca4db6892e8"


def read(path):
    return json.loads(Path(path).read_text())


def source_inventory():
    paths = list((ROOT/"src/sim_vla/latency_v3").glob("*.py"))
    paths += list((ROOT/"src/sim_vla/latency_v2").glob("*.py"))
    paths += [ROOT/"src/sim_vla/latency_eval"/p for p in ("policy_adapter.py", "env_adapter.py", "contracts.py")]
    paths += [ROOT/"src/sim_vla"/p for p in ("assets.py", "runtime.py", "tasks.py", "protocol.py")]
    paths += [ROOT/"scripts/run_latency_v3.sh"]
    upstream = ROOT/".venv/lib/python3.12/site-packages/lerobot/policies"
    paths += [upstream/p for p in ("rtc/modeling_rtc.py", "rtc/configuration_rtc.py", "rtc/debug_tracker.py",
                                "smolvla/modeling_smolvla.py")]
    paths += [Path(entry["path"]) for entry in read(ROOT/"runs/latency_v2_audit/audit.json")["upstream_source"].values()]
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(set(paths))}


def progress(stage, **kwargs):
    next_stage = {"dev_screen": "dev_confirm", "dev_confirm": "dev_strong_raw", "dev_strong_raw": "dev_strong_shift",
        "dev_strong_shift": "dev_legacy", "dev_legacy": "dev_async_r1", "dev_async_r1": "physical_restore_and_freeze",
        "frozen": "formal_core", "formal_core": "formal_async", "formal_async": "finish", "complete": "complete"}
    payload = {"stage": stage, "next_stage": next_stage.get(stage),
        "resume_command": ("" if RUN_ID == "latency_v3_plan05" else f"SIM_VLA_LATENCY_V3_RUN_ID={RUN_ID} ")
                          + "bash scripts/run_latency_v3.sh pipeline",
        "remaining_budget": Budget(ROOT/"runs/latency_v2_budget.json").remaining(), **kwargs}
    publish(RUN/"progress.json", payload)
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS/"PROGRESS.md").write_text("# PLAN-05 进度\n\n```json\n"+json.dumps(payload, indent=2, ensure_ascii=False)+"\n```\n")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def budget():
    result = Budget(ROOT/"runs/latency_v2_budget.json")
    result.reserve(RUN_ID, min(360, result.remaining()+sum(
        r.get("run_id") == RUN_ID for r in read(result.path)["attempts"])))
    return result


def config(method, period, **kwargs):
    return {"method": method, "period": period, **kwargs}


def screen_trials():
    return [base_trial(task, 0, L, config(method, K)) for task, L, K, method in itertools.product(
        TASK_BY_KEY, (10, 20), (20, 25), METHODS)]


def rows(stage):
    directory = RUN/stage
    expected = read(directory/"expected_trials.json")
    values = []
    for trial in expected:
        p = directory/"trials"/trial["key"]/"result.json"
        if not p.exists():
            raise ValueError(f"incomplete stage {stage}: {len(values)}/{len(expected)}")
        result = read(p)
        if any(result[k] != v for k, v in trial.items() if k != "key"):
            raise ValueError("result differs from predeclared trial")
        values.append(result)
    return values


def score(values):
    return {"success_rate": float(np.mean([np.mean([r["success"] for r in values if r["task"] == task])
                                         for task in TASK_BY_KEY])),
            "generation_s_per_sim_s": float(np.mean([r["generation_seconds_per_sim_second"] for r in values])),
            "request_frequency": float(np.mean([r["requests_per_sim_second"] for r in values]))}


def ranking(item):
    return (-item["success_rate"], item["generation_s_per_sim_s"], item["request_frequency"], item["period"])


def select_short():
    output = {"short": {}, "scan": []}
    values = rows("dev_screen")
    for L in (10, 20):
        output["short"][str(L)] = {}
        for method in METHODS:
            options = []
            for K in (20, 25):
                subset = [r for r in values if r["latency"] == L and r["config"] == config(method, K)]
                item = {"latency": L, "method": method, "period": K, **score(subset)}
                options.append(item)
                output["scan"].append(item)
            output["short"][str(L)][method] = min(options, key=ranking)["period"]
    publish(RUN/"short_selection.json", output)
    return output


def confirmation_trials(short):
    return [base_trial(task, 1, L, config(method, short["short"][str(L)][method]))
            for task, L, method in itertools.product(TASK_BY_KEY, (10, 20), METHODS)]


def strong_trials(method):
    return [base_trial(task, init, L, config(method, 10 if method == "B_SHIFT" and L == 10 else 50))
            for task, init, L in itertools.product(TASK_BY_KEY, (0, 1), (10, 20))]


def legacy_trials():
    return [base_trial(task, 0, L, config("P_HARD", 50, legacy=True))
            for task, L in itertools.product(TASK_BY_KEY, (10, 20))]


def finalize_selection(short):
    output = {**short, "configs": {}, "main_candidate": "P_HARD", "main_baseline": "B_RAW",
        "rule": "short K chosen on init0, confirmed on init1; baseline compares confirmed short and historical strong"}
    all_values = rows("dev_screen")+rows("dev_confirm")+rows("dev_strong_raw")+rows("dev_strong_shift")
    for L in (10, 20):
        output["configs"][str(L)] = {}
        for method in METHODS:
            K = short["short"][str(L)][method]
            if method.startswith("B_"):
                options = []
                for period in (K, 10 if method == "B_SHIFT" and L == 10 else 50):
                    subset = [r for r in all_values if r["latency"] == L and r["config"] == config(method, period)]
                    options.append({"period": period, **score(subset)})
                K = min(options, key=ranking)["period"]
            output["configs"][str(L)][method] = config(method, K)
    publish(RUN/"selection.json", output)
    return output


def async_dev_trials(selection):
    return [base_trial(task, 0, 20, {**selection["configs"]["20"][method],
                                   **({"adaptive": True} if method in ("P_HARD", "RTC_GUIDED") else {})})
            for task, method in itertools.product(TASK_BY_KEY, METHODS)]


def physical_assets():
    manifest = read(BANK)
    states, poses = {}, {}
    for task, entry in manifest["tasks"].items():
        path = Path(entry["states_path"])
        if not path.is_absolute():
            path = BANK.parent/path
        if file_sha256(path) != entry["states_sha256"]:
            raise ValueError("physical states changed")
        pose_path = Path(entry["model_poses_path"])
        if not pose_path.is_absolute():
            pose_path = BANK.parent/pose_path
        if file_sha256(pose_path) != entry["model_poses_sha256"]:
            raise ValueError("physical model poses changed")
        states[task] = np.load(path, allow_pickle=False)
        poses[task] = ModelPoseBank.load(pose_path, expected_states=len(states[task]))
    return states, poses


def run_matrix(stage, trials, *, asynchronous=False, physical=False, dry_run=False):
    formal = stage in ("formal_core", "formal_async")
    if formal:
        frozen = read(RUN/"frozen/experiment_spec.json")
        ensure_frozen_unchanged(frozen)
        if (trials != frozen["async_trials" if asynchronous else "core_trials"] or not physical
                or asynchronous != (stage == "formal_async")):
            raise ValueError("formal matrix or backend differs from frozen execution")
        settings = frozen["execution_settings"]

    from ..latency_v2.env import ControlledEnv
    from ..latency_v2.worker import InferenceWorker
    from .policy import ResearchPolicyAdapter
    from .runner import run_episode

    directory = RUN/stage
    code = source_inventory()
    spec = {"stage": stage, "asynchronous": asynchronous, "physical": physical,
        "startup": "stationary_native_prefill", "observation_mode": "on_request",
        "checkpoint": CHECKPOINT, "model_sha256": MODEL_SHA, "semantic_code": code,
        "semantic_code_sha256": canonical_hash(code), "run_id": RUN_ID,
        "physical_bank_manifest": str(BANK.relative_to(ROOT)) if physical else None,
        "physical_bank_manifest_sha256": file_sha256(BANK) if physical else None}
    if dry_run:
        return {"stage": stage, "count": len(trials), "unique": len({trial_key(t) for t in trials})}
    if not formal:
        settings, _ = capture_execution_settings(RUN)
    spec["execution_settings"] = settings
    if file_sha256(ROOT/CHECKPOINT/"model.safetensors") != MODEL_SHA:
        raise ValueError("E1 weights changed")
    states, poses = physical_assets() if physical else ({}, {})
    with run_lock(directory):
        saved_spec = directory/"spec.json"
        if saved_spec.exists():
            previous = read(saved_spec)
            if previous["expected_trials"] != trials:
                raise ValueError("completed/resumed matrix differs from its declared trials")
            if formal and (previous.get("semantic_code_sha256") != spec["semantic_code_sha256"]
                           or previous.get("execution_settings") != settings):
                raise ValueError("formal stage uses different frozen code or execution settings; use a new run")
            expected = read(directory/"expected_trials.json")
            if all((directory/"trials"/t["key"]/"result.json").exists() for t in expected):
                values = rows(stage)
                if any(r["code_sha256"] != previous["semantic_code_sha256"] for r in values):
                    raise ValueError("completed results differ from their original code inventory")
                return {"stage": stage, "completed": len(expected), "skipped_completed": True,
                        "original_code_sha256": previous["semantic_code_sha256"]}
        prepare_run(directory, spec, trials, resume=saved_spec.exists())
        b = budget()
        expected = read(directory/"expected_trials.json")
        remaining = [t for t in expected if not (directory/"trials"/t["key"]/"result.json").exists()]
        if len(remaining) > b.remaining():
            raise RuntimeError("matrix exceeds available shared attempt budget")
        if not remaining:
            return {"stage": stage, "completed": len(expected)}
        engine = InferenceWorker(factory_name="sim_vla.latency_v3.policy:ResearchPolicyAdapter",
            factory_kwargs={"checkpoint": ROOT/CHECKPOINT, "warmup_calls": 3},
            startup_timeout=180.) if asynchronous else ResearchPolicyAdapter(
                checkpoint=ROOT/CHECKPOINT, warmup_calls=3)
        if not asynchronous:
            engine.warmup()
        envs = {}
        try:
            for trial in remaining:
                actual_trial = {k: v for k, v in trial.items() if k != "key"}
                target = directory/"trials"/trial["key"]
                b.prepare_retry(target)
                with b.track(trial["key"], stage, target, actual_trial, run_id=RUN_ID) as attempt:
                    task = trial["task"]
                    if task not in envs:
                        envs[task] = ControlledEnv(task_key=task, observation_mode="on_request")
                        envs[task].model_pose_bank = poses.get(task)
                    result = run_episode(envs[task], engine, actual_trial, target, asynchronous=asynchronous,
                        states=states.get(task), generation=attempt, version=spec["semantic_code_sha256"],
                        initial_p95=settings["initial_endpoint_p95_s"][trial["config"]["method"]],
                        estimator_window=settings["delay_estimator"]["window"],
                        tick_seconds=settings["control"]["tick_seconds"],
                        cycle_limit_seconds=settings["control"]["cycle_limit_seconds"],
                        required_cycle_fraction=settings["control"]["required_cycle_fraction"],
                        video=stage.startswith("formal") and trial["init_index"] == min(t["init_index"] for t in trials))
                done = sum((directory/"trials"/t["key"]/"result.json").exists() for t in expected)
                progress(stage, completed=done, total=len(expected), task=task, method=trial["config"]["method"],
                         latency=trial["latency"], success=result["success"], reason=result["termination_reason"])
                if result["error"]:
                    raise RuntimeError(f"method error retained; stop before changing semantics: {result['error']}")
        finally:
            for env in envs.values():
                env.close()
            if asynchronous:
                publish(directory/"worker_shutdown.json", engine.close())
        publish(directory/"complete.json", {"completed": len(rows(stage)), "code_sha256": spec["semantic_code_sha256"]})
    return {"stage": stage, "completed": len(trials)}


def freeze(selection):
    import time

    target = RUN/"frozen/experiment_spec.json"
    if target.exists():
        existing = read(target)
        ensure_frozen_unchanged(existing)
        return existing
    from .audit import verify_execution_assets

    execution_assets = verify_execution_assets()
    exposure = read(RUN/"audit/physical_eligibility.json")
    eligible = exposure["eligible_indices"]
    b = budget()
    negative = all(not r["success"] for r in rows("dev_confirm") if r["config"]["method"] in ("P_HARD", "RTC_GUIDED"))
    options = ((4, 4),) if negative else ((7, 5), (6, 5), (5, 5), (4, 4))
    sizes = next(((n, a) for n, a in options if 24*n+9*a+12 <= b.remaining()
                  and all(len(indices) >= n for indices in eligible.values())), None)
    if sizes is None:
        raise RuntimeError("no complete fair formal matrix fits the fresh states and remaining budget")
    n, a = sizes
    core = [base_trial(task, init, L, selection["configs"][str(L)][method])
            for init_rank, L, method, task in itertools.product(range(n), (10, 20), METHODS, TASK_BY_KEY)
            for init in [eligible[task][init_rank]]]
    async_trials = [base_trial(task, init, 20, {**selection["configs"]["20"][method],
                **({"adaptive": True} if method in ("P_HARD", "RTC_GUIDED") else {})})
            for init_rank, method, task in itertools.product(range(a), ("B_RAW", "P_HARD", "RTC_GUIDED"), TASK_BY_KEY)
            for init in [eligible[task][init_rank]]]
    settings, probe_content = capture_execution_settings(RUN)
    code = source_inventory()
    spec = {"format": "latency-v3-plan05-formal-v2", "run_id": RUN_ID,
        "frozen_unix": time.time(), "frozen_monotonic_ns": time.monotonic_ns(),
        "code_inventory": code, "code_sha256": canonical_hash(code), "model_sha256": MODEL_SHA,
        "execution_assets": execution_assets,
        "execution_settings": settings,
        "checkpoint": CHECKPOINT, "bank_manifest_sha256": file_sha256(BANK),
        "physical_eligibility_sha256": file_sha256(RUN/"audit/physical_eligibility.json"),
        "selection": selection, "core_n": n, "async_n": a, "negative_development_stop": negative,
        "core_trials": core, "async_trials": async_trials, "main_candidate": "P_HARD", "main_baseline": "B_RAW",
        "bootstrap_draws": 10000, "bootstrap_seed": 20261003, "max_ticks": 300,
        "startup": "stationary_native_prefill", "remaining_budget_at_freeze": b.remaining()}
    spec["spec_sha256"] = canonical_hash(spec)
    save_probe_snapshot(RUN, probe_content)
    for relative in code:
        destination = RUN/"frozen/code_snapshot"/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, destination)
    publish(CONFIG, spec)
    publish(target, spec)
    progress("frozen", core=len(core), asynchronous=len(async_trials), spec_sha256=spec["spec_sha256"])
    return spec


def ensure_frozen_unchanged(spec):
    if canonical_hash({k: v for k, v in spec.items() if k != "spec_sha256"}) != spec["spec_sha256"]:
        raise ValueError("frozen manifest content no longer matches its digest")
    verify_execution_settings(spec, RUN)
    if spec.get("run_id") != RUN_ID:
        raise ValueError("frozen experiment belongs to a different run")
    if source_inventory() != spec["code_inventory"] or file_sha256(BANK) != spec["bank_manifest_sha256"]:
        raise ValueError("frozen execution semantics or physical bank changed; do not mix runs")
    if file_sha256(RUN/"audit/physical_eligibility.json") != spec["physical_eligibility_sha256"]:
        raise ValueError("frozen physical-state eligibility changed")
    from .audit import verify_execution_assets

    if verify_execution_assets() != spec["execution_assets"]:
        raise ValueError("frozen checkpoint, task assets, upstream sources or packages changed")
    if read(CONFIG) != spec:
        raise ValueError("reviewable configuration differs from frozen execution spec")
