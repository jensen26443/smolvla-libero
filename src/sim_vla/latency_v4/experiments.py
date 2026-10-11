"""Predeclared grids, honest stopping gates and immutable stage identities."""
import itertools
import time
from collections import Counter

import numpy as np

from ..latency_v2.integrity import file_sha256
from ..latency_v2.locking import run_lock
from ..latency_v2.storage import canonical_hash, prepare_run, publish, trial_key
from ..tasks import TASK_BY_KEY
from .layout import (
    BANK,
    CHECKPOINT,
    ROOT,
    RUN,
    RUN_ID,
    budget,
    immutable_bytes,
    paired_seed,
    physical_assets,
    progress,
    read,
    source_inventory,
)
from .schema import METHODS, SAMPLING_STEPS, MethodConfig

FORMAL_STAGES = ("formal_async", "formal_delay", "formal_ablation")


def config(method, steps, period, *, probe=None):
    probe = probe or read(RUN/"profile/production_probe.json")
    path = "rtc" if method in ("RTC_STRICT", "RTC_CATCHUP", "HARD") else "native"
    initial = probe["initial_endpoint_p95_s"][str(steps)][path]
    return MethodConfig(method, steps, period, initial,
                        late_policy="catchup" if method == "RTC_CATCHUP" else "strict").resolved()


def trial(task, init, cfg, *, latency=0, physical=False, manifest=None):
    if physical:
        manifest = manifest or read(BANK)
        state_id = manifest["tasks"][task]["states"][init]["scene_sha256"]
    else:
        state_id = f"historically_exposed_native_init{init}"
    return {"task": task, "init_index": init, "state_id": state_id, "seed": paired_seed(task, state_id),
            "latency": latency, "profile": "fixed" if latency else "actual_async", "config": cfg}


def sampling_trials():
    return [trial(task, init, config(method, steps, 50 if method == "RAW" else 25))
            for init, task, steps, method in itertools.product((0, 1), TASK_BY_KEY, SAMPLING_STEPS,
                                                             ("RAW", "RTC_STRICT"))]


def period_trials(selection):
    return [trial(task, init, config(method, selection["sampling_steps"], period))
            for init, task, period in itertools.product((0, 1), TASK_BY_KEY, (10, 25, 50))
            for method in METHODS[init % 4:]+METHODS[:init % 4]]


def sustain_trials(selection):
    return [trial(task, init, config(method, selection["sampling_steps"], selection["periods"]["RTC_STRICT"]))
            for init, task, method in itertools.product((0, 1), TASK_BY_KEY, ("RTC_STRICT", "RTC_CATCHUP", "HARD"))]


def formal_trials(selection, *, probe=None, manifest=None):
    probe = probe or read(RUN/"profile/production_probe.json")
    manifest = manifest or read(BANK)
    steps, periods = selection["sampling_steps"], selection["periods"]
    matrices = {name: [] for name in FORMAL_STAGES}
    for index, task in itertools.product(range(20), TASK_BY_KEY):
        offset = index % 4
        for method in METHODS[offset:]+METHODS[:offset]:
            matrices["formal_async"].append(trial(task, index, config(method, steps, periods[method], probe=probe),
                                                  physical=True, manifest=manifest))
    for latency, index, task in itertools.product((10, 20), range(3), TASK_BY_KEY):
        offset = (index+(latency == 20)) % 4
        for method in METHODS[offset:]+METHODS[:offset]:
            matrices["formal_delay"].append(trial(task, index, config(method, steps, periods[method], probe=probe),
                                                 latency=latency, physical=True, manifest=manifest))
    ablation_configs = tuple(dict.fromkeys((10, steps)))
    for index, task in itertools.product(range(5), TASK_BY_KEY):
        options = [(s, method) for s in ablation_configs for method in ("RTC_STRICT", "RTC_CATCHUP")]
        offset = index % len(options)
        for s, method in options[offset:]+options[:offset]:
            matrices["formal_ablation"].append(trial(task, index,
                config(method, s, periods["RTC_STRICT"], probe=probe), physical=True, manifest=manifest))
    return matrices


def rows(stage, *, require_complete=True):
    directory = RUN/stage
    expected_path = directory/"expected_trials.json"
    if not expected_path.exists():
        if require_complete:
            raise ValueError(f"missing stage {stage}")
        return []
    expected = read(expected_path)
    values = []
    for item in expected:
        path = directory/"trials"/item["key"]/"result.json"
        if not path.exists():
            if require_complete:
                raise ValueError(f"incomplete {stage}: {len(values)}/{len(expected)}")
            continue
        value = read(path)
        if any(value.get(key) != content for key, content in item.items() if key != "key"):
            raise ValueError("completed result differs from its declared trial")
        MethodConfig.from_resolved(value["effective_config"])
        if value["effective_config"] != item["config"]:
            raise ValueError("completed effective parameters differ from the declared configuration")
        values.append(value)
    return values


def sampling_selection(values, probe):
    checks = []
    for steps in SAMPLING_STEPS:
        entry = {"sampling_steps": steps, "methods": {}, "eligible": True, "rejections": []}
        for method in ("RAW", "RTC_STRICT"):
            baseline = [row for row in values if row["config"]["method"] == method
                        and row["config"]["sampling_steps"] == 10]
            subset = [row for row in values if row["config"]["method"] == method
                      and row["config"]["sampling_steps"] == steps]
            if len(baseline) != 6 or len(subset) != 6:
                raise ValueError("sampling selection requires the complete predeclared 36 episodes")
            successes = sum(row["success"] for row in subset)
            baseline_successes = sum(row["success"] for row in baseline)
            entry["methods"][method] = {"successes": successes, "n": 6, "ten_step_successes": baseline_successes}
            if successes < baseline_successes-1:
                entry["rejections"].append(method+"_more_than_one_success_lost")
        rtc = [row for row in values if row["config"]["method"] == "RTC_STRICT"
               and row["config"]["sampling_steps"] == steps]
        entry["rtc_endpoint_p95_s"] = probe["initial_endpoint_p95_s"][str(steps)]["rtc"]
        if entry["rtc_endpoint_p95_s"] > 1.:
            entry["rejections"].append("rtc_endpoint_p95_exceeds_one_second")
        activated_tasks = sorted({row["task"] for row in rtc if row["prefix_requests"] and row["prefix_activations"]})
        entry["tasks_with_prefix_activation"] = activated_tasks
        if len(activated_tasks) < 2:
            entry["rejections"].append("prefix_activation_in_fewer_than_two_tasks")
        if any(row.get("error") or not row["control_step_invariant"] for row in values
               if row["config"]["sampling_steps"] == steps):
            entry["rejections"].append("method_or_control_error")
        entry["eligible"] = not entry["rejections"]
        checks.append(entry)
    eligible = [entry for entry in checks if entry["eligible"]]
    winner = None
    if eligible:
        best = min(eligible, key=lambda entry: entry["rtc_endpoint_p95_s"])
        # Within 5% of the fastest endpoint, retain the greater step count.
        tied = [entry for entry in eligible if entry["rtc_endpoint_p95_s"] < best["rtc_endpoint_p95_s"]*1.05]
        winner = max(tied, key=lambda entry: entry["sampling_steps"])["sampling_steps"]
    return {"passed": winner is not None, "sampling_steps": winner, "candidates": checks,
        "rule": "complete paired development counts; latency lowest, <5% tie prefers more steps",
        "claim": "configuration screening on exposed init0/init1, not a no-loss or generalization proof"}


def score(values):
    return {"task_equal_success_rate": float(np.mean([
        np.mean([row["success"] for row in values if row["task"] == task]) for task in TASK_BY_KEY])),
        "coverage_failures": sum(row["termination_reason"] == "coverage_infeasible" for row in values),
        "generation_s_per_sim_s": float(np.mean([row["generation_seconds_per_sim_second"] for row in values])),
        "request_frequency": float(np.mean([row["requests_per_sim_second"] for row in values]))}


def select_periods(values, sampling):
    if len(values) != 72:
        raise ValueError("period selection requires all 72 predeclared episodes")
    scans, periods = [], {}
    for method in METHODS:
        options = []
        for period in (10, 25, 50):
            subset = [row for row in values if row["config"]["method"] == method and row["config"]["period"] == period]
            entry = {"method": method, "period": period, **score(subset)}
            scans.append(entry)
            options.append(entry)
        ranking = lambda entry: (-entry["task_equal_success_rate"], entry["coverage_failures"],
            entry["generation_s_per_sim_s"], entry["request_frequency"], -entry["period"])
        periods[method] = min(options, key=ranking)["period"]
    periods["RTC_CATCHUP"] = periods["RTC_STRICT"]
    return {**sampling, "periods": periods, "period_scores": scans,
        "rtc_period_selection": "RTC_STRICT scores only; same period for both RTC rules"}


def sustained_gate(values):
    if len(values) != 18:
        raise ValueError("sustained gate requires all 18 diagnostic episodes")
    checks = {}
    for method in ("RTC_STRICT", "RTC_CATCHUP", "HARD"):
        subset = [row for row in values if row["config"]["method"] == method]
        completed = sum(row["sustained_300_ticks"] for row in subset)
        tasks = {task: any(row["prefix_activations"] >= 3 for row in subset if row["task"] == task)
                 for task in TASK_BY_KEY}
        checks[method] = {"full_300_ticks": completed, "n": 6, "tasks_with_three_activations": tasks,
            "invariant_errors": sum(not row["control_step_invariant"] or bool(row["error"]) for row in subset),
            "termination_reasons": dict(Counter(row["termination_reason"] for row in subset)),
            "passed": completed >= 4 and all(tasks.values())
                      and all(row["control_step_invariant"] and not row["error"] for row in subset)}
    return {"passed": checks["RTC_STRICT"]["passed"] and checks["RTC_CATCHUP"]["passed"],
            "methods": checks, "success_rates_reported": False, "hard_is_diagnostic_only": True}


def run_matrix(stage, trials, *, physical=False, asynchronous=True, sustain=False, dry_run=False):
    formal = stage in FORMAL_STAGES
    if formal:
        from .freeze import verify_frozen

        frozen = verify_frozen()
        if trials != frozen["matrices"][stage] or not physical or asynchronous != (stage != "formal_delay"):
            raise ValueError("formal matrix or backend differs from the immutable freeze")
    keys = [trial_key(item) for item in trials]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate trial identities")
    for item in trials:
        MethodConfig.from_resolved(item["config"])
    if dry_run:
        return {"stage": stage, "count": len(trials), "unique": len(set(keys))}
    code = source_inventory()
    probe_sha = file_sha256(RUN/"profile/production_probe.json")
    spec = {"stage": stage, "asynchronous": asynchronous, "physical": physical, "sustain": sustain,
        "checkpoint": CHECKPOINT, "code_inventory": code, "code_sha256": canonical_hash(code),
        "run_id": RUN_ID, "production_probe_sha256": probe_sha,
        "bank_sha256": file_sha256(BANK) if physical else None}
    directory = RUN/stage
    with run_lock(directory):
        saved = directory/"spec.json"
        if saved.exists():
            previous = read(saved)
            if previous["expected_trials"] != trials:
                raise ValueError("stage matrix changed")
            if previous["code_sha256"] != spec["code_sha256"]:
                # Complete development measurements may survive a documented
                # accounting/bank/recovery fix. Runtime primitives and the
                # probe/config/backend must still match; partial stages never mix.
                allowed = {"src/sim_vla/latency_v4/"+name for name in (
                    "layout.py", "bank.py", "freeze.py", "experiments.py", "delivery.py", "videos.py", "materials.py",
                    "continuation.py", "physical.py")}
                changed = {path for path in previous["code_inventory"].keys() | code.keys()
                           if previous["code_inventory"].get(path) != code.get(path)}
                completed = read(directory/"expected_trials.json")
                metadata_equal = all(previous.get(key) == value for key, value in spec.items()
                                     if key not in {"code_inventory", "code_sha256"})
                if formal or not changed <= allowed or not metadata_equal or not all(
                        (directory/"trials"/entry["key"]/"result.json").exists() for entry in completed):
                    raise ValueError("stage source changed; preserve the run and use an isolated version")
                for relative, digest in previous["code_inventory"].items():
                    snapshot = directory/"code_snapshot"/relative
                    if not snapshot.exists() or file_sha256(snapshot) != digest:
                        raise ValueError("original completed development source snapshot changed or missing")
                values = rows(stage)
                if any(row["code_sha256"] != previous["code_sha256"] for row in values):
                    raise ValueError("completed result differs from its original development source")
                return {"stage": stage, "completed": len(values), "skipped_completed": True,
                        "original_code_sha256": previous["code_sha256"], "support_source_changes": sorted(changed)}
        prepare_run(directory, spec, trials, resume=saved.exists())
        expected = read(directory/"expected_trials.json")
        remaining = [item for item in expected if not (directory/"trials"/item["key"]/"result.json").exists()]
        if not remaining:
            values = rows(stage)
            if any(row["code_sha256"] != spec["code_sha256"] for row in values):
                raise ValueError("completed result source differs from the stage")
            return {"stage": stage, "completed": len(values), "skipped_completed": True}
        b = budget()
        if len(remaining) > b.remaining():
            raise RuntimeError("complete declared matrix exceeds the additional budget")
        for relative in code:
            immutable_bytes(directory/"code_snapshot"/relative, (ROOT/relative).read_bytes())
        from ..latency_v2.env import ControlledEnv
        from ..latency_v2.worker import InferenceWorker
        from .physical import VerifiedPhysicalEnv
        from .policy import SamplingPolicyAdapter
        from .runner import run_episode

        states, poses = physical_assets() if physical else ({}, {})
        manifest = read(BANK) if physical else None
        step_counts = {item["config"]["sampling_steps"] for item in trials}
        bound_steps = next(iter(step_counts)) if len(step_counts) == 1 else None
        job_started_unix, job_started_ns = time.time(), time.monotonic_ns()
        engine = InferenceWorker(factory_name="sim_vla.latency_v4.policy:SamplingPolicyAdapter",
            factory_kwargs={"checkpoint": ROOT/CHECKPOINT, "sampling_steps": bound_steps, "warmup_calls": 3},
            startup_timeout=180.) if asynchronous else SamplingPolicyAdapter(
                checkpoint=ROOT/CHECKPOINT, sampling_steps=bound_steps, warmup_calls=3)
        if not asynchronous:
            engine.warmup()
        engine_startup_s = (time.monotonic_ns()-job_started_ns)/1e9
        envs = {}
        try:
            for item in remaining:
                current = {key: value for key, value in item.items() if key != "key"}
                target = directory/"trials"/item["key"]
                b.prepare_retry(target)
                task = item["task"]
                with b.track(item["key"], stage, target, current, run_id=RUN_ID) as attempt:
                    if task not in envs:
                        envs[task] = (VerifiedPhysicalEnv(task_key=task, observation_mode="on_request",
                            expected_scenes=[entry["after_settle_scene_sha256"] for entry in manifest["tasks"][task]["states"]])
                            if physical else ControlledEnv(task_key=task, observation_mode="on_request"))
                        envs[task].model_pose_bank = poses.get(task)
                    result = run_episode(envs[task], engine, current, target, asynchronous=asynchronous,
                        states=states.get(task), generation=attempt, version=spec["code_sha256"], sustain=sustain,
                        video=(stage == "formal_async" and item["init_index"] == 0) or
                              (stage == "develop_period" and item["init_index"] == 0 and item["config"]["period"] == 25) or
                              (stage == "sustain" and item["init_index"] == 0))
                done = sum((directory/"trials"/entry["key"]/"result.json").exists() for entry in expected)
                progress(stage, completed=done, total=len(expected), task=task, method=item["config"]["method"],
                    sampling_steps=item["config"]["sampling_steps"], period=item["config"]["period"],
                    success=result["success"], steps=result["steps"], reason=result["termination_reason"])
                if result["error"]:
                    raise RuntimeError("method error retained; stop before changing control semantics: "+result["error"])
        finally:
            for env in envs.values():
                env.close()
            if asynchronous:
                publish(directory/"worker_shutdown.json", engine.close())
            publish(directory/f"job_accounting_{job_started_ns}.json", {
                "started_unix": job_started_unix, "finished_unix": time.time(),
                "started_monotonic_ns": job_started_ns, "finished_monotonic_ns": time.monotonic_ns(),
                "total_job_wall_s": (time.monotonic_ns()-job_started_ns)/1e9,
                "engine_loading_and_warmup_wall_s": engine_startup_s,
                "scope": "includes loading/warmup, reset, startup/control/tail, serialization and shutdown"})
        publish(directory/"complete.json", {"completed": len(rows(stage)), "code_sha256": spec["code_sha256"]})
    return {"stage": stage, "completed": len(trials)}


def develop_sampling():
    run_matrix("develop_sampling", sampling_trials())
    result = sampling_selection(rows("develop_sampling"), read(RUN/"profile/production_probe.json"))
    publish(RUN/"sampling_selection.json", result)
    return result


def develop_period():
    selection = read(RUN/"sampling_selection.json")
    if not selection["passed"]:
        raise RuntimeError("sampling gate failed; no period search is authorized by the plan")
    run_matrix("develop_period", period_trials(selection))
    result = select_periods(rows("develop_period"), selection)
    publish(RUN/"selection.json", result)
    return result


def sustain():
    selection = read(RUN/"selection.json")
    run_matrix("sustain", sustain_trials(selection), sustain=True)
    result = sustained_gate(rows("sustain"))
    publish(RUN/"development_gate.json", result)
    return result


def pipeline():
    from .audit import audit, profile
    from .delivery import report, verify

    frozen_path = RUN/"frozen/experiment_spec.json"
    if frozen_path.exists():
        from .freeze import verify_frozen

        verify_frozen()
    audit()
    profile()
    from .continuation import AUTHORIZATION, formal_eligibility, verify_continuation

    continuing = (RUN/AUTHORIZATION).exists()
    if continuing:
        verify_continuation(run=RUN, run_id=RUN_ID)
        selection = read(RUN/"sampling_selection.json")
    else:
        selection = develop_sampling()
    if not selection["passed"]:
        progress("stopped_sampling_gate", reason="no_eligible_sampling_configuration")
        result = report()
        verify()
        return result
    if continuing:
        gate = read(RUN/"development_gate.json")
        formal_eligibility(gate, run=RUN, run_id=RUN_ID)
    else:
        develop_period()
        gate = sustain()
    if not gate["passed"] and not continuing:
        progress("stopped_sustained_gate", reason="declared_sustained_execution_gate_not_met")
        result = report()
        verify()
        return result
    from .bank import prepare_bank
    from .freeze import freeze

    prepare_bank()
    spec = freeze()
    for stage in FORMAL_STAGES:
        run_matrix(stage, spec["matrices"][stage], physical=True, asynchronous=stage != "formal_delay")
    result = report()
    verify()
    progress("complete", algorithm_result=result["algorithm_result"])
    return result


def dispatch(command):
    if command == "dry-run":
        result = {"commands": 15, "development": {"sampling": 36, "period": 72, "sustain": 18},
            "formal": {"async": 240, "delay": 72, "ablation": "60, or 30 if selected steps=10"},
            "bank": 60, "restore": 6, "repair_reserve": 36, "additional_cap": 600, "environment_attempts": 0}
        if (RUN/"frozen/experiment_spec.json").exists():
            from .freeze import verify_frozen

            spec = verify_frozen()
            result["frozen_counts"] = {stage: len(matrix) for stage, matrix in spec["matrices"].items()}
        return result
    if command == "pipeline":
        return pipeline()
    if command == "develop-sampling":
        return develop_sampling()
    if command == "develop-period":
        return develop_period()
    if command == "sustain":
        return sustain()
    if command == "prepare-bank":
        from .bank import prepare_bank

        return prepare_bank()
    if command == "freeze":
        from .freeze import freeze

        return freeze()
    if command.startswith("formal-"):
        from .freeze import verify_frozen

        spec = verify_frozen()
        stage = command.replace("-", "_")
        return run_matrix(stage, spec["matrices"][stage], physical=True, asynchronous=stage != "formal_delay")
    if command == "recover":
        recovered = budget().recover(run_id=RUN_ID)
        publish(RUN/"recovery.json", recovered)
        return {"reconciled_attempts": len(recovered), "attempts_refunded": 0}
    from .delivery import report, verify

    return report() if command == "report" else verify()
