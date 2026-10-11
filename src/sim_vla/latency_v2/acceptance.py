"""Budgeted acceptance repair, separate from algorithm selection."""
from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np

from ..assets import project_root
from ..tasks import TASK_BY_KEY
from .env import ControlledEnv
from .experiments import CHECKPOINT, base_trial, completed_results, semantic_inventory
from .initializations import check_formal_manifest
from .integrity import file_sha256, inventory
from .layout import completion_flags
from .schema import CommittedPrefix, InferenceRequest, ObservationSnapshot, digest_arrays, frozen_copy
from .storage import Budget, canonical_hash, publish, trial_key
from .timing import EventLog, provenance_complete, request_record, result_record
from .worker import InferenceWorker

RUN_ID = "latency_v2_acceptance_r1"


def percentiles(values):
    return np.percentile(values, [50, 95, 99]).tolist() if values else []


def profile_paths(directory: Path, *, observation_mode="per_tick", max_ticks=120) -> dict:
    root = project_root()
    if (directory/"complete.json").exists():
        return json.loads((directory/"complete.json").read_text())
    budget = Budget(root/"runs/latency_v2_budget.json")
    source = root/"runs/latency_v2_dev_baselines"
    trials = json.loads((source/"expected_trials.json").read_text())
    worker = InferenceWorker(checkpoint=root/CHECKPOINT)
    results = []
    try:
        for task in TASK_BY_KEY:
            template = next(row for row in trials if row["task"] == task and row["init_index"] == 0
                            and row["latency"] == 20 and row["config"] == {"method": "B_RAW", "period": 40})
            path = source/"trials"/template["key"]/"arrays.npz"
            actions = np.load(path, allow_pickle=False)["actions"][:max_ticks]
            for condition in ("idle", "inference"):
                trial = {"task": task, "condition": condition, "observation_mode": observation_mode,
                         "actions_sha256": digest_arrays(actions), "seed": template["seed"], "init_index": 0}
                target = directory/"trials"/trial_key(trial)
                if (target/"result.json").exists():
                    results.append(json.loads((target/"result.json").read_text()))
                    continue
                budget.prepare_retry(target)
                with budget.track(trial_key(trial), "acceptance_profile", target, trial, run_id=RUN_ID) as attempt:
                    kwargs = {"task_key": task, "profile_env": True}
                    if observation_mode != "per_tick":
                        kwargs["observation_mode"] = observation_mode
                    env = ControlledEnv(**kwargs)
                    events = EventLog(attempt_id=f"attempt-{attempt:06d}", generation=attempt, directory=target)
                    states, controls, sample_rows, inference_rows = [], [], [], []
                    request_rows = []
                    ordinal = 0

                    def record_completed(completed, tick, request_rows=request_rows, events=events):
                        request_rows[completed.request.request_id].update(result_record(completed))
                        events.append({"event": "worker_complete", "tick": None,
                            "request_id": completed.request.request_id,
                            "monotonic_ns": completed.worker_completed_monotonic_ns,
                            "request_tick": completed.request.request_tick, "error": completed.error})
                        events.append({"event": "control_receive", "tick": tick,
                            "request_id": completed.request.request_id,
                            "monotonic_ns": completed.received_monotonic_ns, "error": completed.error})
                        if completed.error:
                            raise RuntimeError(completed.error)

                    try:
                        observation = env.start_episode(0, template["seed"])
                        for tick, action in enumerate(actions):
                            completed = worker.poll()
                            if completed is not None:
                                record_completed(completed, tick)
                                inference_rows.append({"request_id": completed.request.request_id,
                                    "forward_s": completed.stages.get("forward_s"),
                                    "error": completed.error, "ready_wall": completed.ready_wall})
                            sample_started = time.monotonic()
                            sampling_timing = {}
                            if condition == "inference" and not worker.busy:
                                if hasattr(env, "sample_observation"):
                                    observation = env.sample_observation(tick)
                                    sampling_timing = env.last_sample_timing
                                snapshot = ObservationSnapshot.capture(target.name, tick, observation, env.instruction)
                                prefix = CommittedPrefix(tick, frozen_copy(np.zeros((0, 7))), (), digest_arrays(np.zeros((0, 7))))
                                request = InferenceRequest(attempt, ordinal, snapshot, tick, prefix,
                                                           template["seed"], time.monotonic(), f"attempt-{attempt:06d}")
                                worker.submit(request)
                                request_rows.append({**request_record(request), "activation_status": "fixed_action_replay"})
                                events.append({"event": "submit", "tick": tick, "request_id": ordinal,
                                               "monotonic_ns": request.submit_monotonic_ns})
                                ordinal += 1
                            busy = worker.busy
                            sample_s = time.monotonic()-sample_started
                            observation, reward, terminated, truncated, info = env.step(action)
                            domain = env._env._env.env
                            states.append(np.asarray(env._env._env.get_sim_state()).copy())
                            controls.append(domain.sim.data.ctrl.copy())
                            sample_rows.append({"tick": tick, "worker_busy": busy, "timing": env.last_step_timing,
                                "sample_path_s": sample_s, "sample_timing": sampling_timing,
                                "reward": reward, "terminated": terminated, "truncated": truncated,
                                "success": bool(info["is_success"]), "sampling": observation.get("sampling", {})})
                            events.append({"event": "step", "tick": tick, "request_id": None,
                                           "success": bool(info["is_success"]), "worker_busy": busy})
                            if terminated or truncated:
                                break
                        if worker.busy:
                            completed = worker.drain()
                            record_completed(completed, None)
                        events.close()
                        values = [row["timing"]["total_s"] for row in sample_rows]
                        stages = {key: percentiles([row["timing"]["exclusive_s"].get(key, 0) for row in sample_rows])
                                  for key in {key for row in sample_rows for key in row["timing"]["exclusive_s"]}}
                        result = {**trial, "attempt_id": f"attempt-{attempt:06d}", "generation": attempt,
                                  "steps": len(states), "n_requests": ordinal,
                                  "step_p50_p95_p99_s": percentiles(values), "stages_p50_p95_p99_s": stages,
                                  "sample_rows": sample_rows, "inference_rows": inference_rows,
                                  "worker_busy_fraction": float(np.mean([row["worker_busy"] for row in sample_rows])),
                                  "trajectory_sha256": digest_arrays(np.asarray(states)),
                                  "controls_sha256": digest_arrays(np.asarray(controls)), "error": None,
                                  "timing_provenance_complete": provenance_complete(events, request_rows, log_error=events.error)}
                        publish(target/"requests.json", request_rows)
                        (target/"events.jsonl").write_text("".join(json.dumps(row)+"\n" for row in events))
                        np.savez_compressed(target/"trajectory.npz", states=states, controls=controls, actions=actions[:len(states)])
                        publish(target/"result.json", result)
                        results.append(result)
                        print(json.dumps({"task": task, "condition": condition, "mode": observation_mode,
                              "steps": len(states), "p50_ms": values and 1000*float(np.median(values)),
                              "budget_remaining": budget.remaining()}), flush=True)
                    finally:
                        events.close()
                        env.close()
    finally:
        publish(directory/"worker_shutdown.json", worker.close())
    summary = {"run_id": RUN_ID, "observation_mode": observation_mode, "results": results,
               "remaining_budget": budget.remaining()}
    publish(directory/"complete.json", summary)
    return summary


def audit_unused_initializations(directory: Path) -> dict:
    root = project_root()
    manifest_path = root/"runs/latency_v2_physical_bank/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audited = json.loads((root/"runs/latency_v2_audit/audit.json").read_text())
    checked = check_formal_manifest(manifest_path, audited["initializations"])
    used = {task: set() for task in TASK_BY_KEY}
    scene_hashes, scanned = set(), []
    for path in sorted((root/"runs").rglob("spec.json")):
        data = json.loads(path.read_text())
        config = data.get("config", {})
        source = config.get("initialization_manifest")
        if source and (root/source).resolve() == manifest_path.resolve():
            scanned.append(str(path.relative_to(root)))
            for trial in data.get("expected_trials", []):
                target = path.parent/"trials"/trial_key(trial)
                if (target/"attempt.json").exists() or (target/"result.json").exists():
                    used[trial["task"]].add(trial["init_index"])
    for path in (root/"runs").rglob("result.json"):
        if "trials" not in path.parts:
            continue
        data = json.loads(path.read_text())
        if data.get("initial_state_after_settle_sha256"):
            scene_hashes.add(data["initial_state_after_settle_sha256"])
    entries = {}
    for task, bank in manifest["tasks"].items():
        validation = manifest_path.parent/bank["reset_validation_path"]
        if file_sha256(validation) != bank["reset_validation_sha256"] or not bank["compatibility_gate_passed"]:
            raise ValueError("physical initialization reset certification changed")
        rows = json.loads(validation.read_text())
        available = [row["index"] for row in rows if row["index"] not in used[task]
                     and row["reset_after_settle_sha256"] not in scene_hashes]
        selected = sorted(available)[:5]
        if len(selected) != 5:
            raise ValueError("not enough policy-unused certified states for a paired five-init matrix")
        entries[task] = {"used_indices": sorted(used[task]), "selected_indices": selected,
                        "validation_sha256": file_sha256(validation),
                        "selected_rows": [row for row in rows if row["index"] in selected]}
    result = {"physical_manifest": str(manifest_path.relative_to(root)),
              "physical_manifest_sha256": file_sha256(manifest_path), "checked_assets": checked,
              "scanned_policy_run_specs": scanned, "tasks": entries,
              "training_overlap": "unknown", "passed": True}
    publish(directory/"policy_usage_audit.json", result)
    return result


def development_gate(directory: Path) -> dict:
    root = project_root()
    physics = json.loads((directory/"physics_equivalence.json").read_text())
    asynchronous = completed_results(directory/"async_smoke")
    simulated = completed_results(directory/"sim_smoke")
    async_flags = completion_flags(directory/"async_smoke", asynchronous, 9)
    sim_flags = completion_flags(directory/"sim_smoke", simulated, 3)
    passed = (physics["passed"] and all(async_flags.values()) and sim_flags["delivery_complete"]
              and sim_flags["timing_provenance_complete"] and not any(row["error"] for row in asynchronous+simulated))
    checks_path = directory/"code_checks.json"
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else {"passed": False}
    passed = passed and checks["passed"]
    evidence = {"passed": passed, "physical_equivalence": physics["passed"], "async": async_flags,
        "simulated": sim_flags, "code_checks": checks,
        "per_episode": [{"task": row["task"], "method": row["config"]["method"],
            "fraction_cycles_under_55ms": row["fraction_cycles_under_55ms"],
            "steps": row["steps"], "timing_provenance_complete": row["timing_provenance_complete"]}
            for row in asynchronous], "semantic_code_at_gate": semantic_inventory(root, asynchronous=True)}
    publish(directory/"gate.json", evidence)
    return evidence


def freeze_acceptance() -> dict:
    root = project_root()
    base = root/"runs"/RUN_ID
    directory = base/"frozen"
    if (directory/"experiment_spec.json").exists():
        return json.loads((directory/"experiment_spec.json").read_text())
    gate = development_gate(base/"dev")
    if not gate["passed"]:
        raise ValueError("development acceptance gate failed; formal budget is preserved")
    usage = audit_unused_initializations(base/"dev")
    previous_path = root/"runs/latency_v2_frozen/experiment_spec.json"
    previous = json.loads(previous_path.read_text())
    matrices = {"core": [], "jitter": [], "async": []}
    for name, rows in previous["formal_trial_matrices"].items():
        for row in rows:
            selected = usage["tasks"][row["task"]]["selected_indices"]
            matrices[name].append(base_trial(row["task"], selected[row["init_index"]], row["latency"],
                                            row["config"], profile=row["profile"]))
    if {name: len(rows) for name, rows in matrices.items()} != {"core": 240, "jitter": 60, "async": 45}:
        raise ValueError("acceptance comparison must preserve the complete 345-trial method matrix")
    budget = Budget(root/"runs/latency_v2_budget.json")
    if budget.remaining() < 345:
        raise RuntimeError("insufficient global attempt budget")
    budget.reserve(RUN_ID, 401)
    spec = copy.deepcopy(previous)
    spec.pop("spec_sha256")
    spec.update(format="latency-v2-formal-spec-v2", run_id=RUN_ID,
        previous_spec_sha256=previous["spec_sha256"], algorithm_parameters_reselected=False,
        semantic_code=semantic_inventory(root, asynchronous=True),
        physical_states=usage["checked_assets"], selected_init_indices={task: row["selected_indices"]
            for task, row in usage["tasks"].items()}, materials_dir=str((base/"materials").relative_to(root)),
        documentation_dir="docs/latency_v2/acceptance_r1",
        budget_before_formal=budget.remaining(), development_acceptance_gate=gate,
        policy_usage_audit={"path": str((base/"dev/policy_usage_audit.json").relative_to(root)),
                            "sha256": file_sha256(base/"dev/policy_usage_audit.json")},
        formal_trial_matrices=matrices, frozen_unix=time.time(),
        acceptance_flags=["delivery_complete", "realtime_20hz_passed", "timing_provenance_complete"])
    spec["timing"].update(observation_mode="on_request", camera_resolution=[256, 256],
        request_sampling="two fresh cameras and state at fixed current physics boundary, charged to control",
        cycle_measurement="adjacent tick start monotonic_ns; final tick explicitly closed; no catchup or skips",
        realtime_gate="each async episode >=95% cycles <=55ms", events_clock="monotonic_ns",
        worker_tick="unknown/null; request_tick separately recorded")
    spec["video_rule"].update(main_examples="minimum selected bank index, L20, each task, regardless of success",
        raw="cached observations with actual frame sample time and age; no extra render or rollout")
    modules = root/"src/sim_vla/latency_v2"
    files = list(modules.glob("*.py"))+list((root/"tests").glob("test_latency_v2*.py"))
    files += [root/entry["path"] for entry in previous["implementation_inventory"]["files"]
              if not entry["path"].startswith(("src/", "tests/"))]
    spec["implementation_inventory"] = inventory(root, set(files))
    templates = {}
    for name in matrices:
        templates[name] = {**previous["formal_run_config_templates"][name], "run_id": RUN_ID,
            "run_dir": str((base/("formal_"+name)).relative_to(root)), "observation_mode": "on_request",
            "profile_env": True, "formal_spec_path": str((directory/"experiment_spec.json").relative_to(root)),
            "trials_path": str((directory/(name+"_trials.json")).relative_to(root))}
    spec["formal_run_config_templates"] = templates
    spec["spec_sha256"] = canonical_hash(spec)
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in matrices.items():
        publish(directory/(name+"_trials.json"), rows)
        publish(directory/(name+"_config.json"), {**templates[name], "formal_spec_sha256": spec["spec_sha256"]})
    for relative in set(spec["semantic_code"]) | {row["path"] for row in spec["implementation_inventory"]["files"]}:
        target = directory/"code_snapshot"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/relative, target)
    publish(directory/"experiment_spec.json", spec)
    return spec
