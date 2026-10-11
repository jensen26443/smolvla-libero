"""Actual GPU-worker/control overlap; no simulated release or inference sleeps."""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

from ..assets import project_root
from .alignment import Executor
from .env import ControlledEnv
from .experiments import CHECKPOINT, completed_results, semantic_inventory
from .initializations import check_formal_manifest
from .layout import completion_flags, write_progress
from .runner_sim import diagnostic_stage, save_trace
from .schema import MethodConfig, ObservationSnapshot
from .storage import Budget, canonical_hash, prepare_run, publish, trial_key
from .worker import CausalDelayEstimator, InferenceWorker


def run_async_episode(env, worker, trial, directory, *, states=None, generation=0, version="",
                      max_ticks=300, video=False, initial_p95=1.10448796484, fallback_limit=60):
    from .timing import request_record, result_record

    config = MethodConfig(**trial["config"])
    observation = env.start_episode(trial["init_index"], trial["seed"], states=states)
    initial_digest = env.state_identity()
    executor = Executor(config, generation=generation, seed=trial["seed"], version=version, directory=directory)
    estimator = CausalDelayEstimator(initial_p95=initial_p95)
    actions, prefixes, requests, chunks, frames, diagnostics, observations = [], [], [], [], [], [], []
    frame_sampling, cycles = [], []
    success, reason, error = False, "timeout", None
    fallback_run, fallback_max = 0, 0
    episode_start = time.monotonic()

    def record_result(result, tick, *, startup=False, after_episode=False):
        if result.received_monotonic_ns is None:
            result.received_monotonic_ns = time.monotonic_ns()
        row = requests[result.request.request_id]
        row.update({**result_record(result), "t_ready": tick, "startup": startup,
                    "completed_after_episode": after_episode})
        if result.actions is not None:
            chunks.append(result.actions)
        if after_episode:
            executor.events.append({"event": "worker_complete", "tick": None,
                "request_id": result.request.request_id, "request_tick": result.request.request_tick,
                "monotonic_ns": result.worker_completed_monotonic_ns, "error": result.error})
            executor.events.append({"event": "control_receive", "tick": None,
                "request_id": result.request.request_id, "monotonic_ns": result.received_monotonic_ns,
                "after_episode": True, "error": result.error})
            if result.error:
                raise RuntimeError(result.error)
        else:
            estimator.observe(result)
            executor.receive(result, tick)
            row.update(result_record(result))

    def submit(snapshot, estimate):
        request = executor.submit(snapshot, estimate)
        padded = np.zeros((config.horizon, 7))
        padded[:len(request.prefix.actions)] = request.prefix.actions
        prefixes.append(padded)
        state = snapshot.robot_state
        observations.append(np.concatenate((state["eef"]["pos"], state["eef"]["quat"], state["gripper"]["qpos"][:1])))
        requests.append({**request_record(request), "estimate": estimate})
        worker.submit(request)

    def capture(tick, current):
        if hasattr(env, "sample_observation"):
            current = env.sample_observation(tick)
        return ObservationSnapshot.capture(directory.name, tick, current, env.instruction)

    def complete_cycle(tick, start_ns, end_ns, work_seconds):
        seconds = (end_ns-start_ns)/1e9
        cycles.append(seconds)
        executor.events.append({"event": "cycle_complete", "tick": tick, "tick_wall_s": seconds,
            "start_monotonic_ns": start_ns, "end_monotonic_ns": end_ns, "control_work_s": work_seconds,
            "deadline_missed": work_seconds > .05, "monotonic_ns": end_ns})

    startup_seconds, control_seconds = 0.0, 0.0
    previous_cycle, control_start_ns = None, None
    try:
        # No physics steps during the identical stationary startup protocol.
        submit(capture(0, observation), 0)
        record_result(worker.drain(), 0, startup=True)
        executor.advance(0)
        startup_seconds = time.monotonic()-episode_start
        control_start_ns = time.monotonic_ns()
        for tick in range(max_ticks):
            tick_start_ns = time.monotonic_ns()
            if previous_cycle is not None:
                complete_cycle(*previous_cycle[:2], tick_start_ns, previous_cycle[2])
                previous_cycle = None
            result = worker.poll()
            if result is not None:
                record_result(result, tick)
            executor.advance(tick)
            if not worker.busy and executor.pending is None and (
                    executor.urgent or tick-executor.last_request_tick >= config.period):
                submit(capture(tick, observation), estimator.ticks())
            action, meta = executor.action(tick)
            fallback_run = fallback_run+1 if meta["neutral"] else 0
            fallback_max = max(fallback_max, fallback_run)
            env_start = time.monotonic()
            observation, _, terminated, truncated, info = env.step(action)
            env_seconds = time.monotonic()-env_start
            actions.append(action.copy())
            diagnostics.append(env.diagnostic())
            if video:
                frames.append(env.cached_frame())
                frame_sampling.append({"tick": tick+1, "sampling": observation.get("sampling", {}),
                                       "frame_age_ticks": tick+1-observation.get("sampling", {}).get("image", {}).get("tick", tick+1)})
            success = bool(info.get("is_success", False))
            executor.events.append({"event": "step", "tick": tick, **meta, "action": action.tolist(),
                "success": success, "in_flight": worker.busy, "env_step_s": env_seconds,
                "env_timing": getattr(env, "last_step_timing", {}),
                "sampling": observation.get("sampling", {}), "causal_estimate": estimator.ticks()})
            # Includes sampling, diagnostics, event construction and nonblocking enqueue.
            work_seconds = (time.monotonic_ns()-tick_start_ns)/1e9
            if work_seconds < .05:
                time.sleep(.05-work_seconds)
            previous_cycle = (tick, tick_start_ns, work_seconds)
            if success or terminated or truncated or fallback_run >= fallback_limit:
                reason = "success" if success else ("control_failure" if fallback_run >= fallback_limit else "env_done")
                break
    except Exception:  # noqa: BLE001 -- retain and score the exact method/worker failure
        error, reason = traceback.format_exc(), "method_error"
        executor.events.append({"event": "error", "tick": len(actions), "error": error})
    except BaseException:
        executor.events.append({"event": "interrupt", "tick": len(actions)})
        executor.events.close()
        raise
    if previous_cycle is not None:
        complete_cycle(*previous_cycle[:2], time.monotonic_ns(), previous_cycle[2])
    if control_start_ns is not None:
        control_seconds = (time.monotonic_ns()-control_start_ns)/1e9
    executor.events.append({"event": "control_end", "tick": len(actions), "reason": reason})
    if worker.busy:
        try:
            record_result(worker.drain(), None, after_episode=True)
        except (RuntimeError, TimeoutError) as failure:
            reason, success, error = "method_error", False, str(failure)
            executor.events.append({"event": "drain_error", "tick": None, "error": error})
    ticks = [row for row in executor.events if row["event"] == "step"]
    realtime = bool(cycles) and len(cycles) == len(actions) and float(np.mean(np.asarray(cycles) <= .055)) >= .95
    outcome = {**trial, "success": success, "termination_reason": reason, "error": error,
        "steps": len(actions), "n_requests": executor.ordinal, "startup_wait_s": startup_seconds,
        "control_wall_s": control_seconds, "wall_seconds": time.monotonic()-episode_start,
        "inference_seconds": sum(row.get("endpoint_s", 0) for row in requests),
        "generation_seconds": sum(row.get("stages", {}).get("forward_s", 0) for row in requests),
        "requests_per_sim_second": executor.ordinal/max(len(actions)*.05, .05),
        "generation_seconds_per_sim_second": sum(row.get("stages", {}).get("forward_s", 0) for row in requests)/max(len(actions)*.05, .05),
        "actual_control_hz": len(actions)/max(control_seconds, 1e-9),
        "cycle_p50_p95_p99_s": np.percentile(cycles, [50, 95, 99]).tolist() if cycles else [],
        "deadline_missed_fraction": float(np.mean([row["control_work_s"] > .05 for row in executor.events if row["event"] == "cycle_complete"])) if cycles else 1.,
        "fraction_cycles_under_55ms": float(np.mean(np.asarray(cycles) <= .055)) if cycles else 0.,
        "realtime_20hz_passed": realtime, "no_catchup_or_skipped_steps": len(cycles) == len(actions),
        "initial_state_after_settle_sha256": initial_digest, "first_success_step": len(actions) if success else None,
        "max_consecutive_fallback": fallback_max, "fallback_ticks": sum(row["neutral"] for row in ticks),
        "late_results": sum(row["event"] == "late_discard" for row in executor.events),
        "expired_results": sum(row["event"] == "expired_discard" for row in executor.events),
        "failure_stage_heuristic": diagnostic_stage(diagnostics, actions, success),
        "config_sha256": canonical_hash(trial["config"]), "code_sha256": version,
        "backend": "actual_spawned_gpu_worker", "frame_sampling": frame_sampling}
    save_trace(directory, outcome, executor.events, actions, prefixes, requests, chunks,
               frames, diagnostics, observations)
    return outcome


def run_async_matrix(config: dict, trials: list[dict], *, resume=False, dry_run=False):
    root, directory = project_root(), project_root()/config["run_dir"]
    code = semantic_inventory(root, asynchronous=True)
    spec = {"config": config, "semantic_code": code, "semantic_code_sha256": canonical_hash(code),
            "backend": "actual_spawned_gpu_worker", "startup_protocol": "stationary original inference ordinal zero",
            "target_hz": 20, "fallback_limit": 60, "estimator": "past completed endpoint p95 window 10"}
    if dry_run:
        return {"expected_count": len(trials), "spec": spec}
    frozen = prepare_run(directory, spec, trials, resume=resume)
    budget = Budget(root/"runs/latency_v2_budget.json")
    states, poses = {}, {}
    if config.get("initialization_manifest"):
        audit = json.loads((root/"runs/latency_v2_audit/audit.json").read_text())
        checked = check_formal_manifest(root/config["initialization_manifest"], audit["initializations"])
        states = {task: np.load(row["path"], allow_pickle=False) for task, row in checked["files"].items()}
        from .model_poses import ModelPoseBank

        poses = {task: ModelPoseBank.load(Path(row["model_poses_path"]), expected_states=len(states[task]))
                 for task, row in checked["files"].items()}
    elif config["stage"].startswith("formal"):
        raise ValueError("formal async requires certified fresh physical states")
    worker = InferenceWorker(checkpoint=root/config.get("checkpoint", CHECKPOINT))
    envs = {}
    try:
        for ordinal, trial in enumerate(trials):
            target = directory/"trials"/trial_key(trial)
            if (target/"result.json").exists():
                continue
            budget.prepare_retry(target)
            with budget.track(trial_key(trial), config["stage"], target, trial,
                              run_id=config.get("run_id", directory.name)) as attempt:
                if trial["task"] not in envs:
                    envs[trial["task"]] = ControlledEnv(task_key=trial["task"],
                        observation_mode=config.get("observation_mode", "per_tick"),
                        profile_env=config.get("profile_env", False))
                    envs[trial["task"]].model_pose_bank = poses.get(trial["task"])
                outcome = run_async_episode(envs[trial["task"]], worker, trial, target,
                    states=states.get(trial["task"]), generation=attempt, version=spec["semantic_code_sha256"],
                    max_ticks=config.get("max_ticks", 300), video=config["stage"].startswith("formal") or trial["init_index"] == 0)
            write_progress(config, directory, completed_results(directory, require_complete=False), len(trials), budget.remaining())
            print(json.dumps({"stage": config["stage"], "completed": ordinal+1, "total": len(trials),
                "trial": trial, "success": outcome["success"], "reason": outcome["termination_reason"],
                "actual_hz": outcome["actual_control_hz"]}), flush=True)
            if outcome["error"]:
                raise RuntimeError(outcome["error"])
    finally:
        publish(directory/"worker_shutdown.json", worker.close())
        for env in envs.values():
            env.close()
    rows = completed_results(directory)
    publish(directory/"complete.json", {"completed": len(rows), "spec_sha256": frozen["spec_sha256"],
        **completion_flags(directory, rows, len(trials))})
    return {"completed": len(rows), "run_dir": str(directory)}
