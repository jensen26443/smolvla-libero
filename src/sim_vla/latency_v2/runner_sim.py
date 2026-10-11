"""Real frozen policy and physics, with simulated control-tick release times."""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .alignment import Executor
from .schema import MethodConfig, ObservationSnapshot
from .storage import canonical_hash, publish
from .timing import provenance_complete, request_record, result_record


def latency_sequence(trial: dict, length: int = 300) -> np.ndarray:
    profile = trial.get("profile", "fixed")
    if profile == "fixed":
        return np.full(length, trial["latency"], dtype=int)
    # Method/K/flow RNG do not participate in this exogenous control-tick trajectory.
    seed = int(canonical_hash([trial["task"], trial["init_index"], profile, 20261001])[:8], 16)
    generator = np.random.default_rng(seed)
    values, probabilities = (([3, 5, 7], [.25, .5, .25]) if profile == "J1"
                             else ([10, 20, 40], [.4, .4, .2]))
    return generator.choice(values, length, p=probabilities)


def diagnostic_stage(diagnostics: list[dict], actions: list[np.ndarray], success: bool) -> str:
    if success:
        return "success"
    if not diagnostics:
        return "unknown"
    object_positions = np.asarray([row["object_position"] for row in diagnostics])
    eef_positions = np.asarray([row["eef_position"] for row in diagnostics])
    lifted = np.max(object_positions[:, 2]) > object_positions[0, 2] + .025
    close = np.min(np.linalg.norm(eef_positions - object_positions, axis=1)) < .08
    if lifted:
        released = any(action[-1] < 0 for action in actions[len(actions)//2:])
        return "release" if released else "transport_contact"
    return "grasp" if close else "approach"


def save_trace(directory: Path, result: dict, events: list[dict], actions: list[np.ndarray],
               prefixes: list[np.ndarray], requests: list[dict], chunks: list[np.ndarray],
               frames: list[np.ndarray], diagnostics: list[dict], observations: list[np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if hasattr(events, "close"):
        events.close()
        result["log_error"] = events.error
        result["timing_provenance_complete"] = provenance_complete(events, requests, log_error=events.error)
        result["attempt_id"] = events.attempt_id
        result["generation"] = events.generation
    for row in events:
        if row["event"] == "activate":
            requests[row["request_id"]].update(t_act=row["tick"], activation_monotonic_ns=row["monotonic_ns"])
        if row["event"] == "cycle_complete":
            step = next(item for item in events if item["event"] == "step" and item["tick"] == row["tick"])
            step["tick_wall_s"] = row["tick_wall_s"]
            step["control_work_s"] = row["control_work_s"]
            step["deadline_missed"] = row["deadline_missed"]
    with (directory / "events.jsonl").open("w") as file:
        for event in events:
            file.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    np.savez_compressed(directory / "arrays.npz", actions=np.asarray(actions).reshape(-1, 7),
                        prefixes=np.asarray(prefixes).reshape(-1, 50, 7),
                        chunks=np.asarray(chunks).reshape(-1, 50, 7),
                        request_observations=np.asarray(observations).reshape(-1, 8))
    publish(directory / "requests.json", requests)
    publish(directory / "diagnostics.json", diagnostics)
    publish(directory / "frame_sampling.json", result.pop("frame_sampling", []))
    if frames:
        import imageio.v2 as imageio

        imageio.mimwrite(directory / "episode.mp4", frames, fps=20, quality=8, macro_block_size=1)
        # Same minimum initialization selected in advance, regardless of success.
        for name, index in (("start", 0), ("middle", len(frames)//2), ("end", len(frames)-1)):
            imageio.imwrite(directory / f"{name}.png", frames[index])
    # This is the only completion marker; interrupted arrays/events cannot appear completed.
    publish(directory / "result.json", result)


def run_episode(env, policy, trial: dict, directory: Path, *, states=None, max_ticks: int = 300,
                video: bool = False, generation: int = 0, version: str = "", fallback_limit: int = 60) -> dict:
    config = MethodConfig(**trial["config"])
    observation = env.start_episode(trial["init_index"], trial["seed"], states=states)
    initial_digest = env.state_identity()
    executor = Executor(config, generation=generation, seed=trial["seed"], version=version, directory=directory)
    actions, prefixes, requests, chunks, frames, diagnostics, observations = [], [], [], [], [], [], []
    frame_sampling = []
    success, reason, error = False, "timeout", None
    pending = None
    elapsed, fallback_run, fallback_max = 0.0, 0, 0
    delays = latency_sequence(trial, max_ticks)
    past_completed = []
    started = time.monotonic()
    try:
        policy.reset()
        for tick in range(max_ticks):
            tick_start = time.monotonic()
            if pending is not None and pending[1] <= tick:
                result, _ = pending
                executor.receive(result, tick)
                requests[result.request.request_id].update(result_record(result))
                past_completed.append(tick - result.request.request_tick)
                pending = None
            executor.advance(tick)
            estimate = trial["latency"]
            if trial.get("profile", "fixed") != "fixed":
                mean = 5 if trial["profile"] == "J1" else 20
                estimate = int(np.ceil(np.mean(past_completed[-10:]))) if past_completed else mean
            # Capturing copies only when the scheduler can submit; no hidden image-based trigger.
            if executor.pending is None and (executor.urgent or tick-executor.last_request_tick >= config.period):
                if hasattr(env, "sample_observation"):
                    observation = env.sample_observation(tick)
                snapshot = ObservationSnapshot.capture(directory.name, tick, observation, env.instruction)
                request = executor.submit(snapshot, estimate)
                padded = np.zeros((50, 7))
                padded[:len(request.prefix.actions)] = request.prefix.actions
                prefixes.append(padded)
                state = observation["robot_state"]
                observations.append(np.concatenate((state["eef"]["pos"], state["eef"]["quat"],
                                                    state["gripper"]["qpos"][:1])))
                requests.append({**request_record(request), "estimate": estimate})
                infer_started = time.monotonic()
                result = policy.infer_request(request)
                endpoint = time.monotonic() - infer_started
                elapsed += endpoint
                if result.actions is not None:
                    chunks.append(result.actions)
                release = tick + int(delays[tick])
                requests[-1].update({**result_record(result),
                                 "t_ready": release, "endpoint_s": endpoint, "estimate": estimate,
                                 "simulated_release_tick": release})
                if release == tick:
                    executor.receive(result, tick)
                    requests[-1].update(result_record(result))
                    executor.advance(tick)
                    past_completed.append(0)
                else:
                    pending = (result, release)
            action, meta = executor.action(tick)
            fallback_run = fallback_run + 1 if meta["neutral"] else 0
            fallback_max = max(fallback_max, fallback_run)
            env_start = time.monotonic()
            observation, _, terminated, truncated, info = env.step(action)
            env_seconds = time.monotonic() - env_start
            actions.append(action.copy())
            diagnostics.append(env.diagnostic())
            if video:
                frames.append(env.cached_frame())
                frame_sampling.append({"tick": tick+1, "sampling": observation.get("sampling", {}),
                    "frame_age_ticks": tick+1-observation.get("sampling", {}).get("image", {}).get("tick", tick+1)})
            success = bool(info.get("is_success", False))
            executor.events.append({"event": "step", "tick": tick, **meta, "action": action.tolist(),
                                    "in_flight": executor.pending is not None, "success": success,
                                    "env_step_s": env_seconds, "tick_wall_s": time.monotonic()-tick_start,
                                    "env_timing": getattr(env, "last_step_timing", {}),
                                    "sampling": observation.get("sampling", {})})
            if success or terminated or truncated or fallback_run >= fallback_limit:
                reason = "success" if success else ("control_failure" if fallback_run >= fallback_limit else "env_done")
                break
    except Exception:  # noqa: BLE001 -- persist the exact method error, never silently retry a scored trial
        error = traceback.format_exc()
        reason = "method_error"
        executor.events.append({"event": "error", "tick": len(actions), "error": error})
    except BaseException:
        executor.events.append({"event": "interrupt", "tick": len(actions)})
        executor.events.close()
        raise
    wall = time.monotonic()-started
    if pending is not None:
        requests[pending[0].request.request_id]["simulated_unreleased"] = True
    result = {**trial, "success": success, "termination_reason": reason, "error": error,
              "steps": len(actions), "n_requests": executor.ordinal,
              "inference_seconds": elapsed, "generation_seconds": sum(
                  row.get("stages", {}).get("forward_s", 0) for row in requests), "wall_seconds": wall,
              "first_success_step": len(actions) if success else None,
              "requests_per_sim_second": executor.ordinal / max(len(actions)*.05, .05),
              "generation_seconds_per_sim_second": sum(row.get("stages", {}).get("forward_s", 0)
                  for row in requests) / max(len(actions)*.05, .05),
              "initial_state_after_settle_sha256": initial_digest,
              "latency_trajectory_sha256": canonical_hash(delays.tolist()),
              "max_consecutive_fallback": fallback_max,
              "fallback_ticks": sum(row.get("neutral", False) for row in executor.events if row["event"] == "step"),
              "late_results": sum(row["event"] == "late_discard" for row in executor.events),
              "expired_results": sum(row["event"] == "expired_discard" for row in executor.events),
              "failure_stage_heuristic": diagnostic_stage(diagnostics, actions, success),
              "config_sha256": canonical_hash(asdict(config)), "code_sha256": version,
              "backend": "simulated_release_real_model_and_physics", "realtime_20hz_passed": None,
              "frame_sampling": frame_sampling}
    save_trace(directory, result, executor.events, actions, prefixes, requests, chunks,
               frames, diagnostics, observations)
    return result
