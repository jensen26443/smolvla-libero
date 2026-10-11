"""Real control ticks with explicit startup, tail cost and sustained diagnostics."""
import time
import traceback

import numpy as np

from ..latency_v2.runner_sim import diagnostic_stage, save_trace
from ..latency_v2.schema import ObservationSnapshot
from ..latency_v2.timing import request_record, result_record
from ..latency_v2.worker import CausalDelayEstimator
from .execution import CatchupExecutor
from .schema import CoverageInfeasible, MethodConfig


def run_episode(env, engine, trial, directory, *, asynchronous=False, states=None,
                generation=0, version="", video=False, sustain=False):
    cfg = MethodConfig.from_resolved(trial["config"])
    initial_p95, max_ticks = cfg.initial_endpoint_p95_s, cfg.max_ticks
    estimator_window, tick_seconds = cfg.estimator_window, cfg.tick_seconds
    cycle_limit_seconds, required_cycle_fraction = cfg.cycle_limit_seconds, cfg.required_cycle_fraction
    reset_started = time.monotonic()
    observation = env.start_episode(trial["init_index"], trial["seed"], states=states)
    reset_seconds = time.monotonic()-reset_started
    initial_digest = env.state_identity()
    executor = CatchupExecutor(cfg, generation=generation, seed=trial["seed"], version=version,
                                attempt_id=f"attempt-{generation:06d}", directory=directory)
    estimator = CausalDelayEstimator(initial_p95=initial_p95, window=estimator_window, tick_seconds=tick_seconds)
    actions, prefixes, guides, requests, chunks = [], [], [], [], []
    frames, diagnostics, observations, frame_sampling, cycles = [], [], [], [], []
    pending_sim = None
    success, reason, error = False, "timeout", None
    fallback_run = fallback_max = 0
    episode_started = time.monotonic()
    startup_seconds = control_seconds = 0.
    first_success_step = None
    previous_cycle = control_start_ns = None

    def capture(tick):
        current = env.sample_observation(tick)
        return ObservationSnapshot.capture(directory.name, tick, current, env.instruction)

    def submit(tick, estimate):
        prepare_started = time.monotonic()
        snapshot = capture(tick)
        request = executor.submit(snapshot, estimate, observation_prepare_s=time.monotonic()-prepare_started)
        padded = np.zeros((50, 7))
        padded[:len(request.prefix.actions)] = request.prefix.actions
        prefixes.append(padded)
        guide = np.zeros((50, 7))
        if request.guide_actions is not None:
            guide[:len(request.guide_actions)] = request.guide_actions
        guides.append(guide)
        state = request.snapshot.robot_state
        observations.append(np.concatenate((state["eef"]["pos"], state["eef"]["quat"],
                                            state["gripper"]["qpos"][:1])))
        requests.append({**request_record(request), "estimate": estimate,
            "remaining_old": request.remaining_old, "effective_period": executor.config.period,
            "guide_length": 0 if request.guide_actions is None else len(request.guide_actions),
            "sampling_steps": request.sampling_steps, "sampler_mode": request.sampler_mode,
            "observation_prepare_s": request.observation_prepare_s,
            "commitment_boundary": request.prefix.end_tick})
        if asynchronous:
            engine.submit(request)
            return None
        result = engine.infer_request(request)
        requests[-1].update(result_record(result))
        if result.actions is not None:
            chunks.append(result.actions)
        return result

    def receive(result, tick, *, startup=False, after_episode=False):
        if result.received_monotonic_ns is None:
            result.received_monotonic_ns = time.monotonic_ns()
        row = requests[result.request.request_id]
        row.update({**result_record(result), "t_ready": tick, "startup": startup,
                    "completed_after_episode": after_episode})
        if asynchronous and result.actions is not None:
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
            if asynchronous and not startup:
                estimator.observe(result)
            executor.receive(result, tick)

    def complete_cycle(tick, start_ns, end_ns, work):
        seconds = (end_ns-start_ns)/1e9
        cycles.append(seconds)
        executor.events.append({"event": "cycle_complete", "tick": tick, "tick_wall_s": seconds,
            "start_monotonic_ns": start_ns, "end_monotonic_ns": end_ns,
            "control_work_s": work, "deadline_missed": work > tick_seconds, "monotonic_ns": end_ns})

    try:
        if not asynchronous:
            engine.reset()
        startup = submit(0, 0)
        receive(engine.drain() if asynchronous else startup, 0, startup=True)
        executor.advance(0)
        startup_seconds = time.monotonic()-episode_started
        control_start_ns = time.monotonic_ns()
        for tick in range(max_ticks):
            tick_start_ns = time.monotonic_ns()
            if previous_cycle is not None:
                complete_cycle(*previous_cycle[:2], tick_start_ns, previous_cycle[2])
                previous_cycle = None
            if asynchronous:
                result = engine.poll()
                if result is not None:
                    receive(result, tick)
            elif pending_sim is not None and pending_sim[1] <= tick:
                receive(pending_sim[0], tick)
                pending_sim = None
            executor.advance(tick)
            estimate = estimator.ticks() if asynchronous else trial["latency"]
            if executor.due(tick, estimate):
                result = submit(tick, estimate)
                if not asynchronous:
                    release = tick+trial["latency"]
                    requests[-1]["simulated_release_tick"] = release
                    pending_sim = result, release
            action, meta = executor.action(tick)
            fallback_run = fallback_run+1 if meta["neutral"] else 0
            fallback_max = max(fallback_max, fallback_run)
            env_started = time.monotonic()
            observation, _, terminated, truncated, info = env.step(action)
            env_seconds = time.monotonic()-env_started
            actions.append(action.copy())
            diagnostics.append(env.diagnostic())
            if video:
                frames.append(env.cached_frame())
                frame_sampling.append({"tick": tick+1, "sampling": observation.get("sampling", {}),
                    "frame_age_ticks": tick+1-observation.get("sampling", {}).get("image", {}).get("tick", tick+1)})
            current_success = bool(info.get("is_success", False))
            success = success or current_success
            if current_success and first_success_step is None:
                first_success_step = tick+1
            executor.events.append({"event": "step", "tick": tick, **meta, "action": action.tolist(),
                "success": success, "in_flight": executor.pending is not None, "env_step_s": env_seconds,
                "env_timing": env.last_step_timing, "sampling": observation.get("sampling", {}),
                "causal_estimate": estimate, "remaining_old": executor.remaining(tick)})
            work = (time.monotonic_ns()-tick_start_ns)/1e9
            if asynchronous and work < tick_seconds:
                time.sleep(tick_seconds-work)
            previous_cycle = tick, tick_start_ns, work
            ordinary_success = current_success and not truncated
            if ((success and not sustain) or truncated or (terminated and not ordinary_success)
                    or fallback_run >= cfg.fallback_limit_ticks):
                reason = "success" if success and not sustain else (
                    "fallback_timeout" if fallback_run >= cfg.fallback_limit_ticks else "env_done")
                break
        else:
            if sustain:
                reason = "sustained_300_ticks"
    except CoverageInfeasible as failure:
        reason = "coverage_infeasible"
        executor.events.append({"event": "policy_failure", "tick": len(actions), "reason": str(failure)})
    except Exception:  # noqa: BLE001 -- preserve method exceptions and their denominator
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
    if asynchronous and engine.busy:
        try:
            receive(engine.drain(), None, after_episode=True)
        except (RuntimeError, TimeoutError) as failure:
            reason, success, error = "method_error", False, str(failure)
            executor.events.append({"event": "drain_error", "tick": None, "error": error})
    if pending_sim is not None:
        requests[pending_sim[0].request.request_id]["simulated_unreleased"] = True
    if video and not frames:
        import imageio.v2 as imageio

        # Actual cached initial scene, explicitly distinct from a control video.
        imageio.imwrite(directory/"initial_no_control.png", env.cached_frame())
    ticks = [row for row in executor.events if row["event"] == "step"]
    sim_seconds = max(tick_seconds*len(actions), tick_seconds)
    generation_seconds = sum(row.get("stages", {}).get("forward_s", 0) for row in requests)
    episode_wall_seconds = time.monotonic()-episode_started
    activations = [row for row in executor.events if row["event"] == "activate"]
    prefix_ids = {row["request_id"] for row in requests if row["prefix_length"]}
    for row in executor.events:
        if row["event"] in ("activate", "late_discard", "expired_discard", "receive_age"):
            requests[row["request_id"]].update({key: row[key] for key in (
                "commitment_boundary", "receive_age", "activation_age", "late_ticks", "catchup_eligible",
                "catchup_accepted", "skipped_actions", "late_fallback_ticks", "start_index", "reason") if key in row})
            requests[row["request_id"]]["decision"] = row["event"]
    step_invariant = [row["tick"] for row in ticks] == list(range(len(actions))) and len(cycles) == len(actions)
    outcome = {**trial, "success": None if sustain else success, "termination_reason": reason, "error": error,
        "sustained_diagnostic": sustain, "sustained_300_ticks": sustain and len(actions) == 300 and not error,
        "reset_wall_s": reset_seconds, "effective_config": cfg.resolved(),
        "delay_estimator": {"initial_endpoint_p95_s": initial_p95, "window": estimator_window,
                            "percentile": 95, "tick_seconds": tick_seconds},
        "control_settings": {"tick_seconds": tick_seconds, "cycle_limit_seconds": cycle_limit_seconds,
                             "required_cycle_fraction": required_cycle_fraction},
        "steps": len(actions), "n_requests": executor.ordinal, "startup_wait_s": startup_seconds,
        "control_wall_s": control_seconds, "wall_seconds": episode_wall_seconds,
        "inference_seconds": sum(row.get("endpoint_s", 0) for row in requests),
        "generation_seconds": generation_seconds,
        "requests_per_sim_second": executor.ordinal/sim_seconds,
        "generation_seconds_per_sim_second": generation_seconds/sim_seconds,
        "requests_per_wall_second": executor.ordinal/max(episode_wall_seconds, 1e-9),
        "generation_seconds_per_wall_second": generation_seconds/max(episode_wall_seconds, 1e-9),
        "actual_control_hz": len(actions)/max(control_seconds, 1e-9),
        "cycle_p50_p95_p99_s": np.percentile(cycles, [50, 95, 99]).tolist() if cycles else [],
        "deadline_missed_fraction": float(np.mean([row["control_work_s"] > tick_seconds for row in executor.events
                                                  if row["event"] == "cycle_complete"])) if cycles else 1.,
        "fraction_cycles_under_55ms": float(np.mean(np.asarray(cycles) <= cycle_limit_seconds)) if cycles else 0.,
        "realtime_20hz_passed": (bool(cycles) and len(cycles) == len(actions)
            and float(np.mean(np.asarray(cycles) <= cycle_limit_seconds)) >= required_cycle_fraction) if asynchronous else None,
        "no_catchup_or_skipped_steps": step_invariant,
        "control_step_invariant": step_invariant,
        "initial_state_after_settle_sha256": initial_digest,
        "first_success_step": first_success_step,
        "prefix_requests": len(prefix_ids),
        "activation_count": len(activations),
        "prefix_activations": sum(row["request_id"] in prefix_ids for row in activations),
        "continuous_continuations": sum(row["request_id"] in prefix_ids for row in activations),
        "catchup_accepted": sum(row["catchup_accepted"] for row in activations),
        "catchup_rejected": sum(row["event"] in ("late_discard", "expired_discard") for row in executor.events),
        "late_fallback_ticks": sum(row.get("late_fallback", False) for row in ticks),
        "switch_command_l2": [row["switch_command_l2"] for row in activations if row["switch_command_l2"] is not None],
        "late_distribution_ticks": [row["late_ticks"] for row in executor.events
                                      if row["event"] in ("activate", "late_discard", "expired_discard")],
        "max_consecutive_fallback": fallback_max, "fallback_ticks": sum(row["neutral"] for row in ticks),
        "late_results": sum(row["event"] == "late_discard" for row in executor.events),
        "expired_results": sum(row["event"] == "expired_discard" for row in executor.events),
        "neutral_prefix_positions": sum(len(row.get("neutral_prefix_positions", [])) for row in executor.events),
        "coverage_infeasible_count": sum(row["event"] == "coverage_infeasible" for row in executor.events),
        "failure_stage_heuristic": diagnostic_stage(diagnostics, actions, success),
        "code_sha256": version, "backend": "actual_spawned_gpu_worker" if asynchronous else "simulated_release_real_model_and_physics",
        "frame_sampling": frame_sampling}
    np.savez_compressed(directory/"guide_arrays.npz", guide_actions=np.asarray(guides).reshape(-1, 50, 7))
    save_trace(directory, outcome, executor.events, actions, prefixes, requests, chunks,
               frames, diagnostics, observations)
    if not step_invariant:
        raise RuntimeError("control step or cycle invariant failed; raw evidence retained")
    return outcome
