"""Executable S0 audit. All real rollouts use development init 0 only."""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..protocol import dump_json
from .initializations import audit_native_initializations
from .integrity import file_sha256, inventory, source_inventory


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(array), "mean": float(array.mean()),
        **{f"p{q}": float(np.percentile(array, q)) for q in (50, 95, 99)},
        "min": float(array.min()), "max": float(array.max()),
    }


def historical_matrix(root: Path) -> dict:
    directory = root / "runs/latency_formal_v1"
    records = [json.loads(path.read_text()) for path in sorted(directory.glob("trials/*/result.json"))]
    ids = [record["trial_id"] for record in records]
    state = json.loads((directory / "matrix.json").read_text())["statuses"]
    groups = Counter((r["task"], r["method"], r["latency"]) for r in records)
    return {
        "source": str(directory), "records": len(records), "unique_trial_ids": len(set(ids)),
        "missing_from_records": sorted(set(state) - set(ids)),
        "unexpected_records": sorted(set(ids) - set(state)),
        "attempt_histogram": dict(Counter(str(r["attempt"]) for r in records)),
        "conditions": [
            {"task": task, "method": method, "latency": latency, "trials": count}
            for (task, method, latency), count in sorted(groups.items())
        ],
        "composition": {"r0": 40, "main": 480, "long_period_ablations": 160, "supplements": 120},
        "audit_discrepancies": [
            "The raw matrix is 40+480+160+120=800; REPORT describes the long-period ablations as 80.",
            "The frozen PLAN-03 notes say planned total 760 while the actual expanded matrix has 800.",
            "Historical checkpoint hash stops before the >64 MiB weights and therefore does not hash the model.",
            "Historical official episode ordinal cannot be read as native init index because success resets advance it.",
            "Historical bootstrap uses task/latency/init keys; PLAN-04 must resample complete init clusters across L10/L20.",
        ],
    }


def packet_from_observation(observation: dict, instruction: str):
    from ..latency_eval.contracts import ObservationPacket
    from ..latency_eval.sensor_queue import _deep_copy_observation

    images, state = _deep_copy_observation(observation)
    return ObservationPacket(0, 0, 0, images, state, instruction, {"images": 0, "state": 0})


def env_probe(env, *, init_index: int, seed: int) -> tuple[dict, dict]:
    from ..latency_eval.contracts import FALLBACK_ACTION

    state_hashes, image_hashes = [], []
    for _ in range(2):
        observation = env.start_episode(init_index, seed)
        state_hashes.append(env.get_sim_state_digest())
        import hashlib

        image_hashes.append(hashlib.sha256(observation["pixels"]["image"].tobytes()).hexdigest())
    controller = env._env._env.robots[0].controller
    step_times = []
    for _ in range(20):
        started = time.perf_counter()
        observation, _, terminated, _, _ = env.step(np.array(FALLBACK_ACTION))
        step_times.append(time.perf_counter() - started)
        if terminated:
            raise RuntimeError("the audited neutral action unexpectedly terminated the episode")
    return {
        "task_id": env.task_id, "instruction": env.instruction,
        "image_shapes": {k: list(v.shape) for k, v in observation["pixels"].items()},
        "reset_init_index": init_index, "seed": seed,
        "reset_state_sha256": state_hashes, "reset_image_sha256": image_hashes,
        "deterministic_state": len(set(state_hashes)) == 1,
        "deterministic_image": len(set(image_hashes)) == 1,
        "env_step_seconds": distribution(step_times), "env_step_seconds_raw": step_times,
        "controller": {
            "class": type(controller).__name__, "use_delta": bool(controller.use_delta),
            "input_min": np.asarray(controller.input_min).tolist(),
            "input_max": np.asarray(controller.input_max).tolist(),
            "output_min": np.asarray(controller.output_min).tolist(),
            "output_max": np.asarray(controller.output_max).tolist(),
            "control_freq": float(env._env._env.env.control_freq),
        },
    }, observation


def model_probe(adapter, packet, config: dict) -> dict:
    import torch
    from lerobot.processor.device_processor import DeviceProcessorStep
    from lerobot.processor.normalize_processor import NormalizerProcessorStep
    from lerobot.types import TransitionKey

    cuda = adapter.device.startswith("cuda")
    def synchronize():
        if cuda:
            torch.cuda.synchronize()

    adapter.warmup(packet)
    h2d = {"start": 0.0, "seconds": 0.0}
    def before(index, transition):
        if isinstance(adapter.preprocessor.steps[index], DeviceProcessorStep):
            synchronize()
            h2d["start"] = time.perf_counter()

    def after(index, transition):
        if isinstance(adapter.preprocessor.steps[index], DeviceProcessorStep):
            synchronize()
            h2d["seconds"] += time.perf_counter() - h2d["start"]

    adapter.preprocessor.register_before_step_hook(before)
    adapter.preprocessor.register_after_step_hook(after)
    samples = []
    try:
        for ordinal in range(config["timing_calls"]):
            synchronize()
            h2d["seconds"] = 0.0
            started = time.perf_counter()
            model_input = adapter._preprocess(packet)
            synchronize()
            prepared = time.perf_counter()
            with torch.inference_mode():
                noise = adapter._make_noise(ordinal)
                raw = adapter.policy.predict_action_chunk(model_input, noise=noise)
            synchronize()
            generated = time.perf_counter()
            actions = adapter._postprocess_chunk(raw)
            finished = time.perf_counter()
            if not np.isfinite(actions).all():
                raise ValueError("non-finite real model action chunk")
            samples.append({
                "ordinal": ordinal,
                "preprocess_s": prepared - started - h2d["seconds"],
                "h2d_s": h2d["seconds"],
                "generation_s": generated - prepared,
                "postprocess_d2h_s": finished - generated,
                "end_to_end_s": finished - started,
            })
            if (ordinal + 1) % 10 == 0:
                print(f"S0 timing: {ordinal + 1}/{config['timing_calls']} real GPU calls", flush=True)
    finally:
        adapter.preprocessor.unregister_before_step_hook(before)
        adapter.preprocessor.unregister_after_step_hook(after)

    with torch.inference_mode():
        a = adapter.policy.predict_action_chunk(model_input, noise=adapter._make_noise(999))
        b = adapter.policy.predict_action_chunk(model_input, noise=adapter._make_noise(999))
    normalized_single = adapter.postprocessor(raw[:, 0]).detach().cpu().numpy()[0]
    rng = np.random.default_rng(42100)
    prefix = rng.uniform(-1, 1, size=(1, adapter.chunk_size, 7)).astype(np.float32)
    # Signed discrete test commands include both gripper events.
    prefix[:, :, 6] = np.where(np.arange(adapter.chunk_size) % 2, 1.0, -1.0)
    normalizer = next(s for s in adapter.preprocessor.steps if isinstance(s, NormalizerProcessorStep))
    tensor = torch.from_numpy(prefix).to(adapter.device)
    normalized = normalizer({TransitionKey.ACTION: tensor})[TransitionKey.ACTION]
    padded = adapter.policy.prepare_action({"action": normalized})
    restored = adapter.postprocessor(normalized).detach().cpu().numpy()
    roundtrip_error = float(np.max(np.abs(restored - prefix)))
    mask = np.arange(adapter.max_action_dim) < 7
    passed = (
        tuple(raw.shape) == (1, 50, 7)
        and np.allclose(actions[0], normalized_single, atol=1e-6)
        and torch.equal(a, b)
        and roundtrip_error < 5e-6
        and bool(torch.all(padded[:, :, 7:] == 0))
        and np.isfinite(actions).all()
    )
    return {
        "passed": bool(passed), "warmup_calls": config["warmup_calls"],
        "timing_calls": len(samples), "timing_samples": samples,
        "timing": {k: distribution([s[k] for s in samples]) for k in samples[0] if k != "ordinal"},
        "raw_chunk_shape": list(raw.shape), "environment_chunk_shape": list(actions.shape),
        "padded_action_shape": list(padded.shape), "valid_action_mask": mask.tolist(),
        "action_roundtrip_max_abs_error": roundtrip_error,
        "action_padding_zero": bool(torch.all(padded[:, :, 7:] == 0)),
        "gripper_roundtrip_sign_equal": bool(np.array_equal(np.sign(restored[:, :, 6]), np.sign(prefix[:, :, 6]))),
        "chunk_postprocessor_matches_single": bool(np.allclose(actions[0], normalized_single, atol=1e-6)),
        "same_noise_bitwise_equal": bool(torch.equal(a, b)),
        "unnormalized_range": [float(actions.min()), float(actions.max())],
        "shared_clipping": [-1.0, 1.0],
        "flow_steps": int(adapter.config.num_steps), "flow_dt": -1.0 / adapter.config.num_steps,
        "maximum_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if cuda else None,
        "torch_num_threads": torch.get_num_threads(),
        "timing_scope": "audit only; device-processor synchronization hooks removed before rollout",
    }


def real_old_executor_trial(env, adapter, *, destination: Path, latency: int, period: int, seed: int) -> dict:
    from ..latency_eval.contracts import ActionSpec, SwitchRule
    from ..latency_eval.experiment_matrix import _write_trial_artifacts
    from ..latency_eval.inference_scheduler import FixedDelay
    from ..latency_eval.runner import run_episode
    from ..latency_eval.triggers import FixedPeriodTrigger

    if destination.exists() or destination.with_name(destination.name + ".tmp").exists():
        raise FileExistsError(f"refusing to replace a prior audit trial: {destination}")
    adapter.set_episode_seed(seed)
    trace = run_episode(
        env=env, policy_adapter=adapter, trigger=FixedPeriodTrigger(period),
        action_spec=ActionSpec(), obs_delay_ticks=0, latency_schedule=FixedDelay(latency),
        switch_rule=SwitchRule.IMMEDIATE, max_ticks=300, init_index=0, seed=seed,
    )
    result = {
        "trial_id": destination.name, "stage": "S0_development", "task": env.task_key,
        "init_index": 0, "seed": seed, "period_ticks": period, "latency_ticks": latency,
        "success": trace.outcome.success, "steps": trace.outcome.steps,
        "n_requests": trace.outcome.n_requests, "wall_seconds": trace.outcome.wall_seconds,
        "failure_type": None if trace.outcome.success else trace.outcome.termination_reason,
        "attempt": 1, "native_cursor_after_episode": int(env._env.init_state_id),
    }
    _write_trial_artifacts(destination, result=result, trace=trace, tick_seconds=0.05)
    from ..latency_eval.timeline import plot_rows, rows_from_events_file, write_rows_csv

    rows = rows_from_events_file(destination / "events.jsonl")
    write_rows_csv(rows, destination / "timeline.csv")
    plot_rows(rows, destination / "timeline.png", title=f"S0 real legacy executor: {env.task_key}, L{latency}, K{period}")
    return result


def run_audit(config: dict, *, root: Path) -> dict:
    run_dir = root / config["run_dir"]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"audit output already exists; preserve it and use a new run_dir: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": "S0", "status": "running", "created_unix": time.time(),
        "root": str(root), "config": config, "rollout_attempts": [],
        "initial_discovery_reset_attempts": 1,
        "initial_discovery_note": "one unscored init-0 environment/model-load probe preceded this audit",
        "unscored_reset_probes_this_attempt": 0,
        "prior_audit_unscored_reset_probes": config.get("prior_audit_unscored_reset_probes", 0),
    }
    def publish(stage):
        report["last_completed_probe"] = stage
        dump_json(report, run_dir / "audit.json")
        print(f"S0 audit checkpoint: {stage}", flush=True)

    checkpoint = root / config["checkpoint"]
    try:
        required = {"config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json", "train_config.json"}
        if not checkpoint.is_dir() or not all((checkpoint / name).is_file() for name in required):
            raise FileNotFoundError(f"incomplete E1 checkpoint: {checkpoint}")
        for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            for step in json.loads((checkpoint / name).read_text())["steps"]:
                state_file = step.get("state_file")
                if state_file and not (checkpoint / state_file).is_file():
                    raise FileNotFoundError(f"missing processor statistics: {state_file}")
        report["checkpoint"] = inventory(checkpoint, checkpoint.glob("*"))
        report["model_config"] = json.loads((checkpoint / "config.json").read_text())
        report["training_provenance"] = json.loads((checkpoint / "train_config.json").read_text())
        report["legacy_inventory_before"] = source_inventory(root, legacy_only=True)
        report["code_inventory"] = source_inventory(root)
        import subprocess

        git = subprocess.run(["git", "status", "--short"], cwd=root, capture_output=True, text=True, check=False)
        report["git"] = {"available": git.returncode == 0, "status_or_error": git.stdout or git.stderr}
        report["agents_files"] = [str(p) for p in root.rglob("AGENTS.md") if ".venv" not in p.parts]
        report["historical_matrix"] = historical_matrix(root)
        report["initializations"] = audit_native_initializations(root, config["tasks"])
        publish("checkpoint_and_history")

        from ..assets import libero_assets_dir
        from ..latency_eval.env_adapter import LiberoLatencyEnv
        from ..latency_eval.policy_adapter import SmolVLAPolicyAdapter
        from ..runtime import import_libero

        # Configure the project-local paths before ANY upstream LIBERO import;
        # importing lerobot.envs.libero first would trigger an interactive prompt.
        import_libero(root=root, assets=libero_assets_dir())

        report["libero_assets"] = inventory(libero_assets_dir(), (
            p for p in libero_assets_dir().rglob("*") if ".cache" not in p.relative_to(libero_assets_dir()).parts
        ))
        report["versions"] = {
            "python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in
               ("torch", "lerobot", "hf-libero", "mujoco", "transformers", "gymnasium")},
        }
        from lerobot.envs.libero import LiberoEnv
        from lerobot.scripts.lerobot_eval import rollout
        from robosuite.controllers.osc import OperationalSpaceController
        from robosuite.models.grippers.panda_gripper import PandaGripper

        report["upstream_source"] = {}
        for cls in (LiberoEnv, rollout, OperationalSpaceController, PandaGripper):
            source = Path(inspect.getfile(cls))
            report["upstream_source"][cls.__name__] = {"path": str(source), "sha256": file_sha256(source)}
        report["env_probes"] = {}
        for task_key in config["tasks"]:
            env = LiberoLatencyEnv(task_key=task_key)
            try:
                report["unscored_reset_probes_this_attempt"] += 2
                publish(f"env_probe_started_{task_key}")
                probe, _ = env_probe(env, init_index=0, seed=config["episode_seed"])
                report["env_probes"][task_key] = probe
                publish(f"env_{task_key}")
            finally:
                env.close()

        adapter = SmolVLAPolicyAdapter(
            checkpoint=checkpoint, device=config["device"], warmup_calls=config["warmup_calls"],
            task_ids_for_cfg=(8, 1, 4), seed=config["episode_seed"],
        )
        env = LiberoLatencyEnv(task_key="goal_bowl_stove")
        try:
            report["unscored_reset_probes_this_attempt"] += 1
            publish("model_input_reset_started")
            obs = env.start_episode(0, config["episode_seed"])
            report["model_probe"] = model_probe(adapter, packet_from_observation(obs, env.instruction), config)
            publish("model_10_warmup_50_timed")
        finally:
            env.close()
        for task_key in config["tasks"]:
            env = LiberoLatencyEnv(task_key=task_key)
            try:
                print(f"S0 near-zero real rollout: {task_key}, dev init 0", flush=True)
                report["rollout_attempts"].append({"task": task_key, "latency": 0, "status": "started"})
                publish(f"rollout_started_{task_key}")
                result = real_old_executor_trial(
                    env, adapter, destination=run_dir / "trials" / f"{task_key}__R0__L0__i00",
                    latency=0, period=1, seed=config["episode_seed"],
                )
                report["rollout_attempts"][-1].update(status="completed", result=result)
                publish(f"rollout_completed_{task_key}")
            finally:
                env.close()
        env = LiberoLatencyEnv(task_key="goal_bowl_stove")
        try:
            report["rollout_attempts"].append({"task": "goal_bowl_stove", "latency": 5, "status": "started"})
            publish("legacy_delayed_timeline_started")
            result = real_old_executor_trial(
                env, adapter, destination=run_dir / "trials" / "goal_bowl_stove__B_RAW_k40__L5__i00",
                latency=5, period=40, seed=config["episode_seed"],
            )
            report["rollout_attempts"][-1].update(status="completed", result=result)
        finally:
            env.close()
        charged_resets = 1 + report["unscored_reset_probes_this_attempt"] + report["prior_audit_unscored_reset_probes"]
        charged_attempts = charged_resets + len(report["rollout_attempts"])
        report["budget"] = {
            "cap": config["episode_budget"], "scored_rollout_attempts": len(report["rollout_attempts"]),
            "unscored_reset_probes": charged_resets,
            "attempts_charged_conservatively": charged_attempts,
            "remaining": config["episode_budget"] - charged_attempts,
            "note": "includes prior audit reset probes and discovery; constructor settling excluded",
        }
        if charged_attempts > config["s0_episode_cap"]:
            raise ValueError("S0 attempts exceeded the frozen cap")
        report["legacy_inventory_after"] = source_inventory(root, legacy_only=True)
        report["legacy_preserved"] = report["legacy_inventory_before"] == report["legacy_inventory_after"]
        report["checkpoint_unchanged"] = report["checkpoint"] == inventory(checkpoint, checkpoint.glob("*"))
        report["interface_gate_passed"] = (
            report["model_probe"]["passed"]
            and all(p["deterministic_state"] and p["deterministic_image"] for p in report["env_probes"].values())
            and report["legacy_preserved"] and report["checkpoint_unchanged"]
        )
        if not report["interface_gate_passed"]:
            raise ValueError("S0 model/environment/integrity gate failed; inspect individual probes")
        report["status"] = "passed" if report["initializations"]["formal_asset_gate_passed"] else "blocked_formal_initialization_assets"
        report["blockers"] = [] if report["status"] == "passed" else [{
            "kind": "missing_certifiable_fresh_physical_initializations",
            "required": "20 never historically exposed physical states per supported task, with provenance and array digests",
            "available": {k: len(v["eligible_native_indices"]) for k, v in report["initializations"]["tasks"].items()},
            "plan_constraint": "PLAN-04 section 3.2 prohibits reusing historical formal states or relabeling changed seeds as new states",
            "recovery": "restore a provenance-backed unexposed task-compatible initialization bank, then validate physical digests before G0",
        }]
        publish("complete")
        return report
    except BaseException as error:
        report["status"] = "audit_interrupted" if isinstance(error, KeyboardInterrupt) else "audit_error"
        report["error"] = f"{type(error).__name__}: {error}"
        publish("error")
        raise
