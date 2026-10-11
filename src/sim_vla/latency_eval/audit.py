"""Stage-0 environment/model audit probes (GPU) for PLAN-03 §3.

Produces the evidence behind ``docs/latency/environment_audit.md``: versions,
reset determinism, observation/action specs, chunk semantics, postprocessor
broadcast behavior, noise reproducibility, per-stage inference timing, and
environment step timing. Every probe records pass/fail plus raw numbers; a
failed probe is reported, never silently skipped.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..assets import libero_assets_dir, project_root
from ..protocol import dump_json


def _versions() -> dict[str, Any]:
    import lerobot
    import mujoco
    import torch
    import transformers

    return {
        "python": platform_python(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "lerobot": lerobot.__version__,
        "mujoco": mujoco.__version__,
        "transformers": transformers.__version__,
    }


def platform_python() -> str:
    import platform
    import sys

    return f"{platform.python_version()} ({sys.executable})"


def _env_probe(task_key: str, init_index: int, seed: int) -> dict[str, Any]:
    from .env_adapter import LiberoLatencyEnv

    env = LiberoLatencyEnv(task_key=task_key)
    try:
        digests = []
        obs_digests = []
        for _ in range(2):
            obs = env.start_episode(init_index=init_index, seed=seed)
            digests.append(env.get_sim_state_digest())
            hasher = hashlib.sha256()
            image = obs["pixels"]["image"]
            hasher.update(np.ascontiguousarray(image).tobytes())
            obs_digests.append(hasher.hexdigest())
        obs = env.start_episode(init_index=init_index, seed=seed)
        step_times: list[float] = []
        from lerobot.envs.libero import get_libero_dummy_action

        dummy = np.array(get_libero_dummy_action(), dtype=np.float64)
        for _ in range(20):
            started = time.perf_counter()
            env.step(dummy)
            step_times.append(time.perf_counter() - started)
        render_frame = env.render_frame()
        return {
            "task": task_key,
            "task_id": env.task_id,
            "instruction": env.instruction,
            "reset_deterministic_sim_state": digests[0] == digests[1],
            "reset_deterministic_first_image": obs_digests[0] == obs_digests[1],
            "image_shape": list(obs["pixels"]["image"].shape),
            "image2_shape": list(obs["pixels"]["image2"].shape),
            "robot_state_keys": sorted(obs.get("robot_state", {}).keys()),
            "env_step_seconds_mean": float(np.mean(step_times[2:])),
            "env_step_seconds_p95": float(np.percentile(step_times[2:], 95)),
            "render_available": render_frame is not None,
            "render_shape": None if render_frame is None else list(render_frame.shape),
            "dummy_action": [float(v) for v in dummy],
        }
    finally:
        env.close()


def _model_probe(checkpoint: str | Path, device: str = "cuda", timing_calls: int = 30) -> dict[str, Any]:
    from .env_adapter import LiberoLatencyEnv
    from .policy_adapter import SmolVLAPolicyAdapter

    env = LiberoLatencyEnv(task_key="goal_bowl_stove")
    adapter = SmolVLAPolicyAdapter(checkpoint=checkpoint, device=device, task_ids_for_cfg=(1, 8))
    try:
        obs = env.start_episode(init_index=10, seed=52)
        from sim_vla.latency_eval.contracts import ObservationPacket
        from sim_vla.latency_eval.sensor_queue import _deep_copy_observation

        images, robot_state = _deep_copy_observation(obs)
        packet = ObservationPacket(
            packet_id=0,
            capture_tick=0,
            arrive_tick=0,
            images=images,
            robot_state=robot_state,
            instruction=env.instruction,
            modality_capture_ticks={},
        )
        warmup_info = adapter.warmup(packet)

        # chunk shape + postprocessor broadcast verification
        import torch

        model_input = adapter._preprocess(packet)
        noise = adapter._make_noise(0)
        with torch.inference_mode():
            raw_chunk = adapter.policy.predict_action_chunk(model_input, noise=noise)
        chunk = adapter._postprocess_chunk(raw_chunk)
        single = adapter.postprocessor(raw_chunk[:, 0, :])
        single_array = (
            single["action"].detach().cpu().numpy()
            if isinstance(single, dict)
            else single.detach().cpu().numpy()
        )
        broadcast_ok = bool(
            np.allclose(chunk[0], single_array[0], atol=1e-9)
        )

        # noise reproducibility: same seed twice -> identical actions
        with torch.inference_mode():
            a = adapter.policy.predict_action_chunk(model_input, noise=adapter._make_noise(5))
            b = adapter.policy.predict_action_chunk(model_input, noise=adapter._make_noise(5))
        noise_reproducible = bool(torch.allclose(a, b))

        # per-stage timing with proper synchronization
        preprocess, forward, postprocess = [], [], []
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        for request_id in range(timing_calls):
            adapter.infer(packet, request_id=request_id, request_tick=0)
            stages = adapter.last_stage_seconds
            preprocess.append(stages["preprocess_s"])
            forward.append(stages["forward_s"])
            postprocess.append(stages["postprocess_s"])

        def _stats(values: list[float]) -> dict[str, float]:
            array = np.asarray(values)
            return {
                "mean_s": float(array.mean()),
                "p50_s": float(np.percentile(array, 50)),
                "p95_s": float(np.percentile(array, 95)),
            }

        total = [p + f + q for p, f, q in zip(preprocess, forward, postprocess, strict=True)]
        measured_inference_ticks = float(np.percentile(total, 50) / 0.05)
        return {
            "checkpoint": str(adapter.checkpoint),
            "chunk_size": adapter.chunk_size,
            "max_action_dim": adapter.max_action_dim,
            "n_action_steps_official": int(adapter.policy.config.n_action_steps),
            "raw_chunk_shape": list(raw_chunk.shape),
            "postprocessed_chunk_shape": list(chunk.shape),
            "postprocessor_chunk_broadcast_verified": broadcast_ok,
            "noise_seed_reproducible": noise_reproducible,
            "warmup": warmup_info,
            "timing_calls": timing_calls,
            "preprocess": _stats(preprocess),
            "forward": _stats(forward),
            "postprocess": _stats(postprocess),
            "total": _stats(total),
            "measured_inference_seconds_p50": float(np.percentile(total, 50)),
            "equivalent_control_ticks_p50": measured_inference_ticks,
            "note": (
                "wall-clock speed of the closed loop and simulated latency are "
                "separate quantities; this number only informs latency-level choice"
            ),
        }
    finally:
        env.close()


def run_audit(
    *,
    checkpoint: str | Path,
    tasks: tuple[str, ...] = ("goal_bowl_stove", "goal_bowl_plate"),
    output: str | Path = "runs/latency_audit/audit.json",
    device: str = "cuda",
    timing_calls: int = 30,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "created_unix": time.time(),
        "project_root": str(project_root()),
        "libero_assets": str(libero_assets_dir()),
        "versions": _versions(),
        "checkpoint": str(checkpoint),
    }
    if not Path(checkpoint).is_dir():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    report["env_probes"] = {}
    for task_key in tasks:
        report["env_probes"][task_key] = _env_probe(task_key, init_index=10, seed=52)
    report["model_probe"] = _model_probe(checkpoint, device=device, timing_calls=timing_calls)
    report["control_clock"] = {
        "control_hz": 20.0,
        "tick_seconds": 0.05,
        "physics": "robosuite OSC_POSE at control_freq=20; MuJoCo substeps stay inside robosuite",
        "max_episode_ticks": 300,
    }
    dump_json(report, output)
    return report
