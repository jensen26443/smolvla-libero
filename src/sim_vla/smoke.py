"""Small, auditable LIBERO reset/step/render compatibility check."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .assets import libero_assets_dir, project_root
from .protocol import SEED, dump_json
from .runtime import import_libero
from .tasks import TASKS


def _array_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    finite = bool(np.isfinite(array).all()) if np.issubdtype(array.dtype, np.number) else None
    result: dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype), "finite": finite}
    if array.size and np.issubdtype(array.dtype, np.number):
        result.update({"min": float(array.min()), "max": float(array.max()), "mean": float(array.mean())})
    return result


def _frame_digest(frame: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def _batch_nested_state(value: Any) -> Any:
    """Add the vector-env batch axis missing from a direct ``LiberoEnv`` call."""

    if isinstance(value, dict):
        return {key: _batch_nested_state(item) for key, item in value.items()}
    if hasattr(value, "unsqueeze") and getattr(value, "ndim", 0) >= 1:
        return value.unsqueeze(0)
    return value


def _write_video(frames: list[np.ndarray], path: Path, fps: int = 10) -> str | None:
    if not frames:
        return "no frames"
    try:
        import imageio.v2 as imageio
    except ImportError:
        return "imageio is not installed"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame, dtype=np.uint8))
    except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - codec is host-specific
        return f"video writer failed: {type(exc).__name__}: {exc}"
    return None


def _run_task(task_spec, *, resets: int, steps: int, video_dir: Path) -> dict[str, Any]:
    # Imports are intentionally delayed until import_libero() has installed the
    # project-local config, otherwise hf-libero prompts on first import.
    from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[task_spec.suite]()
    task = suite.get_task(task_spec.suite_task_id)
    env = LiberoEnv(
        task_suite=suite,
        task_id=task_spec.suite_task_id,
        task_suite_name=task_spec.suite,
        episode_index=0,
        n_envs=1,
        obs_type="pixels_agent_pos",
        observation_width=256,
        observation_height=256,
        control_mode="relative",
    )
    processor = LiberoProcessorStep()
    reset_records: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    processor_summary: dict[str, Any] | None = None
    video_error: str | None = None
    try:
        for reset_index in range(resets):
            observation, info = env.reset(seed=SEED + reset_index)
            pixels = observation["pixels"]
            reset_records.append(
                {
                    "reset_index": reset_index,
                    "info": {str(key): bool(value) if isinstance(value, (bool, np.bool_)) else value for key, value in info.items()},
                    "pixels": {name: _array_summary(value) for name, value in pixels.items()},
                    "robot_state": {
                        "eef_pos": _array_summary(observation["robot_state"]["eef"]["pos"]),
                        "eef_quat": _array_summary(observation["robot_state"]["eef"]["quat"]),
                        "gripper_qpos": _array_summary(observation["robot_state"]["gripper"]["qpos"]),
                    },
                }
            )

            processed = preprocess_observation(observation)
            processed["observation.robot_state"] = _batch_nested_state(processed["observation.robot_state"])
            processed["task"] = task.language
            flattened = processor.observation(processed)
            state = flattened["observation.state"]
            processor_summary = {
                "keys": sorted(flattened),
                "state": _array_summary(state.cpu().numpy()),
                "language": task.language,
            }
            if tuple(state.shape) != (1, 8):
                raise AssertionError(f"LIBERO processor returned state shape {tuple(state.shape)}, expected (1, 8)")

            if reset_index == 0:
                frames.append(env.render())
            action = np.asarray(get_libero_dummy_action(), dtype=np.float32)
            for step_index in range(steps):
                next_observation, reward, terminated, truncated, step_info = env.step(action)
                step_records.append(
                    {
                        "reset_index": reset_index,
                        "step_index": step_index,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "is_success": bool(step_info.get("is_success", False)),
                        "action": action.tolist(),
                    }
                )
                if reset_index == 0:
                    frames.append(env.render())
                # Keep the observation variable live so a future smoke check
                # can extend this loop without changing the action contract.
                observation = next_observation
    finally:
        env.close()

    video_path = video_dir / f"{task_spec.key}.mp4"
    video_error = _write_video(frames, video_path)
    frame_summary = _array_summary(frames[0]) if frames else None
    return {
        "key": task_spec.key,
        "suite": task_spec.suite,
        "suite_task_id": task_spec.suite_task_id,
        "dataset_task_index": task_spec.dataset_task_index,
        "task_name": task.name,
        "language": task.language,
        "bddl": str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
        "action_space": {"shape": list(env.action_space.shape), "low": env.action_space.low.tolist(), "high": env.action_space.high.tolist()},
        "resets": reset_records,
        "steps": step_records,
        "processor": processor_summary,
        "render": {"frame": frame_summary, "first_frame_sha256": _frame_digest(frames[0]) if frames else None},
        "video": {"path": str(video_path), "frames": len(frames), "error": video_error},
    }


def run_libero_smoke(
    *,
    output: str | Path = "runs/libero_smoke.json",
    video_dir: str | Path = "runs/libero_smoke",
    task_keys: tuple[str, ...] | None = None,
    resets: int = 5,
    steps: int = 1,
    assets: str | Path | None = None,
) -> dict[str, Any]:
    """Run the fixed task reset/step/render contract and write a JSON record."""

    if resets <= 0 or steps <= 0:
        raise ValueError("resets and steps must be positive")
    wanted = set(task_keys or (task.key for task in TASKS))
    selected = [task for task in TASKS if task.key in wanted]
    if len(selected) != len(wanted):
        raise ValueError(f"unknown task key(s): {sorted(wanted - {task.key for task in TASKS})}")
    asset_path = Path(assets) if assets is not None else libero_assets_dir()
    if not asset_path.is_dir():
        raise FileNotFoundError(
            f"LIBERO assets not found at {asset_path}; run `sim-vla prepare-assets --kind libero` first"
        )
    config_file = import_libero(root=project_root(), assets=asset_path)
    records = []
    started = time.time()
    for task in selected:
        records.append(_run_task(task, resets=resets, steps=steps, video_dir=Path(video_dir)))
    result = {
        "status": "passed",
        "protocol_version": "sim-vla-libero-v1",
        "captured_unix": started,
        "elapsed_seconds": time.time() - started,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        "libero_config": str(config_file),
        "assets_path": str(asset_path),
        "resets_per_task": resets,
        "steps_per_reset": steps,
        "tasks": records,
    }
    dump_json(result, output)
    return result


def smoke_libero_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run the fixed LIBERO reset/step/render smoke")
    parser.add_argument("--output", default="runs/libero_smoke.json")
    parser.add_argument("--video-dir", default="runs/libero_smoke")
    parser.add_argument("--resets", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--assets")
    args = parser.parse_args(argv)
    run_libero_smoke(
        output=args.output,
        video_dir=args.video_dir,
        resets=args.resets,
        steps=args.steps,
        assets=args.assets,
    )
    print(f"LIBERO smoke passed; record written to {args.output}")
    return 0
