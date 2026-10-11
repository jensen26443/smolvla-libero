"""Auditable E3/E4 language-pair alignment checks.

The three selected LIBERO Goal tasks use the same tabletop asset family, but
language-pair experiments are only valid when the visual scene, the selected
initial-state asset, and the runtime observations line up.  This module keeps
that decision separate from policy evaluation: it never changes a BDDL file,
never changes a success predicate, and never reports a fabricated language
success rate.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .assets import libero_assets_dir, project_root
from .protocol import PROTOCOL_VERSION, SEED, dump_json
from .runtime import configure_libero, import_libero
from .tasks import (
    TaskSpec,
    bddl_scene_signature,
    bddl_visual_scene_signature,
    selected_tasks,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    result: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": _sha256_bytes(contiguous.tobytes()),
    }
    if array.size and np.issubdtype(array.dtype, np.number):
        result.update(
            {
                "finite": bool(np.isfinite(array).all()),
                "min": float(array.min()),
                "max": float(array.max()),
                "mean": float(array.mean()),
            }
        )
    return result


def _nested_array_digests(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _nested_array_digests(item) for key, item in value.items()}
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return value
    return _array_digest(value)


def _flatten_arrays(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(value, Mapping):
        flattened: dict[str, np.ndarray] = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_arrays(item, name))
        return flattened
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return {}
    return {prefix: np.asarray(value)}


def _array_difference(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"shape_match": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    if left.dtype.kind not in "biufc" or right.dtype.kind not in "biufc":
        return {"shape_match": True, "exact": bool(np.array_equal(left, right))}
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {
        "shape_match": True,
        "exact": bool(np.array_equal(left, right)),
        "max_abs": float(delta.max()) if delta.size else 0.0,
        "mean_abs": float(delta.mean()) if delta.size else 0.0,
    }


def _load_init_states(path: Path) -> np.ndarray:
    """Load a LIBERO ``*.pruned_init`` array without importing torch globally."""

    import torch

    value = torch.load(path, weights_only=False)  # nosec B614 - pinned local LIBERO asset
    array = np.asarray(value)
    if array.ndim < 2 or array.shape[0] <= 0:
        raise ValueError(f"unexpected init-state shape at {path}: {array.shape}")
    return array


def _init_state_summary(path: Path) -> dict[str, Any]:
    states = _load_init_states(path)
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "shape": list(states.shape),
        "dtype": str(states.dtype),
        "first_state": _array_digest(states[0]),
        "state_count": int(states.shape[0]),
    }


def _compare_init_states(left: Path, right: Path) -> dict[str, Any]:
    first = _load_init_states(left)
    second = _load_init_states(right)
    result: dict[str, Any] = {
        "shape_match": bool(first.shape == second.shape),
        "exact_match": bool(np.array_equal(first, second)),
        "left_sha256": file_sha256(left),
        "right_sha256": file_sha256(right),
    }
    if first.shape == second.shape:
        delta = np.abs(first.astype(np.float64) - second.astype(np.float64))
        result.update(
            {
                "max_abs": float(delta.max()) if delta.size else 0.0,
                "mean_abs": float(delta.mean()) if delta.size else 0.0,
            }
        )
    return result


def _runtime_probe(task_spec: TaskSpec, suite: Any, *, seed: int) -> dict[str, Any]:
    """Capture one official-style reset for one task.

    The probe uses the same observation dimensions, control mode, settling
    steps, and fixed initial-state index convention as the LeRobot evaluator.
    A failure is recorded in the artifact and does not silently turn into a
    passing alignment result.
    """

    from lerobot.envs.libero import LiberoEnv

    env = None
    try:
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
        observation, info = env.reset(seed=seed)
        sim_state = env._env.get_sim_state()
        return {
            "status": "passed",
            "task_key": task_spec.key,
            "suite_task_id": task_spec.suite_task_id,
            "seed": seed,
            "initial_state_index": 0,
            "info": {str(key): bool(value) if isinstance(value, (bool, np.bool_)) else value for key, value in info.items()},
            "observation": _nested_array_digests(observation),
            "sim_state": _array_digest(sim_state),
        }
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - depends on local MuJoCo/EGL
        return {
            "status": "error",
            "task_key": task_spec.key,
            "suite_task_id": task_spec.suite_task_id,
            "seed": seed,
            "initial_state_index": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if env is not None:
            env.close()


def _compare_runtime_probes(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if left.get("status") != "passed" or right.get("status") != "passed":
        return {
            "status": "not_comparable",
            "left_status": left.get("status"),
            "right_status": right.get("status"),
        }
    left_arrays = _flatten_arrays(left.get("observation", {}))
    right_arrays = _flatten_arrays(right.get("observation", {}))
    # The serialized observation contains summaries, so compare the digest
    # leaves directly and retain numeric differences for the actual simulator
    # state captured below.
    left_obs = left.get("observation", {})
    right_obs = right.get("observation", {})
    matching = left_obs == right_obs and left.get("sim_state", {}).get("sha256") == right.get("sim_state", {}).get("sha256")
    result: dict[str, Any] = {
        "status": "passed",
        "exact_observation_match": bool(left_obs == right_obs),
        "exact_sim_state_match": bool(
            left.get("sim_state", {}).get("sha256") == right.get("sim_state", {}).get("sha256")
        ),
        "exact_match": bool(matching),
        "observation_fields_left": sorted(left_arrays),
        "observation_fields_right": sorted(right_arrays),
    }
    if left.get("sim_state", {}).get("shape") == right.get("sim_state", {}).get("shape"):
        # Runtime summaries intentionally omit raw values; exact digests are
        # the required criterion.  This branch documents that the shapes are
        # compatible without manufacturing a numeric difference.
        result["sim_state_shape"] = left["sim_state"]["shape"]
    return result


def _task_files(task_spec: TaskSpec, suite: Any, get_libero_path: Any) -> tuple[Path, Path, Any]:
    task = suite.get_task(task_spec.suite_task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_states = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return bddl, init_states, task


def run_language_alignment_audit(
    *,
    output: str | Path = "results/language_alignment_audit.json",
    task_keys: Sequence[str] | None = None,
    seed: int = SEED,
    assets: str | Path | None = None,
    runtime: bool = True,
) -> dict[str, Any]:
    """Run the static and (optionally) runtime E3/E4 alignment gate."""

    selected = selected_tasks(task_keys)
    asset_path = Path(assets) if assets is not None else libero_assets_dir()
    if runtime and not asset_path.is_dir():
        raise FileNotFoundError(
            f"LIBERO assets not found at {asset_path}; run `sim-vla prepare-assets --kind libero` first"
        )
    started = time.time()
    config_file = configure_libero(root=project_root(), assets=asset_path)
    import_libero(root=project_root(), assets=asset_path)
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[selected[0].suite]()
    files: dict[str, dict[str, Any]] = {}
    runtime_records: dict[str, dict[str, Any]] = {}
    for task_spec in selected:
        bddl, init_states, task = _task_files(task_spec, suite, get_libero_path)
        files[task_spec.key] = {
            "suite_task_id": task_spec.suite_task_id,
            "task_name": task.name,
            "official_language": task.language,
            "bddl": str(bddl),
            "bddl_sha256": file_sha256(bddl),
            "init_states": _init_state_summary(init_states),
        }
        if runtime:
            runtime_records[task_spec.key] = _runtime_probe(task_spec, suite, seed=seed)

    pair_records: list[dict[str, Any]] = []
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            left_files = files[left.key]
            right_files = files[right.key]
            left_bddl = Path(left_files["bddl"])
            right_bddl = Path(right_files["bddl"])
            static_signature = bddl_scene_signature([left_bddl, right_bddl])
            visual_signature = bddl_visual_scene_signature([left_bddl, right_bddl])
            init_match = _compare_init_states(
                Path(left_files["init_states"]["path"]), Path(right_files["init_states"]["path"])
            )
            runtime_match = None
            if runtime:
                runtime_match = _compare_runtime_probes(runtime_records[left.key], runtime_records[right.key])
            blocking: list[str] = []
            if not visual_signature["matching"]:
                blocking.append("visual_scene_declarations_differ")
            if not init_match["exact_match"]:
                blocking.append("initial_state_assets_differ")
            if runtime and (runtime_match is None or runtime_match.get("status") != "passed"):
                blocking.append("runtime_probe_not_comparable")
            elif runtime and not runtime_match.get("exact_match", False):
                blocking.append("runtime_observations_or_sim_state_differ")
            pair_records.append(
                {
                    "left": left.key,
                    "right": right.key,
                    "bddl_scene_signature_without_goal": static_signature,
                    "visual_scene_signature_without_language_target_goal": visual_signature,
                    "initial_state_comparison": init_match,
                    "runtime_comparison": runtime_match,
                    "eligible_for_quantitative_pair": not blocking,
                    "blocking_reasons": blocking,
                }
            )

    eligible = bool(pair_records) and all(pair["eligible_for_quantitative_pair"] for pair in pair_records)
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "audit_version": "sim-vla-language-alignment-v1",
        "captured_unix": started,
        "elapsed_seconds": time.time() - started,
        "suite": selected[0].suite,
        "task_keys": [task.key for task in selected],
        "seed": seed,
        "runtime_probe_requested": runtime,
        "runtime_probe_observation_contract": {
            "obs_type": "pixels_agent_pos",
            "width": 256,
            "height": 256,
            "control_mode": "relative",
            "episode_index": 0,
            "num_steps_wait": 10,
        },
        "libero_config": str(config_file),
        "assets_path": str(asset_path),
        "files": files,
        "runtime": runtime_records,
        "pairs": pair_records,
        "status": "quantitative_pair_eligible" if eligible else "no_quantitative_pair_frozen",
        "decision": (
            "E3/E4 may use the paired quantitative protocol"
            if eligible
            else "E3/E4 remain qualitative only; do not report cross-task success rates"
        ),
    }
    dump_json(result, output)
    return result


def language_alignment_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="audit LIBERO E3/E4 language-pair alignment")
    parser.add_argument("--output", default="results/language_alignment_audit.json")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--assets")
    parser.add_argument("--no-runtime", action="store_true")
    args = parser.parse_args(argv)
    result = run_language_alignment_audit(
        output=args.output,
        seed=args.seed,
        assets=args.assets,
        runtime=not args.no_runtime,
    )
    print(f"language alignment audit: {result['status']}; output={args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(language_alignment_main())
