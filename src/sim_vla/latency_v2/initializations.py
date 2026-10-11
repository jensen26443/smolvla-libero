"""Audit physical initialization identities, including terminal implicit resets.

An episode number or a new noise seed is not a new physical initial state.
Historical official rollouts call reset at entry; LiberoEnv also calls reset
on termination. The latter advances the native-state cursor even when the
evaluator never starts a scored episode from that state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..tasks import TASK_BY_KEY
from .integrity import file_sha256


class InitializationIsolationError(ValueError):
    """The physical states cannot be certified as fresh formal assets."""


def state_digest(state: np.ndarray) -> str:
    array = np.asarray(state, dtype="<f8")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def official_reset_history(successes: list[bool], count: int, *, start: int = 0) -> dict:
    """Reconstruct the minimum resets implied by the installed official path.

    Success necessarily terminates and causes LiberoEnv.step's internal reset.
    For unsuccessful episodes, termination is unknown from success-only logs;
    this is a minimum, not invented exact historical telemetry. NEXT_STEP
    vector autoreset cannot add another reset inside the terminating step.
    """
    cursor = start
    scored, implicit = [], []
    for ordinal, success in enumerate(successes):
        scored.append({"episode_ordinal": ordinal, "init_index": cursor % count})
        cursor += 1
        if success:
            implicit.append({"episode_ordinal": ordinal, "init_index": cursor % count})
            cursor += 1
    seen = {item["init_index"] for item in scored + implicit}
    return {
        "scored_starts_reconstructed": scored,
        "implicit_terminal_resets_reconstructed": implicit,
        "exposed_indices_minimum": sorted(seen),
        "reset_calls_minimum": len(scored) + len(implicit),
        "assumptions": [
            "single environment, native cursor starts at zero",
            "official rollout explicitly resets at every episode entry",
            "LiberoEnv.step internally resets on every successful termination",
            "unsuccessful episodes assumed to reach external timeout (minimum exposure)",
            "current installed source matches the historical documented dependency versions",
        ],
        "evidence_kind": "source-and-success-log reconstruction, not recorded historical init telemetry",
    }


def audit_native_initializations(root: Path, task_keys: list[str]) -> dict:
    import importlib.util

    import torch

    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("the installed LIBERO initialization assets are absent")
    folder = Path(next(iter(spec.submodule_search_locations))) / "libero" / "init_files" / "libero_goal"
    history_path = root / "results/e1_formal/eval_info.json"
    historical = json.loads(history_path.read_text())
    by_id = {int(task["task_id"]): task for task in historical["per_task"]}
    report = {"history_source": str(history_path), "history_sha256": file_sha256(history_path), "tasks": {}}
    for task_key in task_keys:
        task = TASK_BY_KEY[task_key]
        path = folder / (task.task_name + ".pruned_init")
        states = np.asarray(torch.load(path, weights_only=False))
        if states.ndim != 2 or not np.isfinite(states).all():
            raise ValueError(f"invalid native initialization array at {path}: {states.shape}")
        hashes = [state_digest(row) for row in states]
        successes = by_id[task.suite_task_id]["metrics"]["successes"]
        history = official_reset_history(successes, len(states))
        # Explicit old latency-run initializations are independently recorded.
        latency_indices = set()
        for old_path in sorted((root / "runs").glob("latency_*/trials/*/result.json")):
            if "latency_v2" in str(old_path):
                continue
            old = json.loads(old_path.read_text())
            if old.get("task") == task_key:
                latency_indices.add(int(old["init_index"]))
        seen = set(history["exposed_indices_minimum"]) | latency_indices
        seen_hashes = {hashes[index] for index in seen if 0 <= index < len(hashes)}
        eligible = [i for i, digest in enumerate(hashes) if digest not in seen_hashes]
        report["tasks"][task_key] = {
            "task_id": task.suite_task_id,
            "dataset_task_index": task.dataset_task_index,
            "path": str(path),
            "file_sha256": file_sha256(path),
            "shape": list(states.shape),
            "row_sha256": hashes,
            "unique_states": len(set(hashes)),
            "historical_official": history,
            "historical_latency_scored_indices": sorted(latency_indices),
            "quarantined_indices": sorted(seen),
            "quarantined_state_sha256": sorted(seen_hashes),
            "eligible_native_indices": eligible,
            "preferred_30_49_conflicts": [i for i in range(30, min(50, len(states))) if hashes[i] in seen_hashes],
        }
    report["formal_asset_gate_passed"] = all(
        len(task["eligible_native_indices"]) >= 20 for task in report["tasks"].values()
    )
    report["training_overlap"] = "not audited; no claim of absence from model training data"
    return report


def validate_formal_states(entries: list[dict], audit: dict, *, required_per_task: int = 20) -> None:
    """Fail closed on physical-state reuse, duplicates, or insufficient coverage.

    Callers must load the actual state arrays and supply their digests; checking
    a manifest's index/seed alone would reproduce the historical leakage.
    """
    grouped: dict[str, list[str]] = {key: [] for key in audit["tasks"]}
    for entry in entries:
        key = entry["task"]
        if key not in grouped:
            raise InitializationIsolationError(f"unexpected task {key}")
        digest = entry["state_sha256"]
        if digest in audit["tasks"][key]["quarantined_state_sha256"]:
            raise InitializationIsolationError(f"{key}: physical state {digest} already exposed historically")
        grouped[key].append(digest)
    for key, hashes in grouped.items():
        if len(set(hashes)) != len(hashes):
            raise InitializationIsolationError(f"{key}: duplicated physical initialization")
        if len(hashes) < required_per_task:
            raise InitializationIsolationError(f"{key}: {len(hashes)} fresh states, need {required_per_task}")


def check_formal_manifest(manifest_path: Path, audit: dict, *, required_per_task: int = 20) -> dict:
    """Verify actual .npy bank bytes, then identities, rather than trusting labels.

    This gate establishes asset integrity and historical isolation only; a new
    bank additionally needs a task-compatible, documented source and a reset
    validity audit before any formal experiment specification can be frozen.
    """
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "latency-v2-init-bank-v1" or not manifest.get("provenance"):
        raise InitializationIsolationError("initialization bank requires format and provenance")
    if set(manifest["tasks"]) != set(audit["tasks"]):
        raise InitializationIsolationError("manifest must cover exactly the audited supported tasks")
    entries = []
    files = {}
    for task, bank in manifest["tasks"].items():
        path = Path(bank["states_path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        checksum = file_sha256(path)
        if checksum != bank["states_sha256"]:
            raise InitializationIsolationError(f"{task}: bank file SHA-256 mismatch")
        if path.suffix != ".npy":
            raise InitializationIsolationError("formal state banks must be non-pickled .npy arrays")
        array = np.load(path, allow_pickle=False)
        expected_dim = audit["tasks"][task]["shape"][1]
        if array.ndim != 2 or array.shape[1] != expected_dim or not np.isfinite(array).all():
            raise InitializationIsolationError(f"{task}: invalid physical-state array {array.shape}")
        for index in bank["indices"]:
            if not isinstance(index, int) or not 0 <= index < len(array):
                raise InitializationIsolationError(f"{task}: invalid index {index}, no modulo wrapping permitted")
            entries.append({"task": task, "index": index, "state_sha256": state_digest(array[index])})
        files[task] = {"path": str(path.resolve()), "sha256": checksum, "shape": list(array.shape)}
        if "model_poses_path" in bank:
            from .model_poses import ModelPoseBank

            pose_path = Path(bank["model_poses_path"])
            if not pose_path.is_absolute():
                pose_path = manifest_path.parent / pose_path
            if pose_path.suffix != ".npz" or file_sha256(pose_path) != bank["model_poses_sha256"]:
                raise InitializationIsolationError(f"{task}: model fixture pose asset SHA-256 mismatch")
            try:
                poses = ModelPoseBank.load(pose_path, expected_states=len(array))
            except ValueError as error:
                raise InitializationIsolationError(f"{task}: invalid model fixture pose asset: {error}") from error
            files[task].update({"model_poses_path": str(pose_path.resolve()),
                               "model_poses_sha256": bank["model_poses_sha256"],
                               "model_body_count": len(poses.names)})
    validate_formal_states(entries, audit, required_per_task=required_per_task)
    return {"historical_isolation_passed": True, "files": files, "entries": entries,
            "remaining_gate": "task compatibility, documented sampling provenance and reset validity"}
