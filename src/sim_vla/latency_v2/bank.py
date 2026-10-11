"""Policy-blind generation of new full physical states from the audited BDDL."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..assets import project_root
from ..tasks import TASK_BY_KEY
from .env import ControlledEnv
from .initializations import check_formal_manifest, state_digest
from .integrity import file_sha256
from .model_poses import ModelPoseBank, scene_digest
from .storage import Budget, canonical_hash, publish


def make_bank(directory: Path, *, count: int = 20) -> Path:
    root = project_root()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("preserve existing physical bank evidence; incomplete attempts need classified recovery")
    audit = json.loads((root/"runs/latency_v2_audit/audit.json").read_text())
    directory.mkdir(parents=True, exist_ok=True)
    budget = Budget(root/"runs/latency_v2_budget.json")
    manifest = {"format": "latency-v2-init-bank-v1", "tasks": {}, "provenance": {
        "source": "same installed LIBERO BDDL placement sampler; fresh OffScreenRenderEnv.reset physical arrays",
        "sampling_rule": "first 20 predetermined seeds per task; no policy, rollout score, or tuning",
        "base_seed": 20261107, "seed_stride_task": 10000,
        "historical_native_bank_reused": False, "training_overlap": "unknown; not a claim of training novelty",
        "distribution": "new simulator placement draws on the three supported tasks, not native init30-49",
        "settle_steps": 10, "physics": audit["versions"],
        "physical_scene_components": ["MuJoCo flattened state", "all model body_pos", "all model body_quat"],
        "fixture_placement_restoration": "restore frozen model transforms after reset, before injecting state",
        "restore_reset_seed_rule": "same sha256(task/index/42) seed as all paired formal methods",
        "libero_assets_sha256": audit["libero_assets"]["sha256"],
        "reset_probe_budget_rule": "each explicit sampling or restoration reset is one charged attempt; three per bank state"}}
    for task_number, task in enumerate(TASK_BY_KEY):
        env = ControlledEnv(task_key=task)
        try:
            underlying = env._env._env
            native = np.asarray(env._env._init_states, dtype=np.float64)
            nq = underlying.sim.model.nq
            arrays, validations, positions, quaternions = [], [], [], []
            names = None
            for index in range(count):
                seed = 20261107+task_number*10000+index
                attempt = budget.claim(f"bank:{task}:{index}", "S5_asset_validation")
                sample_dir = directory/"samples"/task/f"{index:02d}"
                publish(sample_dir/"attempt.json", {"attempt_number": attempt, "task": task,
                        "bank_index": index, "placement_seed": seed, "policy_used": False})
                underlying.seed(seed)
                underlying.reset()  # Genuine placement draw; NOT native-bank injection with a new RNG seed.
                state = np.asarray(underlying.get_sim_state(), dtype=np.float64).copy()
                if state.shape != (native.shape[1],) or not np.isfinite(state).all():
                    raise ValueError("invalid sampled MuJoCo state")
                np.save(sample_dir/"state.npy", state, allow_pickle=False)
                poses = ModelPoseBank.capture(underlying.sim.model)
                poses.save(sample_dir/"model_poses.npz")
                # Ignore time and velocities: prove the configuration itself differs.
                qpos = state[1:1+nq]
                prior = [row[1:1+nq] for row in native]+[row[1:1+nq] for row in arrays]
                minimum_difference = min(float(np.max(np.abs(qpos-row))) for row in prior)
                if minimum_difference <= 1e-8:
                    raise ValueError("sampled state configuration duplicates a prior state; stop before freezing")
                source_scene_digest = scene_digest(state, underlying.sim.model)
                env.model_pose_bank = poses
                names = poses.names
                restore_attempt = budget.claim(f"bank:{task}:{index}:restore", "S5_asset_validation")
                restore_seed = int(canonical_hash([task, index, 42])[:8], 16) % (2**31-1)
                publish(sample_dir/"restore_attempt.json", {"attempt_number": restore_attempt, "reset_seed": restore_seed})
                observation = env.start_episode(0, restore_seed, states=np.asarray([state]))
                restored = env.state_identity()
                first_images = {camera: value.copy() for camera, value in observation["pixels"].items()}
                np.save(sample_dir/"first_after_settle.npy", underlying.get_sim_state(), allow_pickle=False)
                # A different reset seed must not change the frozen physical scene.
                repeat_attempt = budget.claim(f"bank:{task}:{index}:repeat", "S5_asset_validation")
                repeat_seed = seed+1000000
                publish(sample_dir/"repeat_attempt.json", {"attempt_number": repeat_attempt, "reset_seed": repeat_seed})
                again = env.start_episode(0, repeat_seed, states=np.asarray([state]))
                repeat_digest = env.state_identity()
                np.save(sample_dir/"repeat_after_settle.npy", underlying.get_sim_state(), allow_pickle=False)
                max_image_difference = max(int(np.max(np.abs(first_images[camera].astype(int)
                                                              -again["pixels"][camera].astype(int))))
                                           for camera in first_images)
                np.savez_compressed(sample_dir/"reset_images.npz",
                                    **{f"first_{camera}": first_images[camera] for camera in first_images},
                                    **{f"repeat_{camera}": again["pixels"][camera] for camera in first_images})
                publish(sample_dir/"scene_repeat.json", {"first_after_settle_sha256": restored,
                        "repeat_after_settle_sha256": repeat_digest, "equal": restored == repeat_digest,
                        "image_repeat_max_difference": max_image_difference})
                if restored != repeat_digest:
                    raise ValueError("physical state restoration is not deterministic")
                if max_image_difference != 0 or underlying.check_success():
                    raise ValueError("invalid initial state: unstable images or goal already satisfied")
                arrays.append(state)
                positions.append(poses.positions[0])
                quaternions.append(poses.quaternions[0])
                validations.append({"index": index, "placement_seed": seed, "attempt_number": attempt,
                    "restore_attempt_number": restore_attempt, "repeat_attempt_number": repeat_attempt,
                    "restore_reset_seed": restore_seed, "repeat_reset_seed": repeat_seed,
                    "source_scene_sha256": source_scene_digest,
                    "model_fixture_poses_restored": True, "cross_seed_scene_repeat_passed": True,
                    "state_sha256": state_digest(state), "qpos_sha256": state_digest(qpos),
                    "min_qpos_linf_difference_from_prior": minimum_difference,
                    "reset_after_settle_sha256": restored, "image_repeat_max_difference": max_image_difference,
                    "initial_success": False})
                publish(sample_dir/"validation.json", validations[-1])
                print(json.dumps({"bank_task": task, "sample": index+1, "count": count}), flush=True)
            path = directory/(task+".npy")
            np.save(path, np.asarray(arrays), allow_pickle=False)
            pose_path = directory/(task+"_model_poses.npz")
            ModelPoseBank(names, np.asarray(positions), np.asarray(quaternions)).save(pose_path)
            publish(directory/(task+"_reset_validation.json"), validations)
            manifest["tasks"][task] = {"states_path": path.name, "states_sha256": file_sha256(path),
                "model_poses_path": pose_path.name, "model_poses_sha256": file_sha256(pose_path),
                "indices": list(range(count)), "native_task_id": TASK_BY_KEY[task].suite_task_id,
                "qpos_dimension": nq, "reset_validation_path": task+"_reset_validation.json",
                "reset_validation_sha256": file_sha256(directory/(task+"_reset_validation.json")),
                "compatibility_gate_passed": True}
        finally:
            env.close()
    manifest["manifest_content_sha256"] = canonical_hash(manifest)
    publish(directory/"manifest.json", manifest)
    checked = check_formal_manifest(directory/"manifest.json", audit["initializations"], required_per_task=count)
    publish(directory/"isolation_check.json", checked)
    return directory/"manifest.json"
