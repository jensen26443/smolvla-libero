"""Policy-blind placement draws: one charged attempt per reset, six restore checks."""
import numpy as np

from ..latency_v2.integrity import file_sha256
from ..latency_v2.locking import run_lock
from ..latency_v2.model_poses import ModelPoseBank, scene_digest
from ..latency_v2.schema import digest_arrays
from ..latency_v2.storage import canonical_hash, publish
from ..tasks import TASK_BY_KEY
from .continuation import formal_eligibility
from .layout import BANK, ROOT, RUN, RUN_ID, budget, immutable_json, physical_assets, read


def configuration_digest(state, positions, quaternions, nq):
    return digest_arrays({"qpos": np.asarray(state[1:1+nq], dtype="<f8"),
        "body_pos": np.asarray(positions, dtype="<f8"), "body_quat": np.asarray(quaternions, dtype="<f8")})


def historical_configurations(task, nq):
    manifest_path = ROOT/"runs/latency_v2_physical_bank/manifest.json"
    entry = read(manifest_path)["tasks"][task]
    states = np.load(manifest_path.parent/entry["states_path"], allow_pickle=False)
    poses = ModelPoseBank.load(manifest_path.parent/entry["model_poses_path"], expected_states=len(states))
    return {configuration_digest(state, poses.positions[index], poses.quaternions[index], nq)
            for index, state in enumerate(states)}


def prepare_bank():
    formal_eligibility(read(RUN/"development_gate.json"), run=RUN, run_id=RUN_ID)
    if BANK.exists():
        manifest = verify_bank()
        restore_checks()
        return {"manifest_sha256": file_sha256(BANK), "states": sum(len(entry["states"]) for entry in manifest["tasks"].values())}
    from ..latency_v2.env import ControlledEnv

    directory = BANK.parent
    with run_lock(directory):
        plan = {"base_seed": 20261005, "task_seed_stride": 10000, "target_per_task": 20,
            "order": "first 20 valid draws in ascending predetermined seed order",
            "validity": ["finite full state and normalized model transforms", "not initially successful",
                         "no exact duplicate full configuration including fixtures"],
            "forbidden_selection": "no policy calls, no difficulty judgment or success-score selection"}
        immutable_json(directory/"generation_plan.json", plan)
        history = historical_scene_inventory()
        immutable_json(directory/"historical_scene_inventory.json", history)
        historical_scenes = {item["scene_sha256"] for item in history}
        manifest = {"format": "latency-v4-physical-bank-v1", "generation_plan": plan,
            "generation_plan_sha256": file_sha256(directory/"generation_plan.json"), "tasks": {},
            "training_overlap": "unknown; fresh simulator draws do not imply training novelty"}
        b = budget()
        for task_number, task in enumerate(TASK_BY_KEY):
            accepted, configurations = [], set()
            env = None
            candidate_index = 0
            try:
                while len(accepted) < 20:
                    seed = plan["base_seed"]+task_number*plan["task_seed_stride"]+candidate_index
                    target = directory/"samples"/task/f"{candidate_index:03d}"
                    if (target/"result.json").exists():
                        result = read(target/"result.json")
                    else:
                        # Always preserve enough attempts for remaining placements,
                        # six restorations and the complete formal matrix.
                        selected_steps = read(RUN/"selection.json")["sampling_steps"]
                        formal_count = 240+72+(30 if selected_steps == 10 else 60)
                        remaining_draws = 20-len(accepted)+(2-task_number)*20
                        if b.remaining() < remaining_draws+6+formal_count:
                            raise RuntimeError("physical generation would consume budget needed by the frozen matrix")
                        b.prepare_retry(target)
                        current = {"task": task, "placement_seed": seed, "candidate_index": candidate_index,
                                   "probe": "new_physical_state_no_policy"}
                        with b.track(f"bank:{task}:{seed}", "prepare_bank", target, current, run_id=RUN_ID):
                            if env is None:
                                env = ControlledEnv(task_key=task, observation_mode="on_request")
                            underlying = env._env._env
                            nq = underlying.sim.model.nq
                            prior = historical_configurations(task, nq)
                            underlying.seed(seed)
                            underlying.reset()
                            state = np.asarray(underlying.get_sim_state(), dtype=np.float64).copy()
                            poses = ModelPoseBank.capture(underlying.sim.model)
                            poses.validate(1)
                            identity = configuration_digest(state, poses.positions[0], poses.quaternions[0], nq)
                            reasons = []
                            if state.shape != (env._env._init_states.shape[1],) or not np.isfinite(state).all():
                                reasons.append("invalid_state")
                            source_scene = scene_digest(state, underlying.sim.model)
                            initial_success = bool(underlying.check_success())
                            # Settle within this same charged placement attempt;
                            # no additional reset or policy invocation.
                            from lerobot.envs.libero import get_libero_dummy_action

                            for _ in range(env._env.num_steps_wait):
                                underlying.step(get_libero_dummy_action())
                            settled_scene = env.state_identity()
                            if initial_success or underlying.check_success():
                                reasons.append("initially_successful")
                            if identity in prior or identity in configurations or settled_scene in historical_scenes:
                                reasons.append("exact_prior_configuration_duplicate")
                            np.save(target/"state.npy", state, allow_pickle=False)
                            poses.save(target/"model_poses.npz")
                            result = {**current, "accepted": not reasons, "rejection_reasons": reasons,
                                "scene_sha256": source_scene, "after_settle_scene_sha256": settled_scene,
                                "configuration_sha256": identity, "policy_used": False,
                                "state_sha256": file_sha256(target/"state.npy"),
                                "model_poses_sha256": file_sha256(target/"model_poses.npz"), "qpos_dimension": nq}
                            publish(target/"result.json", result)
                    if result["accepted"]:
                        configurations.add(result["configuration_sha256"])
                        accepted.append({**result, "sample_path": str(target.relative_to(directory)), "index": len(accepted)})
                    candidate_index += 1
                    print(f"bank {task}: {len(accepted)}/20 accepted from {candidate_index} charged draws", flush=True)
                states, positions, quaternions = [], [], []
                names = None
                for item in accepted:
                    sample = directory/item["sample_path"]
                    if (file_sha256(sample/"state.npy") != item["state_sha256"]
                            or file_sha256(sample/"model_poses.npz") != item["model_poses_sha256"]):
                        raise ValueError("saved placement evidence changed before bank assembly")
                    states.append(np.load(sample/"state.npy", allow_pickle=False))
                    pose = ModelPoseBank.load(sample/"model_poses.npz", expected_states=1)
                    names = pose.names
                    positions.append(pose.positions[0])
                    quaternions.append(pose.quaternions[0])
                state_path, pose_path = directory/(task+".npy"), directory/(task+"_model_poses.npz")
                np.save(state_path, np.asarray(states), allow_pickle=False)
                ModelPoseBank(names, np.asarray(positions), np.asarray(quaternions)).save(pose_path)
                manifest["tasks"][task] = {"states": accepted, "states_path": state_path.name,
                    "states_sha256": file_sha256(state_path), "model_poses_path": pose_path.name,
                    "model_poses_sha256": file_sha256(pose_path)}
            finally:
                if env is not None:
                    env.close()
        manifest["manifest_content_sha256"] = canonical_hash(manifest)
        immutable_json(BANK, manifest)
    restore_checks()
    return {"states": 60, "manifest_sha256": file_sha256(BANK)}


def historical_scene_inventory():
    """Recorded full scenes include fixtures; qpos alone is not a duplicate."""
    sources = []
    for pattern in ("latency_v2*", "latency_v3*"):
        for directory in (ROOT/"runs").glob(pattern):
            sources.extend(directory.rglob("result.json"))
    for stage in ("develop_sampling", "develop_period", "sustain"):
        sources.extend((RUN/stage).rglob("result.json"))
    output = []
    for path in sorted(set(sources)):
        result = read(path)
        digest = result.get("initial_state_after_settle_sha256")
        if digest:
            output.append({"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path), "scene_sha256": digest})
    return output


def verify_bank():
    manifest = read(BANK)
    if (manifest.get("format") != "latency-v4-physical-bank-v1" or set(manifest["tasks"]) != set(TASK_BY_KEY)
            or canonical_hash({key: value for key, value in manifest.items() if key != "manifest_content_sha256"})
            != manifest["manifest_content_sha256"]):
        raise ValueError("physical bank manifest changed or incomplete")
    if file_sha256(BANK.parent/"generation_plan.json") != manifest["generation_plan_sha256"]:
        raise ValueError("physical generation protocol changed")
    states, poses = physical_assets()
    for task, entry in manifest["tasks"].items():
        if len(entry["states"]) != 20 or len({item["configuration_sha256"] for item in entry["states"]}) != 20:
            raise ValueError("physical bank must contain 20 distinct states per task")
        for index, item in enumerate(entry["states"]):
            if item["index"] != index or not item["accepted"] or item["policy_used"]:
                raise ValueError("physical state order or eligibility changed")
            actual = digest_arrays({"state": np.asarray(states[task][index], dtype="<f8"),
                "body_pos": np.asarray(poses[task].positions[index], dtype="<f8"),
                "body_quat": np.asarray(poses[task].quaternions[index], dtype="<f8")})
            if actual != item["scene_sha256"]:
                raise ValueError("physical scene identity differs from its generation evidence")
    return manifest


def restore_checks():
    from .physical import VerifiedPhysicalEnv

    target_complete = RUN/"physical_restore/complete.json"
    if target_complete.exists():
        return read(target_complete)
    states, poses = physical_assets()
    manifest = read(BANK)
    b, checks = budget(), []
    for task in TASK_BY_KEY:
        for seed in (42, 20261005):
            target = RUN/"physical_restore/trials"/f"{task}_{seed}"
            if (target/"result.json").exists():
                checks.append(read(target/"result.json"))
                continue
            current = {"task": task, "init_index": 0, "seed": seed, "probe": "physical_restore_no_policy"}
            b.prepare_retry(target)
            with b.track(target.name, "physical_restore", target, current, run_id=RUN_ID):
                env = VerifiedPhysicalEnv(task_key=task, observation_mode="on_request",
                    expected_scenes=[item["after_settle_scene_sha256"] for item in manifest["tasks"][task]["states"]])
                env.model_pose_bank = poses[task]
                try:
                    env.start_episode(0, seed, states=states[task])
                    observation = env.sample_observation(0)
                    result = {**current, "scene_sha256": env.state_identity(),
                              "images_sha256": digest_arrays(observation["pixels"]),
                              "initial_success": bool(env._env._env.check_success())}
                    if result["initial_success"]:
                        raise ValueError("restored state already meets the goal")
                    publish(target/"result.json", result)
                    checks.append(result)
                finally:
                    env.close()
    for task in TASK_BY_KEY:
        subset = [item for item in checks if item["task"] == task]
        if len({(item["scene_sha256"], item["images_sha256"]) for item in subset}) != 1:
            raise ValueError("restored scene or camera images differ across reset seeds")
    result = {"passed": True, "attempts": 6, "checks": checks, "policy_used": False}
    immutable_json(target_complete, result)
    return result
