"""Real snapshot probes, counted resets and physical-state exposure evidence."""
import pickle
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..latency_v2.env import ControlledEnv
from ..latency_v2.integrity import file_sha256, inventory
from ..latency_v2.schema import CommittedPrefix, ObservationSnapshot, digest_arrays, frozen_copy
from ..latency_v2.storage import publish
from .experiments import (
    BANK,
    CHECKPOINT,
    DOCS,
    MODEL_SHA,
    ROOT,
    RUN,
    RUN_ID,
    budget,
    physical_assets,
    read,
    source_inventory,
)
from .policy import ResearchPolicyAdapter
from .schema import ResearchRequest


def exposure_inventory():
    manifest = read(BANK)
    b = read(ROOT/"runs/latency_v2_budget.json")
    exposed = {task: set() for task in manifest["tasks"]}
    evidence = []
    sources = set()
    for attempt in b["attempts"]:
        target_text = attempt.get("target")
        if not target_text:
            continue
        target = Path(target_text)
        sources.add((target/"attempt.json", target.parent.parent/"spec.json"))
    # Earlier ledger entries lack target paths; their on-disk attempts still
    # establish policy exposure and must be included independently.
    for directory in [*(ROOT/"runs").glob("latency_v2*"), *(ROOT/"runs").glob("latency_v3*"), RUN]:
        for sp in directory.rglob("spec.json"):
            sources.update((ap, sp) for ap in (sp.parent/"trials").glob("*/attempt.json"))
    for ap, sp in sorted(sources):
        if not ap.exists() or not sp.exists():
            continue
        spec = read(sp)
        conf = spec.get("config", {})
        bank_path = conf.get("initialization_manifest") or spec.get("physical_bank_manifest")
        if not bank_path:
            continue
        bank_path = Path(bank_path)
        if not bank_path.is_absolute():
            bank_path = ROOT/bank_path
        if bank_path.resolve() != BANK.resolve() and file_sha256(bank_path) != file_sha256(BANK):
            continue
        trial = read(ap).get("trial", {})
        if trial.get("probe") == "physical_restore_no_policy":
            continue
        if trial.get("task") in exposed:
            exposed[trial["task"]].add(trial["init_index"])
            evidence.append({"attempt": str(ap.relative_to(ROOT)), "spec_sha256": file_sha256(sp),
                             "task": trial["task"], "index": trial["init_index"]})
    states, _ = physical_assets()
    eligibility = {}
    entries = []
    for task, entry in manifest["tasks"].items():
        # Existing reset validity covers all twenty physical placements.
        validity = BANK.parent/entry["reset_validation_path"]
        if file_sha256(validity) != entry["reset_validation_sha256"] or not entry["compatibility_gate_passed"]:
            raise ValueError("physical bank validation provenance changed")
        eligible = []
        for idx in entry["indices"]:
            usable = idx >= 10 and idx not in exposed[task]
            if usable:
                eligible.append(idx)
            entries.append({"task": task, "index": idx, "state_sha256": digest_arrays(states[task][idx]),
                "state_file_sha256": entry["states_sha256"], "model_poses_sha256": entry["model_poses_sha256"],
                "policy_exposed": idx in exposed[task], "eligible": usable})
        eligibility[task] = eligible
    result = {"physical_manifest": str(BANK.relative_to(ROOT)), "physical_manifest_sha256": file_sha256(BANK),
        "exposed_indices": {k: sorted(v) for k, v in exposed.items()}, "eligible_indices": eligibility,
        "attempt_sources": evidence, "entries": entries,
        "training_overlap": "unknown", "claim": "unused for policy evaluation in recorded physical-bank runs"}
    publish(RUN/"audit/physical_eligibility.json", result)
    return result


def verify_execution_assets():
    """Rehash the complete pinned assets, with no model or simulator calls."""
    import importlib.metadata
    import platform

    from ..assets import libero_assets_dir
    from ..tasks import TASK_BY_KEY

    old = read(ROOT/"runs/latency_v2_audit/audit.json")
    supplemental = read(ROOT/"runs/latency_v2_audit/supplemental_provenance.json")
    expected_checkpoint = read(RUN/"audit/audit.json")["checkpoint_inventory"]
    checkpoint = inventory(ROOT/CHECKPOINT, (ROOT/CHECKPOINT).rglob("*"))
    assets_root = libero_assets_dir()
    assets = inventory(assets_root, (assets_root/e["path"] for e in old["libero_assets"]["files"]))
    vlm_root = Path(supplemental["snapshot"])
    vlm = inventory(vlm_root, (vlm_root/e["path"] for e in supplemental["inventory"]["files"]))
    if checkpoint != expected_checkpoint or assets != old["libero_assets"] or vlm != supplemental["inventory"]:
        raise ValueError("pinned checkpoint, processor, tokenizer or LIBERO asset content changed")
    upstream = {}
    for name, entry in old["upstream_source"].items():
        digest = file_sha256(Path(entry["path"]))
        if digest != entry["sha256"]:
            raise ValueError(f"upstream control code changed: {name}")
        upstream[name] = entry
    task_root = ROOT/".venv/lib/python3.12/site-packages/libero/libero"
    task_files = [task_root/subdir/task.suite/(task.task_name+extension)
                  for task in TASK_BY_KEY.values()
                  for subdir, extension in (("bddl_files", ".bddl"), ("init_files", ".pruned_init"))]
    result = {"checkpoint": checkpoint, "libero_assets": assets, "vlm_processor": vlm,
        "vlm_snapshot": str(vlm_root), "upstream_control": upstream,
        "task_files": inventory(task_root, task_files),
        "versions": {name: importlib.metadata.version(name) for name in (
            "lerobot", "torch", "transformers", "hf-libero", "mujoco", "gymnasium")},
        "python": platform.python_version()}
    publish(RUN/"audit/execution_assets.json", result)
    return result


def audit():
    import torch

    destination = RUN/"audit"
    if (destination/"complete.json").exists():
        return read(destination/"audit.json")
    destination.mkdir(parents=True, exist_ok=True)
    if file_sha256(ROOT/CHECKPOINT/"model.safetensors") != MODEL_SHA:
        raise ValueError("E1 weights no longer match")
    old_files = {str(p.relative_to(ROOT)): file_sha256(p) for p in (ROOT/"src/sim_vla/latency_v2").glob("*.py")}
    publish(destination/"historical_source_inventory.json", old_files)
    exposure_inventory()
    adapter = ResearchPolicyAdapter(checkpoint=ROOT/CHECKPOINT, warmup_calls=3)
    adapter.warmup()
    snapshots = []
    b = budget()
    from ..tasks import TASK_BY_KEY

    for task in TASK_BY_KEY:
        target = destination/"compatibility"/task
        if (target/"result.json").exists():
            with (target/"snapshot.pkl").open("rb") as handle:
                snapshots.append(pickle.load(handle))
            continue
        with b.track("compatibility_"+task, "s0_compatibility", target, {"task": task, "probe": "native_init0"}, run_id=RUN_ID):
            env = ControlledEnv(task_key=task, observation_mode="on_request")
            try:
                env.start_episode(0, 42)
                snapshot = ObservationSnapshot.capture(task, 0, env.sample_observation(0), env.instruction)
                target.mkdir(parents=True, exist_ok=True)
                with (target/"snapshot.pkl").open("wb") as handle:
                    pickle.dump(snapshot, handle)
                snapshots.append(snapshot)
                for _ in range(20):
                    env.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
                publish(target/"result.json", {"task": task, "success": True,
                    "kind": "unscored_compatibility", "finite_observation": bool(np.isfinite(
                        snapshot.robot_state["eef"]["pos"]).all()), "state_identity": env.state_identity()})
            finally:
                env.close()
    snapshot = snapshots[0]
    empty = CommittedPrefix(0, frozen_copy(np.zeros((0, 7))), (), digest_arrays(np.zeros((0, 7))))
    base = ResearchRequest(1, 77, snapshot, 0, empty, 42, time.monotonic(), sampler_mode="B_RAW")
    native = adapter.infer_request(base).actions
    prefix = frozen_copy(native[:22])
    committed = CommittedPrefix(0, prefix, (), digest_arrays(prefix))
    probe = {}
    for mode in ("B_RAW", "P_HARD", "RTC_GUIDED"):
        request = replace(base, sampler_mode=mode,
                          prefix=empty if mode == "B_RAW" else committed,
                          guide_actions=frozen_copy(native[:25]) if mode == "RTC_GUIDED" else None)
        for _ in range(3):
            adapter.infer_request(request)
        torch.cuda.reset_peak_memory_stats()
        measurements = []
        for _ in range(10):
            started = time.monotonic()
            result = adapter.infer_request(request)
            if result.actions.shape != (50, 7) or not np.isfinite(result.actions).all():
                raise ValueError("invalid real policy output")
            measurements.append({"endpoint_s": time.monotonic()-started, **result.stages})
        probe[mode] = {"endpoint_p50_s": float(np.median([r["endpoint_s"] for r in measurements])),
            "endpoint_p95_s": float(np.percentile([r["endpoint_s"] for r in measurements], 95)),
            "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
            "reserved_peak_bytes": torch.cuda.max_memory_reserved(), "measurements": measurements}
    probe["B_SHIFT"] = probe["B_RAW"]
    no_prefix = adapter.infer_request(replace(base, sampler_mode="RTC_GUIDED")).actions
    hard = adapter.infer_request(replace(base, sampler_mode="P_HARD", prefix=committed))
    adapter.probe_debug = True
    guide_request = replace(base, sampler_mode="RTC_GUIDED", prefix=committed, guide_actions=frozen_copy(native[:25]))
    guided = adapter.infer_request(guide_request)
    changed_actions = native[:25].copy()
    changed_actions[:22, :3] = np.clip(changed_actions[:22, :3]+.1, -1, 1)
    changed_prefix = frozen_copy(changed_actions[:22])
    changed = adapter.infer_request(replace(guide_request,
        prefix=CommittedPrefix(0, changed_prefix, (), digest_arrays(changed_prefix)),
        guide_actions=frozen_copy(changed_actions)))
    mechanism = {"no_prefix_bitwise_equal": bool(np.array_equal(native, no_prefix)),
        "hard_prefix_roundtrip_error": hard.sampler["environment_prefix_roundtrip_max_error"],
        "hard_projection_errors": hard.sampler["projection_errors_before_after"],
        "rtc_suffix_intervention_linf": float(np.max(np.abs(guided.actions[22:]-changed.actions[22:]))),
        "rtc_guidance": guided.sampler, "changed_rtc_guidance": changed.sampler}
    if not mechanism["no_prefix_bitwise_equal"] or mechanism["rtc_suffix_intervention_linf"] <= 0:
        raise ValueError("real sampler compatibility/conditioning gate failed")
    np.savez_compressed(destination/"mechanism_arrays.npz", native=native, hard=hard.actions,
                        guided=guided.actions, changed=changed.actions, prefix=prefix)
    publish(destination/"mechanism_probe.json", mechanism)
    publish(destination/"resource_probe.json", probe)
    checkpoint_inventory = inventory(ROOT/CHECKPOINT, (ROOT/CHECKPOINT).rglob("*"))
    result = {"checkpoint": CHECKPOINT, "model_sha256": MODEL_SHA,
        "checkpoint_inventory": checkpoint_inventory, "gpu": torch.cuda.get_device_name(),
        "source_inventory": source_inventory(), "historical_source_inventory": old_files,
        "sampling": {"steps": 10, "dt": -.1, "horizon": 50, "valid_action_dim": 7, "padded_action_dim": 32},
        "upstream_rtc_note": "native installed formula retained, including its order of requires_grad and velocity evaluation",
        "resource_probe": "resource_probe.json", "mechanism_probe": "mechanism_probe.json"}
    publish(destination/"audit.json", result)
    publish(destination/"complete.json", {"passed": True, "compatibility_resets": 3})
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS/"AUDIT.md").write_text("# PLAN-05 S0 审计\n\nE1 全权重哈希、三任务开发状态、原生采样、硬投影和安装版本 RTC guidance 均通过真实检查。\n\n"
        "原生 RTC 的局部 autograd 运行在 no_grad 内，未使用 inference_mode；保持安装源码公式和 padding guidance。"
        "它是项目控制器适配，不是原论文全系统复现。完整数值和源码哈希见 ../../runs/latency_v3_plan05/audit。\n")
    return result


def verify_physical_starts():
    destination = RUN/"physical_restore"
    if (destination/"complete.json").exists():
        return read(destination/"complete.json")
    eligibility = exposure_inventory()
    states, poses = physical_assets()
    checks = []
    b = budget()
    for task, indices in eligibility["eligible_indices"].items():
        index = indices[0]
        for seed in (42, 20261003):
            target = destination/"trials"/f"{task}__{seed}"
            if (target/"result.json").exists():
                checks.append(read(target/"result.json"))
                continue
            with b.track(target.name, "physical_restore", target, {"task": task, "init_index": index,
                         "seed": seed, "probe": "physical_restore_no_policy"}, run_id=RUN_ID):
                env = ControlledEnv(task_key=task, observation_mode="on_request")
                env.model_pose_bank = poses[task]
                try:
                    env.start_episode(index, seed, states=states[task])
                    snap = env.sample_observation(0)
                    initial_success = bool(env._env._env.check_success())
                    result = {"task": task, "init_index": index, "seed": seed, "success": True,
                        "kind": "unscored_restore_no_policy", "scene_sha256": env.state_identity(),
                        "images_sha256": digest_arrays(snap["pixels"]), "initial_success": initial_success}
                    if initial_success:
                        raise ValueError("formal state already meets task success predicate")
                    publish(target/"result.json", result)
                    checks.append(result)
                finally:
                    env.close()
    for task in states:
        subset = [r for r in checks if r["task"] == task]
        if len({r["scene_sha256"] for r in subset}) != 1 or len({r["images_sha256"] for r in subset}) != 1:
            raise ValueError("physical scene or images differ across reset seeds")
    result = {"passed": True, "attempts": len(checks), "checks": checks}
    publish(destination/"complete.json", result)
    return result
