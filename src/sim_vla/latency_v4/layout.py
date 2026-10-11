"""Isolated paths, complete asset fingerprints and an additional authorized budget."""
import json
import os
import re
from pathlib import Path

from ..assets import project_root
from ..latency_v2.experiments import CHECKPOINT
from ..latency_v2.integrity import file_sha256, inventory
from ..latency_v2.storage import Budget, canonical_hash, publish

ROOT = project_root()
RUN_ID = os.environ.get("SIM_VLA_LATENCY_V4_RUN_ID", "latency_v4_plan06")
if not re.fullmatch(r"latency_v4_[A-Za-z0-9_-]+", RUN_ID):
    raise ValueError("invalid isolated v4 run name")
RUN = ROOT/"runs"/RUN_ID
DOCS = ROOT/"docs/latency_v4" if RUN_ID == "latency_v4_plan06" else ROOT/"docs/latency_v4"/RUN_ID
CONFIG = ROOT/"configs/latency_v4"/RUN_ID/"experiment_spec.json"
BANK = RUN/"physical_bank/manifest.json"
OLD_LEDGER = ROOT/"runs/latency_v2_budget.json"
BUDGET_PATH = ROOT/"runs/latency_v4_plan06/budget.json"
MODEL_SHA = "e65416ea9ef2fc80f3e29aff5c70f972aa5009518c675d3cbaf4fca4db6892e8"


def read(path):
    return json.loads(Path(path).read_text())


def immutable_json(path, value):
    content = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode()
    immutable_bytes(path, content)


def immutable_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"immutable evidence already exists with different content: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())


def budget():
    # All isolated v4 run versions share the one additional authorization.
    path = BUDGET_PATH
    if not path.exists():
        if any((ROOT/"runs").glob("latency_v4_*/audit/audit.json")):
            raise ValueError("authorized v4 ledger is missing; existing audited runs cannot renew the budget")
        old = read(OLD_LEDGER)
        used = old["initial_s0_and_profile_attempts"]+len(old["attempts"])
        publish(path, {"cap": 600, "initial_s0_and_profile_attempts": 0, "attempts": [],
            "authorization": "user instruction to execute PLAN-06; additional 600 actual environment attempts",
            "historical_ledger": str(OLD_LEDGER.relative_to(ROOT)),
            "historical_ledger_start_sha256": file_sha256(OLD_LEDGER),
            "historical_used_at_start": used, "historical_cap": old["cap"],
            "historical_remaining_at_start": old["cap"]-used,
            "run_attempt_caps": {RUN_ID: 600}})
    data = read(path)
    if data["cap"] != 600 or data["initial_s0_and_profile_attempts"] != 0:
        raise ValueError("additional PLAN-06 budget must not change")
    if file_sha256(OLD_LEDGER) != data["historical_ledger_start_sha256"]:
        raise ValueError("historical ledger changed during this independent run")
    return Budget(path, cap=600, initial=0)


def source_inventory():
    paths = []
    for name in ("latency_v4", "latency_v2"):
        paths += list((ROOT/"src/sim_vla"/name).glob("*.py"))
    paths += [ROOT/"src/sim_vla/latency_eval"/name for name in ("policy_adapter.py", "env_adapter.py", "contracts.py")]
    paths += [ROOT/"src/sim_vla"/name for name in ("assets.py", "runtime.py", "tasks.py", "protocol.py")]
    paths += [ROOT/"scripts/run_latency_v4.sh"]
    upstream = ROOT/".venv/lib/python3.12/site-packages/lerobot/policies"
    paths += [upstream/name for name in ("smolvla/modeling_smolvla.py", "smolvla/configuration_smolvla.py",
        "rtc/modeling_rtc.py", "rtc/configuration_rtc.py", "rtc/debug_tracker.py")]
    paths += [Path(entry["path"]) for entry in read(ROOT/"runs/latency_v2_audit/audit.json")["upstream_source"].values()]
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in sorted(set(paths))}


def execution_assets():
    import importlib.metadata
    import platform

    from ..assets import libero_assets_dir
    from ..tasks import TASK_BY_KEY

    old = read(ROOT/"runs/latency_v2_audit/audit.json")
    supplemental = read(ROOT/"runs/latency_v2_audit/supplemental_provenance.json")
    checkpoint = inventory(ROOT/CHECKPOINT, (ROOT/CHECKPOINT).rglob("*"))
    if (file_sha256(ROOT/CHECKPOINT/"model.safetensors") != MODEL_SHA
            or checkpoint != read(ROOT/"runs/latency_v3_plan05/audit/audit.json")["checkpoint_inventory"]):
        raise ValueError("E1 weights or processor changed")
    asset_root = libero_assets_dir()
    assets = inventory(asset_root, (asset_root/entry["path"] for entry in old["libero_assets"]["files"]))
    vlm_root = Path(supplemental["snapshot"])
    vlm = inventory(vlm_root, (vlm_root/entry["path"] for entry in supplemental["inventory"]["files"]))
    if assets != old["libero_assets"] or vlm != supplemental["inventory"]:
        raise ValueError("pinned LIBERO or VLM assets changed")
    for entry in old["upstream_source"].values():
        if file_sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("installed upstream control source changed")
    task_root = ROOT/".venv/lib/python3.12/site-packages/libero/libero"
    task_files = [task_root/subdir/task.suite/(task.task_name+extension) for task in TASK_BY_KEY.values()
                  for subdir, extension in (("bddl_files", ".bddl"), ("init_files", ".pruned_init"))]
    return {"checkpoint": checkpoint, "libero_assets": assets, "vlm_processor": vlm,
        "vlm_snapshot": str(vlm_root), "upstream_control": old["upstream_source"],
        "task_files": inventory(task_root, task_files),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("lerobot", "torch", "transformers", "hf-libero", "mujoco", "gymnasium")},
        "python": platform.python_version()}


def progress(stage, **fields):
    payload = {"stage": stage, "run_id": RUN_ID, "additional_budget_used": 600-budget().remaining(),
        "remaining_budget": budget().remaining(),
        "resume_command": f"SIM_VLA_LATENCY_V4_RUN_ID={RUN_ID} bash scripts/run_latency_v4.sh pipeline", **fields}
    publish(RUN/"progress.json", payload)
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS/"PROGRESS.md").write_text("# PLAN-06 进度\n\n```json\n"+
        json.dumps(payload, indent=2, ensure_ascii=False)+"\n```\n")
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def physical_assets():
    import numpy as np

    from ..latency_v2.model_poses import ModelPoseBank

    manifest = read(BANK)
    states, poses = {}, {}
    for task, entry in manifest["tasks"].items():
        state_path, pose_path = BANK.parent/entry["states_path"], BANK.parent/entry["model_poses_path"]
        if file_sha256(state_path) != entry["states_sha256"] or file_sha256(pose_path) != entry["model_poses_sha256"]:
            raise ValueError("physical state or fixture pose asset changed")
        states[task] = np.load(state_path, allow_pickle=False)
        poses[task] = ModelPoseBank.load(pose_path, expected_states=20)
    return states, poses


def paired_seed(task, state_id):
    return int(canonical_hash(["PLAN-06", task, state_id, 20261005])[:8], 16) % (2**31-1)
