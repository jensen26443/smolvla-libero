"""Read-only historical audit and probes on already exposed observation snapshots."""
import pickle
import subprocess
import time
from dataclasses import replace

import numpy as np

from ..latency_v2.integrity import file_sha256, inventory
from ..latency_v2.locking import run_lock
from ..latency_v2.schema import CommittedPrefix, digest_arrays, frozen_copy
from ..latency_v2.storage import publish
from ..tasks import TASK_BY_KEY
from .layout import (
    CHECKPOINT,
    DOCS,
    ROOT,
    RUN,
    budget,
    execution_assets,
    immutable_json,
    read,
    source_inventory,
)
from .schema import SAMPLING_STEPS, Request


def audit():
    budget()
    path = RUN/"audit/audit.json"
    if path.exists():
        result = read(path)
        if execution_assets() != result["execution_assets"]:
            raise ValueError("audited execution assets changed")
        return result
    snapshots = {}
    for task in TASK_BY_KEY:
        snapshot = ROOT/"runs/latency_v3_plan05/audit/compatibility"/task/"snapshot.pkl"
        snapshots[task] = {"path": str(snapshot.relative_to(ROOT)), "sha256": file_sha256(snapshot),
                           "eligibility": "historically policy-exposed native init0; development only"}
    protected = []
    for pattern in ("latency_v2*", "latency_v3*"):
        for directory in (ROOT/"runs").glob(pattern):
            protected.extend(path for path in directory.rglob("*") if path.is_file() and path.suffix != ".lock")
    for directory in (ROOT/"src/sim_vla/latency_v2", ROOT/"src/sim_vla/latency_v3",
                      ROOT/"configs/latency_v3", ROOT/"docs/latency_v3", ROOT/"docs/latency_v2"):
        protected.extend(path for path in directory.rglob("*") if path.is_file()
                         and "__pycache__" not in path.parts and path.name != "RUNBOOK.md")
    protected.append(ROOT.parent/"plan/02/PLAN-06_推理加速与迟到动作接续研究.md")
    # PLAN is outside ROOT; preserve it separately from the relative-path inventory.
    plan = protected.pop()
    immutable_json(RUN/"audit/historical_inventory.json", inventory(ROOT, sorted(set(protected))))
    old = read(ROOT/"runs/latency_v2_budget.json")
    assets = execution_assets()
    result = {"format": "latency-v4-audit-v1", "started_unix": time.time(), "environment_attempts": 0,
        "checkpoint": CHECKPOINT, "execution_assets": assets, "source_inventory_at_audit": source_inventory(),
        "existing_snapshots": snapshots, "plan_sha256": file_sha256(plan),
        "historical_attempts": old["initial_s0_and_profile_attempts"]+len(old["attempts"]),
        "historical_cap": old["cap"], "historical_freeze_complete": False,
        "historical_limit": "PLAN-05 initial endpoint p95 was not bound before the first formal score",
        "git_status": subprocess.run(["git", "status", "--short", "--branch"], cwd=ROOT,
                                     capture_output=True, text=True, check=False).stderr.strip(),
        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.strip()}
    immutable_json(path, result)
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS/"AUDIT.md").write_text("# PLAN-06 审计\n\n已核对 E1 权重、处理器、上游源码、LIBERO 资产和任务文件；"
        "使用历史已暴露快照，审计不 reset 环境。历史原始证据清单保存在新 run 的 audit 中。\n\n"
        f"旧账本 {result['historical_attempts']}/{old['cap']}；本轮独立新增额度 600，绑定旧账本起始哈希。"
        "当前 .git 无法提供有效仓库状态，未初始化或执行提交。\n\n"
        "PLAN-05 严格冻结未通过；F1 当前代码修复不能追溯补齐历史证据。最新历史结论见 "
        "[ACCEPTANCE_RECHECK](../latency_v3/ACCEPTANCE_RECHECK.md)。\n")
    return result


def load_snapshots():
    evidence = read(RUN/"audit/audit.json")["existing_snapshots"]
    snapshots = {}
    for task, entry in evidence.items():
        path = ROOT/entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("development snapshot changed")
        with path.open("rb") as handle:
            snapshots[task] = pickle.load(handle)
    return snapshots


def empty_prefix(tick=0):
    array = frozen_copy(np.zeros((0, 7)))
    return CommittedPrefix(tick, array, (), digest_arrays(array))


def mechanism_probe(snapshots):
    import torch

    from ..latency_v2.policy import PrefixPolicyAdapter
    from .policy import SamplingPolicyAdapter

    adapter = SamplingPolicyAdapter(checkpoint=ROOT/CHECKPOINT, sampling_steps=None, warmup_calls=3)
    adapter.warmup()
    records, arrays, prefixes = [], {}, {}
    for task, snapshot in snapshots.items():
        base = Request(1, 77, snapshot, 0, empty_prefix(), 42, time.monotonic(), sampler_mode="RAW")
        adapter.set_sampling_steps(10)
        adapter._disable_rtc()
        adapter.reset()
        reference = PrefixPolicyAdapter.infer_request(adapter, base).actions
        adapter.reset()
        native = adapter.infer_request(base).actions
        maximum_error = float(np.max(np.abs(reference-native)))
        if maximum_error > 2e-5:
            raise ValueError("ten-step native reference regression")
        prefix_actions, guide_actions = frozen_copy(native[:22]), frozen_copy(native[:25])
        prefix = CommittedPrefix(0, prefix_actions, (), digest_arrays(prefix_actions))
        prefixes[task] = (prefix, guide_actions)
        arrays[task+"_reference"] = reference
        arrays[task+"_native"] = native
        for steps in SAMPLING_STEPS:
            for mode in ("RAW", "RTC_STRICT"):
                request = replace(base, sampling_steps=steps, sampler_mode=mode,
                    prefix=empty_prefix() if mode == "RAW" else prefix,
                    guide_actions=None if mode == "RAW" else guide_actions)
                original = adapter.policy.model.denoise_step
                count = [0]
                flow_events = []

                def counted(*args, _count=count, _original=original, _events=flow_events, **kwargs):
                    _count[0] += 1
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    output = _original(*args, **kwargs)
                    end.record()
                    _events.append((start, end))
                    return output

                adapter.policy.model.denoise_step = counted
                try:
                    output = adapter.infer_request(request)
                finally:
                    adapter.policy.model.denoise_step = original
                if count[0] != steps or output.sampler["steps"] != steps:
                    raise ValueError("reported steps do not match actual flow evaluations")
                records.append({"task": task, "steps": steps, "path": mode,
                    "denoise_evaluations": count[0], "native_reference_max_error": maximum_error,
                    "parameter_gradients_present": output.sampler["parameter_gradients_present"],
                    "diagnostic_flow_gpu_s": [start.elapsed_time(end)/1000 for start, end in flow_events],
                    "profiler_used_for_delay_initialization": False})
                arrays[f"{task}_{steps}_{mode}"] = output.actions
        guided_request = replace(base, sampler_mode="RTC_STRICT", prefix=prefix, guide_actions=guide_actions)
        guided = adapter.infer_request(guided_request).actions
        changed = guide_actions.copy()
        changed[:22, :3] = np.clip(changed[:22, :3]+.1, -1, 1)
        changed = frozen_copy(changed)
        changed_prefix = CommittedPrefix(0, frozen_copy(changed[:22]), (), digest_arrays(changed[:22]))
        intervention = adapter.infer_request(replace(guided_request, prefix=changed_prefix, guide_actions=changed))
        response = float(np.max(np.abs(guided[22:]-intervention.actions[22:])))
        if response <= 0:
            raise ValueError("RTC suffix did not respond to the guidance input")
        records.append({"task": task, "rtc_suffix_intervention_linf": response})
    np.savez_compressed(RUN/"profile/mechanism_arrays.npz", **arrays)
    immutable_json(RUN/"profile/mechanism.json", {"passed": True, "checks": records,
        "fine_timing_scope": "CUDA denoise events; visual/embedding/transfer subcomponents unknown",
        "normalizer_and_padding": "unchanged inherited native chain; no cross-observation cache"})
    del adapter
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return prefixes


def profile():
    from ..latency_v2.worker import InferenceWorker

    audit()
    target = RUN/"profile/production_probe.json"
    if target.exists():
        return read(target)
    with run_lock(RUN/"profile"):
        snapshots = load_snapshots()
        started = time.time()
        prefixes = mechanism_probe(snapshots)
        engine = InferenceWorker(factory_name="sim_vla.latency_v4.policy:SamplingPolicyAdapter",
            factory_kwargs={"checkpoint": ROOT/CHECKPOINT, "sampling_steps": None, "warmup_calls": 3},
            startup_timeout=180.)
        measurements = []
        combinations = [(task, steps, mode) for task in TASK_BY_KEY for steps in SAMPLING_STEPS
                        for mode in ("RAW", "RTC_STRICT")]
        code = {key: value for key, value in source_inventory().items() if "latency_v4" not in key
                or key.endswith(("audit.py", "policy.py", "schema.py", "run_latency_v4.sh", "layout.py"))}
        ordinal = 0
        try:
            for round_index in range(35):
                offset = round_index % len(combinations)
                for task, steps, mode in combinations[offset:]+combinations[:offset]:
                    prefix, guide = prefixes[task]
                    request = Request(1, ordinal, snapshots[task], 0,
                        empty_prefix() if mode == "RAW" else prefix, 42, time.monotonic(),
                        sampler_mode=mode, sampling_steps=steps, guide_actions=None if mode == "RAW" else guide)
                    ordinal += 1
                    engine.submit(request)
                    result = engine.drain()
                    if result.error or result.sampler["steps"] != steps:
                        raise RuntimeError(result.error or "worker sampling configuration mismatch")
                    endpoint = result.ready_wall-request.submit_wall
                    row = {"task": task, "steps": steps, "path": mode, "round": round_index,
                        "warmup": round_index < 5, "endpoint_s": endpoint,
                        "control_receive_s": (result.received_monotonic_ns-result.worker_completed_monotonic_ns)/1e9,
                        "stages": result.stages, "sampler": result.sampler}
                    measurements.append(row)
                publish(RUN/"profile/measurements.partial.json", measurements)
                print(f"profile round {round_index+1}/35; completed {ordinal} snapshot model calls", flush=True)
        finally:
            publish(RUN/"profile/worker_shutdown.json", engine.close())
        cells = []
        for task, steps, mode in combinations:
            values = [row for row in measurements if row["task"] == task and row["steps"] == steps
                      and row["path"] == mode and not row["warmup"]]
            cells.append({"task": task, "steps": steps, "path": mode, "effective_calls": len(values),
                "warmup_calls": 5, "endpoint_p50_p95_p99_s": np.percentile(
                    [row["endpoint_s"] for row in values], [50, 95, 99]).tolist()})
        estimates = {}
        for steps in SAMPLING_STEPS:
            estimates[str(steps)] = {}
            for path in ("RAW", "RTC_STRICT"):
                values = [row["endpoint_s"] for row in measurements if row["steps"] == steps
                          and row["path"] == path and not row["warmup"]]
                estimates[str(steps)]["native" if path == "RAW" else "rtc"] = float(np.percentile(values, 95))
        report = {"format": "latency-v4-production-probe-v1", "condition": "idle_spawned_worker_existing_snapshots",
            "warmups_per_combination": 5, "effective_calls_per_combination": 30,
            "order": "round-robin shifted by one combination each round", "source_inventory": code,
            "started_unix": started, "finished_unix": time.time(), "total_wall_s": time.time()-started,
            "initial_endpoint_p95_s": estimates, "cells": cells, "measurements": measurements,
            "camera_observation_prepare": "not remeasured here: historical frozen snapshots, no reset",
            "visual_encoding_s": None, "transfer_s": None,
            "concurrent_control_condition": "measured separately from real development request traces",
            "fine_profiler_used": False, "environment_attempts": 0}
        immutable_json(target, report)
        return report
