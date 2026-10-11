"""Rebuild gate evidence from real cycles, with explicit research boundaries."""
import csv
import json
import pickle
from collections import Counter

import numpy as np

from ..latency_v2.integrity import file_sha256, inventory
from ..latency_v2.storage import publish, trial_key
from ..latency_v2.timing import provenance_complete
from ..latency_v4.layout import MODEL_SHA
from ..tasks import TASK_BY_KEY
from .experiments import DEVELOPMENT_TASKS, STAGES, rows, verify_frozen
from .layout import CHECKPOINT, DOCS, METHODS, ROOT, RUN, RUN_ID, RUNTIME, budget, read


def quantiles(values):
    return np.percentile(values, [50, 95, 99]).tolist() if values else []


def gate(values, stage="validation"):
    n, minimum_full = (2, 2) if stage == "development" else (6, 4)
    tasks = DEVELOPMENT_TASKS if stage == "development" else tuple(TASK_BY_KEY)
    methods = {}
    for method in METHODS:
        subset = [row for row in values if row["config"]["method"] == method]
        full = sum(row["sustained_300_ticks"] for row in subset)
        task_activations = {task: any(row["sustained_300_ticks"] and row["prefix_activations"] >= 3
                                    for row in subset if row["task"] == task) for task in tasks}
        errors = sum(bool(row["error"]) or not row["control_step_invariant"] for row in subset)
        methods[method] = {"n": len(subset), "full_300_ticks": full,
            "tasks_with_three_activations_in_full_episode": task_activations,
            "prefix_requests": sum(row["prefix_requests"] for row in subset),
            "prefix_activations": sum(row["prefix_activations"] for row in subset),
            "catchup_accepted": sum(row["catchup_accepted"] for row in subset),
            "realtime_passed": sum(row["realtime_20hz_passed"] for row in subset),
            "actual_hz_p50_p95_p99": quantiles([row["actual_control_hz"] for row in subset]),
            "full_episode_actual_hz_p50_p95_p99": quantiles([row["actual_control_hz"] for row in subset
                                                          if row["sustained_300_ticks"]]),
            "termination_reasons": dict(Counter(row["termination_reason"] for row in subset)),
            "invariant_errors": errors,
            "passed": len(subset) == n and full >= minimum_full and all(task_activations.values()) and errors == 0}
    return {"passed": all(item["passed"] for item in methods.values()), "methods": methods,
            "realtime_gate_passed": len(values) == 2*n and all(row["realtime_20hz_passed"] and
                                                            row["control_step_invariant"] for row in values)}


def report():
    DOCS.mkdir(parents=True, exist_ok=True)
    values = {stage: rows(stage, complete=False) for stage in STAGES}
    gates = {stage: gate(subset, stage) for stage, subset in values.items()}
    ledger = read(budget().path)
    probe = read(RUN / "profile_v3/production_probe.json") if (RUN / "profile_v3/production_probe.json").exists() else None
    testing = read(RUN / "testing/checks.json") if (RUN / "testing/checks.json").exists() else {}
    complete = len(values["development"]) == 4 and len(values["validation"]) == 12
    summary = {"run_id": RUN_ID, "implementation_complete": bool(testing.get("passed")),
        "experiment_complete": complete, "frozen_configuration_complete": (RUN / "frozen/experiment_spec.json").exists(),
        "timing_provenance_complete": complete and all(row["timing_provenance_complete"] for subset in values.values() for row in subset),
        "sustained_execution_supported": gates["validation"]["passed"],
        "realtime_gate_passed": gates["validation"]["realtime_gate_passed"],
        "engineering_problem_resolved_within_validation": gates["validation"]["passed"] and gates["validation"]["realtime_gate_passed"],
        "resume_claims_verified": (RUN / "testing/resume_check.json").exists(),
        "counts": {stage: len(subset) for stage, subset in values.items()}, "gates": gates,
        "attempts_used": len(ledger["attempts"]), "remaining_attempts": budget().remaining(),
        "total_plan06_and_plan07_additional_attempts": 564 + len(ledger["attempts"]),
        "historical_total_attempts": 2334 + len(ledger["attempts"]),
        "attempt_status_counts": dict(Counter(item["status"] for item in ledger["attempts"])),
        "preflight_attempts_retained": 10, "development_scope_amended_before_scoring": True,
        "runtime": dict(RUNTIME), "probe_endpoint_p95_s": probe["initial_endpoint_p95_s"] if probe else None,
        "original_plan06_development_gate_passed": False, "original_plan06_realtime_gate_passed": False,
        "scope": "fixed engineering validation on exposed simulator states; no new task-success or real-robot claim"}
    requests = {}
    for stage, subset in values.items():
        for method in METHODS:
            selected = [row for row in subset if row["config"]["method"] == method]
            records = []
            for row in selected:
                identity = {key: row[key] for key in ("task", "init_index", "seed", "state_id", "latency", "profile", "config", "runtime")}
                records += read(RUN / stage / "trials" / trial_key(identity) / "requests.json")
            requests[stage + ":" + method] = {"n": len(records),
                "state_capture_ms_p50_p95_p99": quantiles([row["observation_prepare_s"] * 1000 for row in records]),
                "render_ms_p50_p95_p99": quantiles([row["stages"]["render_scene_s"] * 1000 for row in records]),
                "endpoint_s_p50_p95_p99": quantiles([row["endpoint_s"] for row in records])}
    summary["request_distributions"] = requests
    publish(DOCS / "summary.json", summary)
    with (DOCS / "trials.csv").open("w", newline="") as handle:
        fields = ["stage", "task", "init_index", "method", "steps", "prefix_requests", "prefix_activations",
                  "catchup_accepted", "termination_reason", "realtime_20hz_passed", "actual_control_hz",
                  "fraction_cycles_under_55ms", "wall_seconds", "effective_config", "effective_runtime"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stage, subset in values.items():
            for row in subset:
                item = {key: row.get(key) for key in fields}
                item.update(stage=stage, method=row["config"]["method"],
                            effective_config=json.dumps(row["effective_config"], sort_keys=True),
                            effective_runtime=json.dumps(row["effective_runtime"], sort_keys=True))
                writer.writerow(item)
    lines = ["# PLAN-07 持续实时执行报告", "",
        f"**固定工程验证问题解决：{summary['engineering_problem_resolved_within_validation']}。**", "",
        "控制线程冻结完整物理场景，推理进程在独立副本恢复后渲染同分辨率双相机；三任务的同场景像素必须完全一致。",
        "图像物理身份为冻结 tick，实际渲染发生在稍后的工作进程；render、GPU推理和输出准备完整计入端点。",
        "冻结场景还包含model site RGBA，保留夹爪标记和炉灶状态的实际可视性。",
        "近期p95保留本轮生产探针下限。两条RTC规则共用这些变化，STRICT仍严格丢弃，CATCHUP仍最多3tick。", "",
        (f"本轮实际尝试 {summary['attempts_used']}/36，剩 {summary['remaining_attempts']}；原600次授权累计 "
         f"{summary['total_plan06_and_plan07_additional_attempts']}/600，历史累计 {summary['historical_total_attempts']}。"),
        "构造与reset单独计费；旧账本和旧验收材料不覆盖。",
        "初次校准因副本未同步可视site而失败，第二次校准误拒绝原库用于隐藏标记的负alpha；10次预评分尝试保留计费。",
        "评分前将开发诊断缩为前两个任务的4回合；固定验证仍覆盖全部三任务，共12回合，门槛保持。", "",
        "| 阶段 | 方法 | 完整300tick / n | 各任务连续激活 | 实时周期通过 / n | 实际Hz中位数 |",
        "| --- | --- | --- | --- | --- | ---: |"]
    for stage, item in gates.items():
        for method, statistics in item["methods"].items():
            hz = statistics["actual_hz_p50_p95_p99"]
            lines.append(f"| {stage} | {method} | {statistics['full_300_ticks']}/{statistics['n']} | "
                f"{all(statistics['tasks_with_three_activations_in_full_episode'].values())} | "
                f"{statistics['realtime_passed']}/{statistics['n']} | {hz[0] if hz else 0:.2f} |")
    lines += ["", "| 阶段与方法 | 冻结状态准备 p50/p95（ms） | 工作进程渲染 p50/p95（ms） | endpoint p50/p95（s） |",
              "| --- | --- | --- | --- |"]
    for name, item in requests.items():
        if item["n"]:
            lines.append(f"| {name} | {item['state_capture_ms_p50_p95_p99'][0]:.2f}/"
                f"{item['state_capture_ms_p50_p95_p99'][1]:.2f} | {item['render_ms_p50_p95_p99'][0]:.2f}/"
                f"{item['render_ms_p50_p95_p99'][1]:.2f} | {item['endpoint_s_p50_p95_p99'][0]:.3f}/"
                f"{item['endpoint_s_p50_p95_p99'][1]:.3f} |")
    lines += ["", "完整持续回合与短失败回合在summary.json中分开记录；周期门槛≥95%≤55ms，不等于精确20Hz。",
        "本轮是固定既有仿真状态上的工程验证，不报告持续诊断任务成功率，不声称新盲态、训练外泛化或真机部署。",
        "PLAN-06原持续门槛与实时门槛仍失败。本轮改变观测执行路径和估计器，不追溯改写原协议，也不把历史成功率直接相减归因。", ""]
    (DOCS / "REPORT.md").write_text("\n".join(lines) + "\n")
    flags = {key: summary[key] for key in ("implementation_complete", "experiment_complete", "frozen_configuration_complete",
        "timing_provenance_complete", "sustained_execution_supported", "realtime_gate_passed",
        "engineering_problem_resolved_within_validation", "resume_claims_verified")}
    (DOCS / "VALIDATION.md").write_text("# PLAN-07 验收字段\n\n```json\n" +
        json.dumps(flags, indent=2, ensure_ascii=False) + "\n```\n")
    (DOCS / "RUNBOOK.md").write_text("# PLAN-07 复现\n\n从sim_vla运行：\n\n```bash\n"
        "bash scripts/run_latency_v5.sh dry-run\nbash scripts/run_latency_v5.sh pipeline\n"
        "bash scripts/run_latency_v5.sh verify\nbash scripts/run_latency_v5.sh report\n"
        "```\n\n首次pipeline在同一作业内复用环境和推理进程，构造、校准reset和评分reset分别计费。"
        "已完成pipeline恢复先校验冻结和原始记录，在创建环境或worker前返回，0新增尝试。"
        "中断恢复须保留原attempt；recover仅核对死进程，不退款。部分恢复需要为新环境构造留足余额，"
        "没有预算时停止，不重新发额度，不重评已完成方法结果。\n\n"
        "图像由工作进程渲染冻结场景，按其真实物理tick记录；演示编码20fps不能代替实际控制频率。"
        "原PLAN-06资料保留，本轮结果位于docs/latency_v5。\n")
    evidence = [path for path in RUN.rglob("*") if path.is_file() and path.suffix not in (".lock", ".tmp")]
    publish(DOCS / "evidence_inventory.json", inventory(ROOT, evidence))
    return summary


def verify():
    spec = verify_frozen()
    if file_sha256(ROOT / CHECKPOINT / "model.safetensors") != MODEL_SHA:
        raise ValueError("the pinned model changed during the engineering run")
    total = Counter()
    for stage in STAGES:
        values = rows(stage)
        for row in values:
            identity = {key: row[key] for key in ("task", "init_index", "seed", "state_id", "latency", "profile", "config", "runtime")}
            directory = RUN / stage / "trials" / trial_key(identity)
            if row["code_sha256"] != spec["code_sha256"]:
                raise ValueError("result came from a different frozen control version")
            events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
            requests = read(directory / "requests.json")
            with (directory / "scene_snapshots.pkl").open("rb") as handle:
                snapshots = pickle.load(handle)
            if len(snapshots) != len(requests) or not provenance_complete(events, requests, log_error=row["log_error"]):
                raise ValueError("missing frozen scene or timing evidence")
            with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
                actions, prefixes, chunks = arrays["actions"], arrays["prefixes"], arrays["chunks"]
            steps = [event for event in events if event["event"] == "step"]
            cycles = [event for event in events if event["event"] == "cycle_complete"]
            if ([event["tick"] for event in steps] != list(range(row["steps"])) or len(cycles) != row["steps"]
                    or len(actions) != row["steps"] or not np.array_equal(actions, np.asarray([event["action"] for event in steps]))):
                raise ValueError("missing or repeated physical control step")
            fraction = float(np.mean([event["tick_wall_s"] <= .055 for event in cycles])) if cycles else 0.
            if fraction != row["fraction_cycles_under_55ms"] or row["realtime_20hz_passed"] != (bool(cycles) and fraction >= .95):
                raise ValueError("realtime gate does not match raw cycles")
            pending = None
            for event in events:
                if event["event"] == "submit":
                    if pending is not None:
                        raise ValueError("multiple inference requests in flight")
                    pending = event["request_id"]
                elif event["event"] in ("activate", "late_discard", "expired_discard"):
                    if event["request_id"] != pending:
                        raise ValueError("activated or discarded the wrong request")
                    pending = None
                if event["event"] == "activate":
                    request = requests[event["request_id"]]
                    age = event["tick"] - request["t_obs"]
                    late = age - request["prefix_length"]
                    if late < 0 or age >= 50 or (late and (row["config"]["method"] != "RTC_CATCHUP" or late > 3)):
                        raise ValueError("invalid commitment or late activation")
                    index = request["prefix_length"] if late == 0 else age
                    if event["start_index"] != index or (event["tick"] < len(actions) and not np.array_equal(
                            actions[event["tick"]], chunks[event["request_id"], index])):
                        raise ValueError("replayed a past action or activated the wrong command")
            for event in steps:
                if event["source"] == "committed" and not np.array_equal(
                        actions[event["tick"]], prefixes[event["request_id"], event["index"]]):
                    raise ValueError("command differs from its actual frozen commitment")
                if event.get("late_fallback"):
                    tick = event["tick"]
                    gripper = actions[tick - 1, -1] if tick else -1.
                    if np.any(actions[tick, :6] != 0) or actions[tick, -1] != gripper:
                        raise ValueError("late fallback semantics changed")
            for snapshot, request in zip(snapshots, requests):
                snapshot.validate()
                sampler = request["sampler"]
                if (snapshot.scene_sha256 != request["scene_sha256"]
                        or snapshot.digest != request["observation_sha256"]
                        or snapshot.visual_sha256 != request["visual_sha256"]
                        or sampler["rendered_scene_sha256"] != snapshot.scene_sha256
                        or sampler["rendered_visual_sha256"] != snapshot.visual_sha256
                        or sampler["input_physical_tick"] != snapshot.tick or sampler["steps"] != 5
                        or sampler["parameter_gradients_present"]):
                    raise ValueError("renderer or sampler did not use the complete frozen input")
            total.update(actions=len(steps), requests=len(requests), catchup_accepted=row["catchup_accepted"])
    ledger = read(budget().path)
    for attempt in ledger["attempts"]:
        if attempt["stage"] in STAGES and attempt["started_monotonic_ns"] <= spec["frozen_monotonic_ns"]:
            raise ValueError("scoring started before the complete freeze")
    for task in TASK_BY_KEY:
        if not read(RUN / "calibration_v3" / task / "result.json")["pixel_equivalence_verified"]:
            raise ValueError("missing same-scene pixel equivalence")
    result = {"passed": True, "run_id": RUN_ID, "counts": {stage: len(rows(stage)) for stage in STAGES},
              "totals": dict(total), "attempts_used": len(ledger["attempts"]), "remaining": budget().remaining(),
              "old_sources_and_ledgers_unchanged": True, "same_scene_pixels_verified": True,
              "timing_provenance_complete": True, "frozen_configuration_complete": True,
              "code_source_files": len(spec["code_inventory"])}
    publish(DOCS / "verification.json", result)
    return result
