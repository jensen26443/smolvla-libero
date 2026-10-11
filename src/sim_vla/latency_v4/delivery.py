"""Rebuildable evidence, denominator-aware conclusions and raw invariant checks."""
import csv
import json
from collections import Counter

import numpy as np

from ..latency_v2.integrity import file_sha256, inventory
from ..latency_v2.storage import publish, trial_key
from ..latency_v2.timing import provenance_complete
from ..tasks import TASK_BY_KEY
from .continuation import AUTHORIZATION, original_trial_directory, verify_continuation
from .experiments import FORMAL_STAGES, rows
from .layout import CONFIG, DOCS, ROOT, RUN, RUN_ID, budget, execution_assets, read
from .schema import METHODS
from .statistics import cells, classify, paired_gain

STAGES = ("develop_sampling", "develop_period", "sustain", *FORMAL_STAGES)


def request_metrics(stage, values):
    output = []
    for row in values:
        identity = {key: row[key] for key in ("task", "init_index", "state_id", "seed", "latency", "profile", "config")}
        directory = RUN/stage/"trials"/trial_key(identity)
        requests = read(directory/"requests.json")
        for request in requests:
            output.append({"stage": stage, "task": row["task"], "state_id": row["state_id"],
                "method": row["config"]["method"], "sampling_steps": row["config"]["sampling_steps"],
                "prefix_length": request["prefix_length"], "endpoint_s": request.get("endpoint_s"),
                "observation_prepare_s": request["observation_prepare_s"],
                "control_receive_s": ((request["received_monotonic_ns"]-request["worker_completed_monotonic_ns"])/1e9
                    if request.get("received_monotonic_ns") is not None else None),
                "stages": request.get("stages", {}), "sampler": request.get("sampler", {}),
                "late_ticks": request.get("late_ticks"), "decision": request.get("decision"),
                "catchup_accepted": request.get("catchup_accepted", False)})
    return output


def groups(values):
    result = []
    for method in sorted({row["config"]["method"] for row in values}):
        subset = [row for row in values if row["config"]["method"] == method]
        result.append({"method": method, "n": len(subset),
            "steps": sum(row["steps"] for row in subset), "full_300_ticks": sum(row["steps"] == 300 for row in subset),
            "short_failures_under_50_ticks": sum(row["steps"] < 50 and row["success"] is not True for row in subset),
            "sustained_diagnostics": sum(row["sustained_diagnostic"] for row in subset),
            "termination_reasons": dict(Counter(row["termination_reason"] for row in subset)),
            "prefix_requests": sum(row["prefix_requests"] for row in subset),
            "prefix_activations": sum(row["prefix_activations"] for row in subset),
            "catchup_accepted": sum(row["catchup_accepted"] for row in subset),
            "catchup_rejected": sum(row["catchup_rejected"] for row in subset),
            "late_results": sum(row["late_results"] for row in subset),
            "expired_results": sum(row["expired_results"] for row in subset),
            "coverage_failures": sum(row["coverage_infeasible_count"] for row in subset),
            "fallback_fraction": sum(row["fallback_ticks"] for row in subset)/max(1, sum(row["steps"] for row in subset)),
            "longest_consecutive_fallback": max(row["max_consecutive_fallback"] for row in subset),
            "late_fallback_ticks": sum(row["late_fallback_ticks"] for row in subset),
            "switch_command_l2_p50_p95_p99": quantiles([value for row in subset for value in row["switch_command_l2"]]),
            "actual_hz_episode_distribution": [row["actual_control_hz"] for row in subset],
            "full_300_actual_hz": [row["actual_control_hz"] for row in subset if row["steps"] == 300],
            "short_failure_actual_hz": [row["actual_control_hz"] for row in subset if row["steps"] < 50 and row["success"] is not True],
            "realtime_gate_passed_episodes": sum(row["realtime_20hz_passed"] is True for row in subset),
            "period_p50_p95_p99_per_episode": [row["cycle_p50_p95_p99_s"] for row in subset],
            "work_over_50ms_per_episode": [row["deadline_missed_fraction"] for row in subset],
            "total_task_wall_s": sum(row["wall_seconds"]+row["reset_wall_s"] for row in subset),
            "total_generation_s": sum(row["generation_seconds"] for row in subset),
            "total_endpoint_s": sum(row["inference_seconds"] for row in subset),
            "n_requests": sum(row["n_requests"] for row in subset)})
    return result


def quantiles(values):
    return np.percentile(values, [50, 95, 99]).tolist() if values else None


def request_distributions(details):
    output = []
    for stage, method, steps in sorted({(row["stage"], row["method"], row["sampling_steps"]) for row in details}):
        subset = [row for row in details if (row["stage"], row["method"], row["sampling_steps"]) == (stage, method, steps)]
        output.append({"stage": stage, "method": method, "sampling_steps": steps, "n": len(subset),
            "endpoint_p50_p95_p99_s": quantiles([row["endpoint_s"] for row in subset if row["endpoint_s"] is not None]),
            "observation_prepare_p50_p95_p99_s": quantiles([row["observation_prepare_s"] for row in subset]),
            "control_receive_p50_p95_p99_s": quantiles([row["control_receive_s"] for row in subset
                                                       if row["control_receive_s"] is not None]),
            "worker_peak_allocated_bytes": max((row["sampler"].get("gpu_allocated_peak_bytes", 0) for row in subset), default=0),
            "worker_peak_reserved_bytes": max((row["sampler"].get("gpu_reserved_peak_bytes", 0) for row in subset), default=0),
            "peak_scope": "maximum worker process memory including warmup; not per-request allocation"})
    return output


def ablation(values):
    steps = sorted({row["config"]["sampling_steps"] for row in values}, reverse=True)
    output = {"sampling_steps": steps, "acceleration_identifiable": len(steps) == 2, "rule_effects": {}}
    for step in steps:
        output["rule_effects"][str(step)] = paired_gain([row for row in values if row["config"]["sampling_steps"] == step])
    if len(steps) == 2:
        output["acceleration_effects"] = {}
        selected = min(steps)
        for method in ("RTC_STRICT", "RTC_CATCHUP"):
            synthetic = []
            for row in values:
                if row["config"]["method"] != method:
                    continue
                synthetic.append({**row, "config": {**row["config"],
                    "method": "accelerated" if row["config"]["sampling_steps"] == selected else "ten_step"}})
            output["acceleration_effects"][method] = paired_gain(synthetic, "accelerated", "ten_step")
    output["cross_stage_reuse"] = "same physical states, paired repeated measures; not pooled across stages"
    return output


def report():
    DOCS.mkdir(parents=True, exist_ok=True)
    continuation = verify_continuation(run=RUN, run_id=RUN_ID) if (RUN/AUTHORIZATION).exists() else None
    values = {stage: rows(stage, require_complete=False) for stage in STAGES}
    expected = {stage: len(read(RUN/stage/"expected_trials.json")) if (RUN/stage/"expected_trials.json").exists() else 0
                for stage in STAGES}
    counts = {stage: len(subset) for stage, subset in values.items()}
    sampling = read(RUN/"sampling_selection.json") if (RUN/"sampling_selection.json").exists() else None
    gate = read(RUN/"development_gate.json") if (RUN/"development_gate.json").exists() else None
    frozen = read(RUN/"frozen/experiment_spec.json") if (RUN/"frozen/experiment_spec.json").exists() else None
    formal_complete = bool(frozen) and all(counts[stage] == len(frozen["matrices"][stage]) for stage in FORMAL_STAGES)
    main = secondary = factors = fixed_delay = None
    algorithm_result = "experiment_incomplete"
    if sampling and not sampling["passed"] or gate and not gate["passed"]:
        algorithm_result = "development_stage_infeasible"
    if formal_complete:
        main = paired_gain(values["formal_async"])
        secondary = [paired_gain(values["formal_async"], baseline=method) for method in ("RAW", "SHIFT")]
        factors = ablation(values["formal_ablation"])
        fixed_delay = {str(latency): paired_gain([row for row in values["formal_delay"] if row["latency"] == latency])
                       for latency in (10, 20)}
        algorithm_result = classify(main, secondary, values["formal_async"])
    checks = read(RUN/"testing/checks.json") if (RUN/"testing/checks.json").exists() else {}
    b = budget()
    ledger = read(b.path)
    job_costs = [read(path) for stage in STAGES for path in (RUN/stage).glob("job_accounting_*.json")]
    probe = read(RUN/"profile/production_probe.json")
    mechanism = read(RUN/"profile/mechanism.json")
    request_details = [item for stage in STAGES for item in request_metrics(stage, values[stage])]
    concurrent = []
    for task in TASK_BY_KEY:
        for steps in (10, 7, 5):
            for path, method in (("native", "RAW"), ("rtc", "RTC_STRICT")):
                subset = [item for item in request_details if item["stage"] == "develop_sampling" and item["task"] == task
                          and item["sampling_steps"] == steps and item["method"] == method
                          and (path == "native" or item["prefix_length"] > 0) and item["endpoint_s"] is not None]
                concurrent.append({"task": task, "sampling_steps": steps, "path": path,
                    "actual_requests": len(subset), "endpoint_p50_p95_p99_s": quantiles([item["endpoint_s"] for item in subset]),
                    "observation_prepare_p50_p95_p99_s": quantiles([item["observation_prepare_s"] for item in subset]),
                    "control_receive_p50_p95_p99_s": quantiles([item["control_receive_s"] for item in subset]),
                    "scope": "actual development control/render concurrent; variable request counts; no injected sleep"})
    publish(DOCS/"concurrent_probe.json", concurrent)
    publish(DOCS/"request_metrics.json", request_details)
    scored = [row for stage in STAGES for row in values[stage] if row["success"] is not None]
    timed_formal = [row for stage in ("formal_async", "formal_ablation") for row in values[stage]]
    summary = {"format": "latency-v4-plan06-summary-v1", "implementation_complete": bool(checks.get("passed")),
        "development_gate_passed": bool(gate and gate["passed"]),
        "sampling_gate_passed": bool(sampling and sampling["passed"]),
        "frozen_configuration_complete": bool(frozen), "experiment_complete": formal_complete,
        "timing_provenance_complete": bool(scored) and all(row["timing_provenance_complete"] for stage in STAGES for row in values[stage]),
        "sustained_execution_supported": bool(gate and gate["passed"]),
        "realtime_gate_passed": bool(timed_formal) and all(row["realtime_20hz_passed"] for row in timed_formal),
        "algorithm_result": algorithm_result, "resume_claims_verified": bool(checks.get("resume_claims_verified")),
        "continuation_authorized": bool(continuation),
        "original_gate_requirement_satisfied": bool(gate and gate["passed"]),
        "original_protocol_complete": formal_complete and bool(gate and gate["passed"]),
        "continuation": {"parent_run_id": continuation["parent_run_id"],
            "authorization_sha256": file_sha256(RUN/AUTHORIZATION),
            "original_failed_gate_retained": True,
            "scope": "fixed matrix extension authorized after the failed development gate; control feasibility limitation retained"}
            if continuation else None,
        "counts": counts, "expected_completed_stages": expected, "main": main, "secondary": secondary,
        "ablation": factors, "fixed_delay": fixed_delay, "sampling_selection": sampling, "sustained_gate": gate,
        "request_distributions": request_distributions(request_details),
        "cells": {stage: cells(subset) for stage, subset in values.items()},
        "groups": {stage: groups(subset) for stage, subset in values.items()},
        "additional_budget_used": 600-b.remaining(), "additional_budget_cap": 600,
        "remaining_additional_budget": b.remaining(), "historical_budget_used_at_start": ledger["historical_used_at_start"],
        "historical_ledger_unchanged": file_sha256(ROOT/ledger["historical_ledger"]) == ledger["historical_ledger_start_sha256"],
        "historical_strict_freeze_complete": False, "total_cumulative_attempts": ledger["historical_used_at_start"]+600-b.remaining(),
        "cost_scope": "task wall includes reset, policy startup, control and tail; excludes shared model loading, profiling and serialization",
        "successful_completion_selection_bias": "success-only durations condition on outcome; not a fair all-episode speed comparison",
        "switch_command_scope": "command difference is not contact force or physical trajectory stability truth",
        "noise_limitation": "paired seed is shared; different request counts give incomplete noise-sequence correspondence",
        "formal_spec_sha256": frozen["spec_sha256"] if frozen else None}
    summary["job_accounting"] = {"stage_jobs": job_costs,
        "total_stage_job_wall_s": sum(item["total_job_wall_s"] for item in job_costs),
        "reused_development_job_wall_s": sum(read(path)["total_job_wall_s"]
            for stage in ("develop_sampling", "develop_period", "sustain")
            for path in (RUN/stage).glob("job_accounting_*.json")) if continuation else 0.,
        "current_formal_job_wall_s": sum(read(path)["total_job_wall_s"]
            for stage in FORMAL_STAGES for path in (RUN/stage).glob("job_accounting_*.json")),
        "current_bank_and_restore_attempt_wall_s": sum(
            (item["finished_monotonic_ns"]-item["started_monotonic_ns"])/1e9
            for item in ledger["attempts"] if item["run_id"] == RUN_ID and item["stage"] in ("prepare_bank", "physical_restore")
            and item.get("finished_monotonic_ns") is not None),
        "profile_total_wall_s": probe["total_wall_s"],
        "profile_measured_forward_wall_s": sum(item["stages"]["forward_s"] for item in probe["measurements"]),
        "profile_forward_time_scope": "CUDA-synchronized wall spans including warmups; not active GPU kernel time",
        "diagnostic_denoise_cuda_event_s": sum(value for item in mechanism["checks"]
                                               for value in item.get("diagnostic_flow_gpu_s", [])),
        "gpu_cost_scope": "denoise stream event spans and worker peak memory recorded; total active GPU/loading time unknown",
        "worker_peak_memory_scope": "per-process maximum including warmup"}
    publish(DOCS/"summary.json", summary)
    export_csv(values)
    write_documents(summary)
    figures(summary)
    from .materials import tables, timeline
    from .videos import build_demo

    summary["catchup_timeline"] = timeline(values)
    tables(values, summary)
    demo = build_demo(values)
    summary["demo"] = demo
    publish(DOCS/"summary.json", summary)
    evidence = [path for path in RUN.rglob("*") if path.is_file() and path.suffix not in (".lock", ".tmp")]
    publish(DOCS/"evidence_inventory.json", inventory(ROOT, evidence))
    return summary


def export_csv(values):
    fields = ["stage", "task", "init_index", "state_id", "seed", "latency", "method", "sampling_steps", "period",
        "success", "termination_reason", "steps", "n_requests", "prefix_requests", "prefix_activations", "catchup_accepted",
        "catchup_rejected", "fallback_ticks", "max_consecutive_fallback", "coverage_infeasible_count", "late_results",
        "expired_results", "startup_wait_s", "wall_seconds", "reset_wall_s", "generation_seconds", "actual_control_hz",
        "realtime_20hz_passed", "sustained_300_ticks", "code_sha256", "effective_config"]
    with (DOCS/"trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for stage, subset in values.items():
            for row in subset:
                item = {key: row.get(key) for key in fields}
                item.update(stage=stage, method=row["config"]["method"], sampling_steps=row["config"]["sampling_steps"],
                            period=row["config"]["period"], effective_config=json.dumps(row["effective_config"], sort_keys=True))
                writer.writerow(item)


def write_documents(summary):
    formal = summary["experiment_complete"]
    continuing = summary["continuation_authorized"]
    probe = read(RUN/"profile/production_probe.json")
    endpoints = probe["initial_endpoint_p95_s"]
    rtc_reduction = 1-endpoints["5"]["rtc"]/endpoints["10"]["rtc"]
    gate = summary["sustained_gate"]
    strict_full = gate["methods"]["RTC_STRICT"]["full_300_ticks"] if gate else 0
    catchup_full = gate["methods"]["RTC_CATCHUP"]["full_300_ticks"] if gate else 0
    title = "正式矩阵完成" if formal else "开发阶段结果；正式矩阵未执行完成"
    if continuing:
        title = "授权续跑矩阵完成；原开发门槛未通过" if formal else "授权续跑进行中；原开发门槛未通过"
    lines = ["# PLAN-06 实验报告", "", f"**{title}。结论：`{summary['algorithm_result']}`。**", "",
        (f"新增实际尝试 {summary['additional_budget_used']}/600；旧账本保持 {summary['historical_budget_used_at_start']}/1800，"
         f"本轮累计历史成本 {summary['total_cumulative_attempts']}。没有动用旧余额 30。"), "",
        "| 阶段 | 已完成回合 |", "| --- | ---: |"]
    lines += [f"| {stage} | {count} |" for stage, count in summary["counts"].items()]
    if continuing:
        lines += ["", ("本轮依据用户追加指令“继续完成剩余任务”执行预先固定的剩余矩阵。原126个开发回合直接复用，"
            "没有重新评分或计作新回合；原3/6与6/6持续门槛记录保持不变。"),
            "这是门槛失败后的独立后续实验，不能称原PLAN-06门槛通过；STRICT早停和可行域限制必须与成功率差同时解释。"]
    lines += ["", "## 推理耗时与研究问题", "",
        "| Flow步数 | native endpoint p95秒 | RTC endpoint p95秒 |", "| ---: | ---: | ---: |"]
    for step in (10, 7, 5):
        lines.append(f"| {step} | {endpoints[str(step)]['native']:.6f} | {endpoints[str(step)]['rtc']:.6f} |")
    lines += ["", (f"10→5步 RTC endpoint p95 下降 {rtc_reduction:.1%}。每格三个任务各30次有效调用，"
        f"另各5次warmup；生产测量共 {len(probe['measurements'])} 次调用，总墙钟 {probe['total_wall_s']:.2f} 秒。"),
        "Q1 的时延降低与开发筛选可行性得到支持；任务表现无损和新状态泛化未获证明。",
        ("Q2 的正式闭环成功率差与固定次要比较见下文；结论仅适用于本次授权续跑与固定任务。" if formal else
         "Q2 的正式闭环成功率增益未获检验；开发持续诊断可用于描述执行机制，不能替代正式主比较。")]
    lines += ["", "采样和周期开发使用已暴露 native init0/init1；小样本筛选不证明无损加速或独立泛化。",
        "静止原生启动、单在途、H=50、每 tick 一次物理 step、50ms 目标和完整尾请求成本一致。",
        "STRICT/CATCHUP 共用 guidance、估计器、周期与中性后备；接续仅允许 1–3 tick 迟到并从实际年龄索引激活。",
        "中性后备可能偏离生成时预期状态；时间对应修正不保证状态一致或轨迹平滑。切换命令差不能代替接触力。", ""]
    sampling = summary["sampling_selection"]
    if sampling:
        lines += [f"采样选择：`{sampling['sampling_steps']}` 步；资格判定 `{sampling['passed']}`。",
            "完整候选分数、拒绝理由和时延值见 summary.json 的 sampling_selection；所有落选配置也保留。", ""]
    if summary["sustained_gate"]:
        lines += ["| 持续诊断方法 | 完整 300 tick / 6 | 每任务至少三次前缀激活 | 门槛 |",
                  "| --- | --- | --- | --- |"]
        for method, item in summary["sustained_gate"]["methods"].items():
            lines.append(f"| {method} | {item['full_300_ticks']}/6 | {all(item['tasks_with_three_activations'].values())} | {item['passed']} |")
        lines += ["", "持续诊断不报告任务成功率；短失败回合的 Hz 和周期通过不能证明可持续运行。", ""]
        if not gate["passed"]:
            missing = [task for task, passed in gate["methods"]["RTC_STRICT"]["tasks_with_three_activations"].items()
                       if not passed]
            lines += [(f"原停止原因：STRICT 完整运行 {strict_full}/6，低于4/6；未满足前缀激活要求的任务为 "
                f"`{', '.join(missing)}`。CATCHUP 的 {catchup_full}/6 仅支持当前开发条件下的持续执行。"),
                ("后续用户指令授权在该门槛失败事实保留的条件下执行固定剩余矩阵，未重新搜索或重评开发回合。" if continuing else
                 "没有生成正式新状态池、完成正式冻结或执行正式异步/固定释放/因子消融矩阵。"), ""]
    if summary["main"]:
        main = summary["main"]
        catchup = next(item for item in summary["groups"]["formal_async"] if item["method"] == "RTC_CATCHUP")
        lines += [(f"固定主比较 CATCHUP−STRICT：{main['gain']:.2%}，95% 配对 CI "
                   f"[{main['ci95'][0]:.2%}, {main['ci95'][1]:.2%}]。三任务等权，物理状态内配对，10,000 draws，seed=20261005。"),
                  "区间包含零不意味着等效或无损；RAW/SHIFT 为预声明次要比较。",
                  (f"主矩阵CATCHUP共接受{catchup['catchup_accepted']}次迟到接续，"
                   f"共{catchup['prefix_requests']}次前缀请求；实际激活数量限定机制证据强度。"
                   "两条RTC规则共用配置，但请求数及实际端点耗时可能不同；不把单次接续事件推广为普遍收益。"), ""]
    else:
        lines += ["主成功率差与正式置信区间尚无证据，不能用开发成绩替代正式推断。",
                  ("用户已授权按固定矩阵续跑；新状态生成、恢复与正式结果需完成后单独验收。" if continuing else
                   "按 PLAN-06 门槛停止时，交付正确实现及失败机制，不继续扫描或增加状态。"), ""]
    lines += ["请求成本、实际后备、迟到/过期、切换命令、逐回合周期分布、完整持续回合与短失败回合分别见 summary.json 和 request_metrics.json。",
        "空闲 worker 探针至少 5 warmup+30 有效调用/任务×步数×路径；并行控制/渲染来自真实开发请求，样本数逐格披露。",
        "相机准备在开发请求中实测；视觉编码、传输的独立拆分未知，不编造细粒度耗时。",
        "forward_s为CUDA同步前后的前向墙钟跨度，不能等同GPU核活跃时间；诊断flow的CUDA事件跨度及进程显存峰值另列。",
        "任务墙钟包括 reset、启动、控制、尾请求，模型加载/探针/落盘另列；成功回合完成时间具有条件选择偏差。",
        "历史 PLAN-05 的初始 p95 冻结缺口不能被本轮修复或新结果追溯补齐。", ""]
    (DOCS/"REPORT.md").write_text("\n".join(lines)+"\n")
    flags = {key: summary[key] for key in ("implementation_complete", "development_gate_passed", "frozen_configuration_complete",
        "experiment_complete", "timing_provenance_complete", "sustained_execution_supported", "realtime_gate_passed",
        "algorithm_result", "resume_claims_verified", "continuation_authorized", "original_protocol_complete")}
    (DOCS/"VALIDATION.md").write_text("# PLAN-06 验收字段\n\n```json\n"+json.dumps(flags, indent=2, ensure_ascii=False)+"\n```\n")
    (DOCS/"DECISIONS.md").write_text("# PLAN-06 决策\n\n配置筛选、周期全部分数与持续门槛来自原始回合重算。\n\n```json\n"+
        json.dumps({"sampling": sampling,
                    "periods": read(RUN/"selection.json") if (RUN/"selection.json").exists() else None,
                    "sustained_gate": summary["sustained_gate"]}, indent=2, ensure_ascii=False)+"\n```\n")
    selection = read(RUN/"selection.json") if (RUN/"selection.json").exists() else None
    publish(CONFIG.parent/"development_spec.json", {
        "status": "development_only_not_a_formal_freeze", "selection": selection,
        "development_gate": gate, "production_probe_sha256": file_sha256(RUN/"profile/production_probe.json"),
        "initial_endpoint_p95_s": endpoints,
        "stages": {stage: {"spec_path": str((RUN/stage/"spec.json").relative_to(ROOT)),
                           "spec_sha256": file_sha256(RUN/stage/"spec.json"),
                           "expected_trials_sha256": file_sha256(RUN/stage/"expected_trials.json")}
                   for stage in ("develop_sampling", "develop_period", "sustain") if (RUN/stage/"spec.json").exists()}})
    (DOCS/"RESUME.md").write_text("# 可使用的项目描述\n\n项目：**SmolVLA 多任务适配与异步动作执行研究**。\n\n"
        "- 基于单卡 RTX 3090 和 LIBERO Goal 三任务，适配 SmolVLA 双相机、8维状态与7维相对动作执行链；"
        "实现H=50、单在途队列、真实承诺覆盖检查和最多3 tick迟到的时间索引接续，保留原生RTC guidance。\n"
        f"- 对10/7/5步完成630次已有快照生产调用；空闲worker的RTC endpoint p95从 "
        f"{endpoints['10']['rtc']:.3f}秒降至{endpoints['5']['rtc']:.3f}秒（下降{rtc_reduction:.1%}），"
        "以开发任务筛选配置，不声称无损加速。\n"
        f"- 完成{summary['counts']['develop_sampling']}个采样、{summary['counts']['develop_period']}个周期和"
        f"{summary['counts']['sustain']}个持续诊断回合；开发条件下CATCHUP完整300 tick为{catchup_full}/6，"
        f"STRICT为{strict_full}/6，按联合门槛停止正式扩展；实际消耗{summary['additional_budget_used']}/600次尝试。\n\n"
        f"当前算法结论 `{summary['algorithm_result']}`；正式成功率提升、新状态泛化及实时部署均未获验证。"
        "历史训练结果与本轮开发结果分开陈述。"
        "RTC 是已有方法，项目贡献是执行器适配、实验设计与机制诊断。\n")
    (DOCS/"INTERVIEW.md").write_text("# 技术说明与面试问答\n\n"
        "**如何定义迟到？** 输入观测 t_obs，承诺长度 c，边界 t_obs+c。实际激活年龄 age=t_act−t_obs；late=t_act−边界。"
        "提前结果等待边界并从 c 激活；CATCHUP 只在 late∈[1,3] 且 age<H 时从 age 激活。\n\n"
        "**为何可能仍失败？** 迟到期间实际执行的是零运动/沿用夹爪的后备，可能与生成时预期状态不同；"
        "接续后剩余 H−age 动作变少，下一请求仍必须满足真实 remaining_old≥c 和 2c≤H。\n\n"
        "**如何保证公平？** 同一 sampler、步数、噪声 seed、guidance、动态周期、初始 p95 和后备；"
        "RTC 周期只按 STRICT 选择。两者区别仅是迟到结果的激活规则。\n\n"
        "**如何检验贡献？** 固定主比较 CATCHUP−STRICT；RAW/SHIFT 是强基线；10步/选定步数与两条规则做因子消融。"
        "按物理初始化在任务内配对 bootstrap，跨阶段复用状态不会增加独立样本数。\n\n"
        "**如何解释门槛？** 每回合至少 95% 周期≤55ms 且 step 无缺失/重复；它不等于精确 20Hz，"
        "必须另用 300 tick 持续诊断验证连续前缀激活。\n\n"
        "**历史冻结缺口如何处理？** 原始数据保留，新冻结绑定生产探针字节、参数、源码、状态和模型；"
        "不能在看过历史得分后补写绑定并声称旧实验严格验收通过。\n\n"
        f"**本轮实际结论？** RTC探针10→5步p95下降{rtc_reduction:.1%}；开发CATCHUP完整300 tick为"
        f"{catchup_full}/6，STRICT为{strict_full}/6。"+
        ("原联合门槛未通过；用户追加授权后按固定矩阵继续，保留原门槛、全部失败与强基线。\n" if continuing else
         "联合门槛未通过时停止正式扩展，保留负结果与全部基线。\n"))
    (DOCS/"TECHNICAL_SUMMARY.md").write_text("# 一页技术摘要\n\n"+
        "SmolVLA 多任务适配与异步执行，E1 10,000 步 checkpoint，H=50，双相机/state8/action7，3090。\n\n"+
        "加速仅改变 flow 采样步数 10/7/5。RTC 使用上游权重 10、EXP、soft overlap3；局部 autograd 保持，无参数梯度积累。\n\n"+
        "覆盖 c=ceil(p95/50ms)+2，最近10个完成请求/p95；K=min(K_selected,H−c)，2c≤H，remaining_old≥c。\n\n"+
        "提前等待边界；STRICT 丢弃迟到；CATCHUP 仅 1–3 tick 内从 age 激活。相同零运动/沿用夹爪后备。\n\n"+
        f"当前交付状态：{title}；结论 `{summary['algorithm_result']}`。实际尝试 {summary['additional_budget_used']}/600。\n\n"+
        f"空闲RTC endpoint p95：10步{endpoints['10']['rtc']:.3f}s，7步{endpoints['7']['rtc']:.3f}s，"
        f"5步{endpoints['5']['rtc']:.3f}s；10→5步下降{rtc_reduction:.1%}。开发持续300 tick："
        f"CATCHUP {catchup_full}/6，STRICT {strict_full}/6；不据此声称正式成功率增益。\n\n"+
        "统计主单位为 task 内物理状态；三任务等权。"+
        ("本轮依据用户追加授权执行剩余矩阵，原持续门槛仍为失败。\n" if continuing else
         "负结果允许，未通过持续门槛停止正式扩展。\n"))
    if continuing:
        note = ("\n本轮为用户在原持续门槛失败后明确授权的固定矩阵续跑。原门槛仍未通过，"
                "正式差值应结合STRICT短失败、RAW/SHIFT强基线和成本解释。\n")
        for name in ("INTERVIEW.md", "TECHNICAL_SUMMARY.md"):
            path = DOCS/name
            path.write_text(path.read_text()+note)
    if formal:
        main = summary["main"]
        result_text = (f"固定主比较CATCHUP−STRICT为{main['gain']:.1%}，95%配对CI "
                       f"[{main['ci95'][0]:.1%}, {main['ci95'][1]:.1%}]；三任务等权、状态内配对。")
        (DOCS/"RESUME.md").write_text("# 可使用的项目描述\n\n项目：**SmolVLA 多任务适配与异步动作执行研究**。\n\n"
            "- 在RTX 3090与LIBERO Goal三任务上适配SmolVLA双相机/state8/action7执行链，"
            "实现单在途、承诺覆盖检查、最多3 tick迟到的时间索引接续及完整动作与时序审计。\n"
            f"- 对10/7/5步复用630次已有快照生产测量；RTC空闲endpoint p95从{endpoints['10']['rtc']:.3f}s"
            f"降至{endpoints['5']['rtc']:.3f}s（{rtc_reduction:.1%}），以开发数据选择共同配置，不声称任务表现无损。\n"
            f"- 完成{summary['counts']['formal_async']}个真实异步、{summary['counts']['formal_delay']}个固定释放及"
            f"{summary['counts']['formal_ablation']}个因子消融回合。{result_text}\n\n"
            f"结论：`{summary['algorithm_result']}`。RTC为已有方法；贡献为执行适配、评测与机制分析。"
            +(f"原联合持续门槛未通过（STRICT {strict_full}/6，CATCHUP {catchup_full}/6）；"
              "本次为用户明确授权的后续固定矩阵研究，不能宣称原门槛通过。" if continuing else "")+
            "新状态与训练状态是否重叠未知，实时部署与精确20Hz需另按验收字段陈述。\n")
        for name in ("INTERVIEW.md", "TECHNICAL_SUMMARY.md"):
            path = DOCS/name
            path.write_text(path.read_text()+"\n"+result_text+"\n")


def figures(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = DOCS/"figures"
    target.mkdir(exist_ok=True)
    probe = read(RUN/"profile/production_probe.json")
    steps = [10, 7, 5]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for path in ("native", "rtc"):
        axes[0].plot(steps, [probe["initial_endpoint_p95_s"][str(s)][path] for s in steps], "o-", label=path)
    axes[0].axhline(1., color="gray", linestyle="--", label="RTC feasibility screening limit")
    axes[0].set(xlabel="Flow steps", ylabel="Idle worker endpoint p95 (s)")
    axes[0].legend(fontsize=8)
    selection = summary["sampling_selection"]
    if selection:
        for method in ("RAW", "RTC_STRICT"):
            axes[1].plot(steps, [entry["methods"][method]["successes"]/6 for entry in selection["candidates"]], "o-", label=method)
        axes[1].legend()
    axes[1].set(xlabel="Flow steps", ylabel="Exposed-development success fraction", ylim=(-.05, 1.05))
    fig.tight_layout()
    fig.savefig(target/"sampling_latency_and_development.png", dpi=160)
    plt.close(fig)
    sustain = summary["groups"]["sustain"]
    if sustain:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([item["method"] for item in sustain], [item["full_300_ticks"] for item in sustain])
        ax.axhline(4, color="gray", linestyle="--")
        ax.set(ylabel="Full 300-tick diagnostic episodes / 6", ylim=(0, 6.5))
        fig.tight_layout()
        fig.savefig(target/"sustained_execution_gate.png", dpi=160)
        plt.close(fig)
    source = "formal_async" if summary["experiment_complete"] else "develop_period"
    subset = summary["groups"][source]
    if subset:
        reasons = sorted({reason for item in subset for reason in item["termination_reasons"]})
        fig, ax = plt.subplots(figsize=(10, 4))
        bottom = np.zeros(len(subset))
        for reason in reasons:
            heights = [item["termination_reasons"].get(reason, 0) for item in subset]
            ax.bar([item["method"] for item in subset], heights, bottom=bottom, label=reason)
            bottom += heights
        ax.set(ylabel="Episodes", title=source+" termination reasons")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(target/"termination_reasons.png", dpi=160)
        plt.close(fig)
    if summary["main"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        values = summary["cells"]["formal_async"]
        rates = [np.mean([row["success_rate"] for row in values if row["method"] == method]) for method in METHODS]
        ax.bar(METHODS, rates)
        ax.set(ylabel="Equal-task success fraction", ylim=(0, 1))
        fig.tight_layout()
        fig.savefig(target/"formal_async_success.png", dpi=160)
        plt.close(fig)
        scope = "Authorized continuation; original gate failed" if summary["continuation_authorized"] else "Frozen formal matrix"
        comparisons = [summary["main"], *summary["secondary"]]
        gains = np.asarray([item["gain"] for item in comparisons])*100
        intervals = np.asarray([item["ci95"] for item in comparisons])*100
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.errorbar(gains, range(len(comparisons)), xerr=np.stack([gains-intervals[:, 0], intervals[:, 1]-gains]),
                    fmt="o", capsize=5)
        ax.axvline(0, color="gray", linestyle="--")
        ax.set(yticks=range(len(comparisons)), yticklabels=["CATCHUP - "+item["baseline"] for item in comparisons],
               xlabel="Paired success difference (percentage points), 95% CI", title=scope)
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(target/"paired_success_intervals.png", dpi=180)
        fig.savefig(target/"paired_success_intervals.svg")
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        factors = summary["cells"]["formal_ablation"]
        for method in ("RTC_STRICT", "RTC_CATCHUP"):
            sampling = summary["ablation"]["sampling_steps"]
            axes[0].plot(sampling, [np.mean([row["success_rate"] for row in factors
                if row["method"] == method and row["sampling_steps"] == step]) for step in sampling], "o-", label=method)
        axes[0].set(xlabel="Flow steps", ylabel="Equal-task success fraction", ylim=(-.05, 1.05), title="Fixed-state factor ablation")
        axes[0].legend(fontsize=8)
        for method in METHODS:
            axes[1].plot([10, 20], [np.mean([row["success_rate"] for row in summary["cells"]["formal_delay"]
                if row["method"] == method and row["latency"] == latency]) for latency in (10, 20)], "o-", label=method)
        axes[1].set(xlabel="Simulated release delay (ticks)", ylabel="Equal-task success fraction", ylim=(-.05, 1.05),
                    title="Auxiliary mechanism condition")
        axes[1].legend(fontsize=8)
        fig.suptitle(scope, fontsize=11)
        fig.tight_layout()
        fig.savefig(target/"factor_ablation_and_fixed_release.png", dpi=180)
        fig.savefig(target/"factor_ablation_and_fixed_release.svg")
        plt.close(fig)


def verify():
    if (RUN/AUTHORIZATION).exists():
        verify_continuation(run=RUN, run_id=RUN_ID)
    old = read(RUN/"audit/historical_inventory.json")
    for entry in old["files"]:
        if file_sha256(ROOT/entry["path"]) != entry["sha256"]:
            raise ValueError("protected historical evidence changed: "+entry["path"])
    audit = read(RUN/"audit/audit.json")
    if file_sha256(ROOT.parent/"plan/02/PLAN-06_推理加速与迟到动作接续研究.md") != audit["plan_sha256"]:
        raise ValueError("PLAN-06 changed after the initial audit")
    if execution_assets() != audit["execution_assets"]:
        raise ValueError("audited model, processor, task or runtime assets changed")
    b = budget()
    ledger = read(b.path)
    attempts = {item["generation"]: item for item in ledger["attempts"]}
    if len(attempts) != len(ledger["attempts"]) or b.remaining() < 0:
        raise ValueError("attempt generations or budget cap are invalid")
    counts, timing, snapshot_count = {}, True, 0
    probe = read(RUN/"profile/production_probe.json")
    source_sets = [(RUN/"profile/code_snapshot", probe["source_inventory"])]
    for stage in STAGES:
        values = rows(stage, require_complete=False)
        counts[stage] = len(values)
        if values:
            spec = read(RUN/stage/"spec.json")
            source_sets.append((RUN/stage/"code_snapshot", spec["code_inventory"]))
        for row in values:
            identity = {key: row[key] for key in ("task", "init_index", "state_id", "seed", "latency", "profile", "config")}
            directory = RUN/stage/"trials"/trial_key(identity)
            attempt = attempts.get(row["generation"])
            charged_directory = original_trial_directory(stage, spec, trial_key(identity), run=RUN)
            if (not attempt or attempt["attempt_id"] != row["attempt_id"] or attempt["run_id"] != spec["run_id"]
                    or attempt["stage"] != stage or attempt["target"] != str(charged_directory)
                    or attempt["key"] != trial_key(identity)
                    or attempt["status"] != ("method_error" if row["error"] else "completed")
                    or row["code_sha256"] != spec["code_sha256"]):
                raise ValueError("result does not match its original source and charged attempt")
            events = [json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines()]
            requests = read(directory/"requests.json")
            with np.load(directory/"arrays.npz", allow_pickle=False) as arrays:
                actions, prefixes, chunks = arrays["actions"], arrays["prefixes"], arrays["chunks"]
            steps = [item for item in events if item["event"] == "step"]
            cycles = [item for item in events if item["event"] == "cycle_complete"]
            if ([item["tick"] for item in steps] != list(range(row["steps"])) or len(cycles) != len(steps)
                    or len(actions) != len(steps) or not np.array_equal(actions, np.asarray([item["action"] for item in steps]))):
                raise ValueError("one-step-per-tick action evidence mismatch")
            in_flight = None
            for event in events:
                if event["event"] == "submit":
                    if in_flight is not None:
                        raise ValueError("more than one pending request")
                    in_flight = event["request_id"]
                elif event["event"] in ("activate", "late_discard", "expired_discard"):
                    if event["request_id"] != in_flight:
                        raise ValueError("completion/activation identity mismatch")
                    in_flight = None
                if event["event"] == "activate":
                    request = requests[event["request_id"]]
                    age = event["tick"]-request["t_obs"]
                    c = request["prefix_length"]
                    method = row["config"]["method"]
                    if method in ("RTC_STRICT", "RTC_CATCHUP", "HARD"):
                        if event["tick"] < request["t_obs"]+c:
                            raise ValueError("activated before commitment boundary")
                        late = age-c
                        expected = c if late == 0 else age
                        if (late > 0 and (method != "RTC_CATCHUP" or not 1 <= late <= 3)) or age >= 50:
                            raise ValueError("invalid late activation")
                    else:
                        expected = age if method == "SHIFT" else 0
                    if event["start_index"] != expected:
                        raise ValueError("activation index replays past actions")
                    if event["tick"] < len(actions) and not np.array_equal(
                            actions[event["tick"]], chunks[event["request_id"], event["start_index"]]):
                        raise ValueError("activated command differs from the actual generated suffix")
            for step in steps:
                if step["source"] == "committed" and not np.array_equal(
                        actions[step["tick"]], prefixes[step["request_id"], step["index"]]):
                    raise ValueError("actual command violates its frozen commitment")
                if step.get("late_fallback"):
                    action = actions[step["tick"]]
                    previous_gripper = actions[step["tick"]-1, -1] if step["tick"] else -1.
                    if np.any(action[:6] != 0) or action[-1] != previous_gripper:
                        raise ValueError("late fallback does not use the common neutral semantics")
            for request in requests:
                if request.get("sampler", {}).get("steps") != row["config"]["sampling_steps"]:
                    raise ValueError("sampling steps not consistent across worker/model/log")
                if request.get("sampler", {}).get("parameter_gradients_present"):
                    raise ValueError("parameter gradients accumulated")
            timing = timing and provenance_complete(events, requests, log_error=row.get("log_error"))
    for directory, sources in source_sets:
        for relative, digest in sources.items():
            path = directory/relative
            if not path.exists() or file_sha256(path) != digest:
                raise ValueError("original measured source snapshot changed or missing: "+str(path))
            snapshot_count += 1
    if not timing:
        raise ValueError("actual timing provenance is incomplete")
    frozen_path = RUN/"frozen/experiment_spec.json"
    if frozen_path.exists():
        from .freeze import verify_frozen

        spec = verify_frozen()
        for attempt in read(b.path)["attempts"]:
            if (attempt["run_id"] == RUN_ID and attempt["stage"] in FORMAL_STAGES
                    and attempt["started_monotonic_ns"] <= spec["frozen_monotonic_ns"]):
                raise ValueError("formal attempt predates the complete freeze")
    result = {"passed": True, "counts": counts, "timing_provenance_complete": timing,
        "historical_files_unchanged": len(old["files"]), "additional_budget_used": 600-b.remaining(),
        "historical_ledger_unchanged": True, "historical_plan_unchanged": True,
        "execution_assets_unchanged": True, "original_source_snapshots_verified": snapshot_count,
        "attempt_status_counts": dict(Counter(item["status"] for item in ledger["attempts"]))}
    publish(DOCS/"verification.json", result)
    return result
