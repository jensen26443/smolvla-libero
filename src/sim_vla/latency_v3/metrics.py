"""Offline costs, clocks and action diagnostics with explicit denominators."""
import json

import numpy as np

from ..latency_v2.statistics import wilson
from ..latency_v2.storage import publish
from ..tasks import TASK_BY_KEY
from .experiments import DOCS, RUN, read, trial_key


def quantiles(values):
    return np.percentile(values, [50, 95, 99]).tolist() if len(values) else None


def detailed_metrics(core, asynchronous):
    details = []
    for stage, values in (("formal_core", core), ("formal_async", asynchronous)):
        for row in values:
            key = trial_key({k: row[k] for k in ("task", "init_index", "seed", "latency", "profile", "config")})
            directory = RUN/stage/"trials"/key
            requests = read(directory/"requests.json")
            events = [json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines()]
            steps = [e for e in events if e["event"] == "step"]
            actions = np.load(directory/"arrays.npz", allow_pickle=False)["actions"]
            switches = [e["tick"] for e in events if e["event"] == "activate" and 0 < e["tick"] < len(actions)]
            images = [e["tick"]+1-s["tick"] for e in steps
                      for name, s in e.get("sampling", {}).items() if name in ("image", "image2")]
            prefix_errors = [q["sampler"].get("environment_generated_prefix_max_error",
                q["sampler"].get("environment_prefix_roundtrip_max_error"))
                for q in requests if q["prefix_length"]]
            prefix_errors = [v for v in prefix_errors if v is not None]
            wall = row["wall_seconds"]
            detail = {"stage": stage, "trial_key": key, "task": row["task"], "init_index": row["init_index"],
                "latency": row["latency"], "method": row["config"]["method"], "success": row["success"],
                "termination_reason": row["termination_reason"], "steps": row["steps"],
                "success_simulation_s": .05*row["steps"] if row["success"] else None,
                "success_episode_wall_s": wall if row["success"] else None,
                "episode_wall_s_including_startup_and_tail": wall,
                "requests_per_episode_wall_s": row["n_requests"]/wall,
                "generation_s_per_episode_wall_s": row["generation_seconds"]/wall,
                "startup_wait_s": row["startup_wait_s"],
                "endpoint_p50_p95_p99_s": quantiles([q["endpoint_s"] for q in requests]),
                "continuation_endpoint_p50_p95_p99_s": quantiles([q["endpoint_s"] for q in requests if q["prefix_length"]]),
                "action_observation_age_p50_p95_p99_ticks": quantiles([e["age"] for e in steps if e["age"] is not None]),
                "cached_image_age_p50_p95_p99_ticks": quantiles(images),
                "switch_translation_delta_p50_p95_p99": quantiles([np.linalg.norm(actions[t, :3]-actions[t-1, :3]) for t in switches]),
                "switch_gripper_change_count": int(sum(actions[t, -1] != actions[t-1, -1] for t in switches)),
                "generated_prefix_max_error_p50_p95_p99": quantiles(prefix_errors),
                "max_consecutive_fallback": row["max_consecutive_fallback"],
                "late_results": row["late_results"], "expired_results": row["expired_results"],
                "coverage_infeasible_count": row["coverage_infeasible_count"],
                "generated_prefix_error_is_not_execution_violation": True}
            details.append(detail)
    groups = []
    for stage in ("formal_core", "formal_async"):
        methods = sorted({d["method"] for d in details if d["stage"] == stage})
        for method in methods:
            subset = [d for d in details if d["stage"] == stage and d["method"] == method]
            groups.append({"stage": stage, "method": method, "episodes": len(subset),
                "successes": sum(d["success"] for d in subset),
                "successful_tasks_per_wall_hour": 3600*sum(d["success"] for d in subset)/sum(d["episode_wall_s_including_startup_and_tail"] for d in subset),
                "wall_s_denominator": sum(d["episode_wall_s_including_startup_and_tail"] for d in subset),
                "mean_startup_wait_s": float(np.mean([d["startup_wait_s"] for d in subset])),
                "termination_reasons": {reason: sum(d["termination_reason"] == reason for d in subset)
                                        for reason in sorted({d["termination_reason"] for d in subset})},
                "short_failure_count_under_50_steps": sum(d["steps"] < 50 and not d["success"] for d in subset)})
    output = {"trials": details, "groups": groups,
        "wall_clock_scope": "episode policy startup + control + submitted tail; excludes reset, model loading, trace/video writes",
        "command_diagnostics_scope": "normalized action commands, not physical contact truth or joint smoothness",
        "percentile_order": [50, 95, 99]}
    publish(DOCS/"detailed_metrics.json", output)
    lines = ["# PLAN-05 逐任务结果与成本", "", "Wilson 区间是描述性逐格区间；主推断仍为任务内物理 init 配对 bootstrap。",
        "成功墙钟吞吐包括策略启动、控制和尾请求，排除 reset、模型载入和落盘；不等同于整机总作业吞吐。", "",
        "| 后端 | 任务 | L | 方法 | 成功/回合 | Wilson 95% |",
        "| --- | --- | --- | --- | --- | --- |"]
    for stage, values in (("formal_core", core), ("formal_async", asynchronous)):
        for task in TASK_BY_KEY:
            for latency in sorted({r["latency"] for r in values}):
                for method in sorted({r["config"]["method"] for r in values}):
                    subset = [r for r in values if r["task"] == task and r["latency"] == latency and r["config"]["method"] == method]
                    s = sum(r["success"] for r in subset)
                    low, high = wilson(s, len(subset))
                    lines.append(f"| {stage} | {task} | {latency} | {method} | {s}/{len(subset)} | [{low:.1%}, {high:.1%}] |")
    lines += ["", "逐 trial 的成功时间、endpoint 分位数、图像/动作观测年龄、切换命令差、迟到/过期、前缀生成误差见 detailed_metrics.json。",
        "承诺执行一致性由原数组逐位核验；生成前缀误差不代表承诺执行违约。S0 资源与显存实测见 audit/resource_probe.json。",
        "短失败回合保留完整成功率分母，但其控制频率和单位仿真时间成本不能代表持续稳定运行。"]
    (DOCS/"TASK_RESULTS.md").write_text("\n".join(lines)+"\n")
    return output
