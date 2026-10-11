"""Illustrate an actual admitted result and retain its original raw evidence."""
import json

import numpy as np

from ..latency_v2.integrity import file_sha256
from ..latency_v2.storage import publish, trial_key
from .continuation import AUTHORIZATION
from .layout import DOCS, ROOT, RUN


def timeline(values):
    selected = None
    for stage in ("formal_async", "develop_period", "sustain"):
        for row in values[stage]:
            if row["config"]["method"] != "RTC_CATCHUP" or not row["catchup_accepted"]:
                continue
            identity = {key: row[key] for key in ("task", "init_index", "state_id", "seed", "latency", "profile", "config")}
            directory = RUN/stage/"trials"/trial_key(identity)
            events = [json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines()]
            admitted = next(event for event in events if event["event"] == "activate" and event["catchup_accepted"])
            selected = stage, row, directory, events, admitted
            break
        if selected:
            break
    if selected is None:
        return {"available": False, "reason": "no actual admitted late result; no invented timeline"}
    stage, row, directory, events, event = selected
    requests = json.loads((directory/"requests.json").read_text())
    request = requests[event["request_id"]]
    start, activation, boundary = request["t_obs"], event["tick"], event["commitment_boundary"]
    steps = [step for step in events if step["event"] == "step" and start <= step["tick"] <= activation+4]
    with np.load(directory/"arrays.npz", allow_pickle=False) as arrays:
        for step in steps:
            if step["tick"] >= activation and not np.array_equal(
                    arrays["actions"][step["tick"]],
                    arrays["chunks"][event["request_id"], event["start_index"]+step["tick"]-activation]):
                raise ValueError("timeline suffix does not match the actual admitted generated actions")
    continuing = (RUN/AUTHORIZATION).exists()
    record = {"available": True, "stage": stage, "formal": stage == "formal_async",
        "authorized_continuation": continuing,
        "selection": "first admitted event in first expected trial, without outcome filtering",
        "task": row["task"], "init_index": row["init_index"], "method": "RTC_CATCHUP",
        "sampling_steps": row["config"]["sampling_steps"], "period": row["config"]["period"],
        "trial_directory": str(directory.relative_to(ROOT)), "request": request, "activation": event,
        "steps": steps, "event_file_sha256": file_sha256(directory/"events.jsonl"),
        "arrays_sha256": file_sha256(directory/"arrays.npz"),
        "renderer_sha256": file_sha256(ROOT/"src/sim_vla/latency_v4/materials.py")}
    publish(DOCS/"catchup_timeline.json", record)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    fig, ax = plt.subplots(figsize=(11, 4.6))
    colors = {"prefix": "#3973b9", "fallback": "#a1a7b1", "suffix": "#269c7b"}
    for step in steps:
        tick = step["tick"]
        source = "prefix" if tick < boundary else ("fallback" if tick < activation else "suffix")
        ax.add_patch(Rectangle((tick-.45, .1), .9, .5, color=colors[source]))
        index = step["index"] if tick < activation else event["start_index"]+tick-activation
        if source == "fallback" or (tick-start) % 3 == 0 or tick == activation:
            ax.text(tick, .72, "neutral" if source == "fallback" else f"i={index}", ha="center", fontsize=8)
    for tick, label, color in ((start, "observation / submit", "#3973b9"),
                               (boundary, "commitment boundary", "#626a78"),
                               (activation, "receive / activate", "#d65353")):
        ax.axvline(tick, color=color, linestyle="--")
        ax.text(tick, 1.1 if tick != activation else 1.35, label,
                ha="left" if tick == start else "center", fontsize=9, color=color)
    scope = "FORMAL" if stage == "formal_async" else "DEVELOPMENT (exposed states)"
    if continuing and stage == "formal_async":
        scope = "AUTHORIZED CONTINUATION (original gate failed)"
    ax.set(xlim=(start-2, activation+6), ylim=(0, 1.65), yticks=[], xlabel="Actual control tick",
        title=f"{scope} / {row['task']} / late={event['late_ticks']} ticks / start index={event['start_index']}")
    ax.legend(handles=[Patch(color=colors[key], label=label) for key, label in (
        ("prefix", "actual committed old actions"), ("fallback", "actual neutral fallback"),
        ("suffix", "actual catch-up suffix actions"))], loc="upper center", bbox_to_anchor=(.5, -.22), ncol=3, fontsize=8)
    fig.tight_layout()
    (DOCS/"figures").mkdir(exist_ok=True)
    fig.savefig(DOCS/"figures/catchup_timeline.png", dpi=180)
    fig.savefig(DOCS/"figures/catchup_timeline.svg")
    plt.close(fig)
    return {"available": True, "stage": stage, "trial_directory": record["trial_directory"],
            "late_ticks": event["late_ticks"], "start_index": event["start_index"]}


def tables(values, summary):
    """Supplement readable task/cost tables, keeping development descriptive."""
    selection_path = RUN/"selection.json"
    if not selection_path.exists():
        return
    selection = json.loads(selection_path.read_text())
    formal = summary["experiment_complete"]
    stage = "formal_async" if formal else "develop_period"
    selected = [row for row in values[stage] if formal or row["config"]["period"] == selection["periods"][row["config"]["method"]]]
    lines = ["", "## 四方法与成本", "",
        ("授权续跑结果，原持续门槛未通过。" if summary["continuation_authorized"] else "正式结果。")
        if formal else "以下为已暴露开发状态上的描述性结果，不是正式推断。",
        "", "| 方法 | 采样/周期 | 成功/回合 | 总任务墙钟秒 | 生成秒/仿真秒均值 |",
        "| --- | --- | --- | ---: | ---: |"]
    for method in ("RAW", "SHIFT", "RTC_STRICT", "RTC_CATCHUP"):
        subset = [row for row in selected if row["config"]["method"] == method]
        if not subset:
            continue
        cfg = subset[0]["config"]
        lines.append(f"| {method} | {cfg['sampling_steps']}/K{cfg['period']} | {sum(row['success'] for row in subset)}/{len(subset)} | "
                     f"{sum(row['wall_seconds']+row['reset_wall_s'] for row in subset):.2f} | "
                     f"{np.mean([row['generation_seconds_per_sim_second'] for row in subset]):.3f} |")
    if formal:
        lines += ["", "## 固定比较与消融", "",
                  "| 比较 | 成功率差（百分点） | 95%任务内配对区间（百分点） | 每任务配对数 |",
                  "| --- | ---: | --- | --- |"]
        comparisons = [("真实异步主比较", summary["main"])]
        comparisons += [("真实异步次要比较", item) for item in summary["secondary"]]
        comparisons += [(f"固定释放L={latency}（模拟条件）", item) for latency, item in summary["fixed_delay"].items()]
        comparisons += [(f"因子消融，{steps}步规则效应", item)
                        for steps, item in summary["ablation"]["rule_effects"].items()]
        comparisons += [(f"因子消融，{method} 5−10步效应", item)
                        for method, item in summary["ablation"].get("acceleration_effects", {}).items()]
        for label, item in comparisons:
            lo, hi = item["ci95"]
            lines.append(f"| {label}：{item['candidate']}−{item['baseline']} | {100*item['gain']:.2f} | "
                         f"[{100*lo:.2f}, {100*hi:.2f}] | {list(item['unit_counts'].values())} |")
        lines += ["", "均使用10,000次bootstrap、seed=20261005、三任务等权。次要及消融区间为逐项区间，未作多重比较校正。",
                  "固定释放使用首次三个状态，消融使用首次五个；跨阶段复用不增加独立样本，不合并阶段分数。",
                  "区间包含零不表示等效。10步可能因真实承诺覆盖不可行而全部早停，应作为可行域结果解释。", "",
                  "## 每任务完整分母与Wilson区间", "",
                  "| 阶段 | 任务 | 方法 | 步数 / K / L | 成功/回合 | 成功率 | Wilson 95% |",
                  "| --- | --- | --- | --- | --- | ---: | --- |"]
        for current in ("formal_async", "formal_delay", "formal_ablation"):
            for cell in summary["cells"][current]:
                lo, hi = cell["wilson95"]
                lines.append(f"| {current} | {cell['task']} | {cell['method']} | "
                             f"{cell['sampling_steps']} / {cell['period']} / {cell['latency']} | "
                             f"{cell['successes']}/{cell['n']} | {cell['success_rate']:.1%} | [{lo:.1%}, {hi:.1%}] |")
        lines += ["", "## 主矩阵成本、控制与失败", "",
                  "| 方法 | 总墙钟秒（含reset） | 成功任务/墙钟小时 | 成功完成秒 p50/p95 | 实际Hz p50/p95 | 请求数 / 生成秒 |",
                  "| --- | ---: | ---: | --- | --- | --- |"]
        for item in summary["groups"]["formal_async"]:
            subset = [row for row in selected if row["config"]["method"] == item["method"]]
            success_times = [row["wall_seconds"]+row["reset_wall_s"] for row in subset if row["success"]]
            times = np.percentile(success_times, [50, 95]).tolist() if success_times else None
            hz = np.percentile(item["actual_hz_episode_distribution"], [50, 95])
            throughput = 3600*sum(row["success"] for row in subset)/item["total_task_wall_s"]
            time_label = f"{times[0]:.2f}/{times[1]:.2f}" if times else "无成功回合"
            lines.append(f"| {item['method']} | {item['total_task_wall_s']:.2f} | {throughput:.2f} | "
                         f"{time_label} | {hz[0]:.2f}/{hz[1]:.2f} | {item['n_requests']} / {item['total_generation_s']:.2f} |")
        lines += ["", "| 方法 | 300tick / 短失败<50tick | 覆盖失败 | 迟到 / 过期 | 接续接受 / 拒绝 | 后备比例 / 最长 | 切换命令L2 p50/p95/p99 |",
                  "| --- | --- | ---: | --- | --- | --- | --- |"]
        for item in summary["groups"]["formal_async"]:
            switch = item["switch_command_l2_p50_p95_p99"]
            switch_label = "/".join(f"{value:.3f}" for value in switch) if switch else "无切换"
            lines.append(f"| {item['method']} | {item['full_300_ticks']} / {item['short_failures_under_50_ticks']} | "
                         f"{item['coverage_failures']} | {item['late_results']} / {item['expired_results']} | "
                         f"{item['catchup_accepted']} / {item['catchup_rejected']} | "
                         f"{item['fallback_fraction']:.1%} / {item['longest_consecutive_fallback']} | {switch_label} |")
        lines += ["", "成功完成时间以结果成功为条件，不能据此声称全回合加速。短失败和完整300tick周期/Hz分布在summary.json分开保存。",
                  "周期p50/p95/p99、工作超50ms比例、终止原因与每回合真实控制频率可由trials.csv、summary.json和事件原文重建。", "",
                  "## 请求时延与显存（主矩阵）", "",
                  "| 方法 | 实际请求 | endpoint p50/p95/p99秒 | worker已分配 / 保留峰值MiB |",
                  "| --- | ---: | --- | --- |"]
        for item in summary["request_distributions"]:
            if item["stage"] != "formal_async":
                continue
            latency = item["endpoint_p50_p95_p99_s"]
            label = "/".join(f"{value:.3f}" for value in latency) if latency else "无有效请求"
            lines.append(f"| {item['method']} | {item['n']} | {label} | "
                         f"{item['worker_peak_allocated_bytes']/2**20:.1f} / {item['worker_peak_reserved_bytes']/2**20:.1f} |")
        lines += ["", "显存为进程包含warmup的峰值；endpoint含队列/IPC，不含观测准备和控制端接收，后二者分布另存。",
                  "任务墙钟不包含共享模型加载、探针或落盘；作业总墙钟见summary.json/job_accounting，活跃GPU核总时间未知。"]
    lines += ["", "墙钟包含reset、策略启动、控制和尾请求，作业总成本另列于summary.json；成功回合完成时间具有条件选择偏差。", "",
              "## 开发持续诊断", "",
              "主候选与对照的持续资格分别判断。整体正式入场门槛要求两者都通过，不能因主候选通过而省略对照门槛。"]
    gate = summary["sustained_gate"]
    if gate:
        for method in ("RTC_STRICT", "RTC_CATCHUP", "HARD"):
            subset = [row for row in values["sustain"] if row["config"]["method"] == method]
            item = gate["methods"][method]
            lines.append(f"\n{method}：完整300tick {item['full_300_ticks']}/6，前缀激活 "
                         f"{sum(row['prefix_activations'] for row in subset)} 次，接受迟到接续 "
                         f"{sum(row['catchup_accepted'] for row in subset)} 次。")
    (DOCS/"REPORT.md").write_text((DOCS/"REPORT.md").read_text()+"\n".join(lines)+"\n")
