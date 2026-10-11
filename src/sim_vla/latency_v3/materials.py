"""Rebuild PLAN-05 reports and figures from completed, immutable trials."""
import csv
import itertools
from collections import Counter, defaultdict

import numpy as np

from ..latency_v2.integrity import file_sha256, inventory
from ..latency_v2.statistics import paired_bootstrap, wilson
from ..latency_v2.storage import publish
from ..tasks import TASK_BY_KEY
from .experiments import DOCS, ROOT, RUN, budget, ensure_frozen_unchanged, read, rows
from .schema import METHODS


def async_interval(values, candidate, baseline):
    lookup = {(r["task"], r["init_index"], r["config"]["method"]): int(r["success"]) for r in values}
    grouped = []
    for task in TASK_BY_KEY:
        indices = sorted({r["init_index"] for r in values if r["task"] == task})
        grouped.append(np.array([lookup[task, init, candidate]-lookup[task, init, baseline] for init in indices]))
    rng = np.random.default_rng(20261003)
    samples = sum(np.mean(g[rng.integers(0, len(g), (10000, len(g)))], axis=1)/3 for g in grouped)
    return {"candidate": candidate, "baseline": baseline, "gain": float(np.mean([g.mean() for g in grouped])),
            "ci95": np.percentile(samples, [2.5, 97.5]).tolist(), "units": "physical init within each fixed task"}


def table_row(values, method, L=None):
    subset = [r for r in values if r["config"]["method"] == method and (L is None or r["latency"] == L)]
    successes = sum(r["success"] for r in subset)
    return {"method": method, "latency": L, "episodes": len(subset), "successes": successes,
        "success_rate": successes/len(subset), "wilson95": wilson(successes, len(subset)),
        "requests_per_sim_second": float(np.mean([r["requests_per_sim_second"] for r in subset])),
        "generation_seconds_per_sim_second": float(np.mean([r["generation_seconds_per_sim_second"] for r in subset])),
        "fallback_fraction": float(np.mean([r["fallback_ticks"]/max(r["steps"], 1) for r in subset])),
        "mean_neutral_prefix_positions": float(np.mean([r["neutral_prefix_positions"] for r in subset])),
        "mean_control_hz": float(np.mean([r["actual_control_hz"] for r in subset])),
        "minimum_fraction_under_55ms": min(r["fraction_cycles_under_55ms"] for r in subset),
        "realtime_episodes_passed": sum(bool(r["realtime_20hz_passed"]) for r in subset),
        "termination_reasons": dict(Counter(r["termination_reason"] for r in subset))}


def validate_raw(spec, core, asynchronous):
    counters = {"steps": 0, "requests": 0, "commitments": 0, "files": 0}
    for stage, values, expected in (("formal_core", core, spec["core_trials"]),
                                    ("formal_async", asynchronous, spec["async_trials"])):
        if len(values) != len(expected):
            raise ValueError("formal denominator incomplete")
        physics = defaultdict(set)
        inv = []
        for row in values:
            physics[row["task"], row["init_index"]].add(row["initial_state_after_settle_sha256"])
        if any(len(v) != 1 for v in physics.values()):
            raise ValueError("paired methods did not restore identical physics")
        for trial in read(RUN/stage/"expected_trials.json"):
            directory = RUN/stage/"trials"/trial["key"]
            row = read(directory/"result.json")
            attempt = read(directory/"attempt.json")
            if attempt["started_unix"] <= spec["frozen_unix"] or attempt["finished_unix"] is None:
                raise ValueError("invalid formal freeze or attempt clocks")
            if not row["timing_provenance_complete"]:
                raise ValueError("incomplete event clocks")
            arrays = np.load(directory/"arrays.npz", allow_pickle=False)
            for name in arrays.files:
                if not np.isfinite(arrays[name]).all():
                    raise ValueError("nonfinite raw arrays")
            import json

            events = [json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines()]
            steps = [e for e in events if e["event"] == "step"]
            if [s["tick"] for s in steps] != list(range(row["steps"])):
                raise ValueError("missing or duplicate control step")
            if len(arrays["actions"]) != len(steps):
                raise ValueError("action/step length mismatch")
            for event in steps:
                np.testing.assert_array_equal(arrays["actions"][event["tick"]], event["action"])
                if event["source"] == "committed":
                    np.testing.assert_array_equal(arrays["actions"][event["tick"]],
                        arrays["prefixes"][event["request_id"], event["index"]])
                    counters["commitments"] += 1
            requests = read(directory/"requests.json")
            for request in requests:
                if request["prefix_length"] and request["remaining_old"] < request["prefix_length"]:
                    raise ValueError("formal prefix fabricated missing old actions")
                if not request["sampling"] or len({s["tick"] for s in request["sampling"].values()}) != 1:
                    raise ValueError("request modalities not fresh/aligned")
            counters["steps"] += len(steps)
            counters["requests"] += len(requests)
            evidence = inventory(directory, directory.rglob("*"))
            inv.append({"trial_key": trial["key"], "inventory": evidence})
            counters["files"] += len(evidence["files"])
        publish(DOCS/f"{stage}_inventory.json", inv)
    original = read(RUN/"audit/historical_source_inventory.json")
    if any(file_sha256(ROOT/p) != digest for p, digest in original.items()):
        raise ValueError("PLAN-04 source files changed")
    return {"passed": True, "counters": counters, "historical_sources_unchanged": True}


def build_figures(core, asynchronous, main):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = DOCS/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method in METHODS:
        points = [table_row(core, method, L) for L in (10, 20)]
        axes[0].plot([500, 1000], [p["success_rate"]*100 for p in points], marker="o", label=method)
        axes[1].scatter([p["generation_seconds_per_sim_second"] for p in points],
                        [p["success_rate"]*100 for p in points], label=method)
    axes[0].set(xlabel="Simulated release delay (ms)", ylabel="Success (%)", ylim=(-3, 103))
    axes[1].set(xlabel="Model generation seconds / simulation second", ylabel="Success (%)", ylim=(-3, 103))
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"P_HARD - B_RAW: {main['gain']*100:+.1f} pp; paired init CI")
    fig.tight_layout()
    fig.savefig(figures/"success_and_cost.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    methods = ("B_RAW", "P_HARD", "RTC_GUIDED")
    points = [table_row(asynchronous, m) for m in methods]
    axes[0].bar(methods, [p["success_rate"]*100 for p in points])
    axes[0].set(ylabel="Actual asynchronous success (%)", ylim=(0, 100))
    axes[1].bar(methods, [p["fallback_fraction"]*100 for p in points])
    axes[1].set(ylabel="Fallback ticks (%)", ylim=(0, 100))
    fig.tight_layout()
    fig.savefig(figures/"async_success_and_fallback.png", dpi=160)
    plt.close(fig)


def finish():
    spec = read(RUN/"frozen/experiment_spec.json")
    ensure_frozen_unchanged(spec)
    core, asynchronous = rows("formal_core"), rows("formal_async")
    main = paired_bootstrap(core, "P_HARD", "B_RAW", seed=20261003)
    rtc = paired_bootstrap(core, "RTC_GUIDED", "B_RAW", seed=20261003)
    rtc_hard = paired_bootstrap(core, "RTC_GUIDED", "P_HARD", seed=20261003)
    async_main = async_interval(asynchronous, "P_HARD", "B_RAW")
    async_rtc = async_interval(asynchronous, "RTC_GUIDED", "B_RAW")
    cells = [table_row(core, m, L) for L, m in itertools.product((10, 20), METHODS)]
    async_cells = [table_row(asynchronous, m) for m in ("B_RAW", "P_HARD", "RTC_GUIDED")]
    classification = "positive_main_candidate" if main["gain"] > 0 and main["ci95"][0] > 0 else (
        "limited_or_uncertain_main_gain" if main["gain"] > 0 else "negative_main_no_improvement")
    if main["gain"] <= 0 and rtc["gain"] > 0:
        classification += "_rtc_reference_gain"
    validation = validate_raw(spec, core, asynchronous)
    from .metrics import detailed_metrics

    details = detailed_metrics(core, asynchronous)
    summary = {"spec_sha256": spec["spec_sha256"], "classification": classification,
        "main": main, "rtc_vs_raw_secondary": rtc, "rtc_vs_hard_secondary": rtc_hard,
        "async_main": async_main, "async_rtc_secondary": async_rtc, "core": cells, "async": async_cells,
        "counts": {"core": len(core), "async": len(asynchronous)}, "raw_validation": validation,
        "implementation_complete": True, "experiment_complete": True,
        "timing_provenance_complete": all(r["timing_provenance_complete"] for r in core+asynchronous),
        "realtime_gate_passed": all(r["realtime_20hz_passed"] for r in asynchronous),
        "remaining_budget": budget().remaining()}
    summary["algorithm_result"] = classification
    summary["wall_throughput_scope"] = details["wall_clock_scope"]
    summary["wall_throughput"] = details["groups"]
    DOCS.mkdir(parents=True, exist_ok=True)
    publish(DOCS/"summary.json", summary)
    fields = ("task", "init_index", "latency", "method", "period", "success", "termination_reason", "steps",
              "n_requests", "generation_seconds", "requests_per_sim_second", "generation_seconds_per_sim_second",
              "fallback_ticks", "neutral_prefix_positions", "actual_control_hz", "fraction_cycles_under_55ms", "backend")
    with (DOCS/"formal_trials.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in core+asynchronous:
            writer.writerow({k: r["config"].get(k) if k in ("method", "period") else r.get(k) for k in fields})
    build_figures(core, asynchronous, main)
    report = ["# PLAN-05 正式报告", "", f"结论：**{classification}**。", "",
        (f"预选主比较 P_HARD−B_RAW：{main['gain']*100:+.2f}pp，任务内物理 init 配对 95% CI "
         f"[{main['ci95'][0]*100:.2f},{main['ci95'][1]*100:.2f}]pp；10,000 次重采样，两个时延一起保留。"),
        f"RTC−RAW 次要比较：{rtc['gain']*100:+.2f}pp，CI [{rtc['ci95'][0]*100:.2f},{rtc['ci95'][1]*100:.2f}]pp。",
        "", f"核心 {len(core)}、真实异步 {len(asynchronous)}；每任务核心独立初始化 {spec['core_n']} 个，异步 {spec['async_n']} 个。",
        "三项既有支持任务、真实 placement draws；训练状态重叠未知，不主张未见任务或跨机器人泛化。",
        "", "## 固定时延核心结果", "", "| L | 方法 | 成功/分母 | 请求/仿真秒 | 生成秒/仿真秒 | 后备比例 |",
        "| --- | --- | --- | --- | --- | --- |"]
    for p in cells:
        report.append(f"| {p['latency']} | {p['method']} | {p['successes']}/{p['episodes']} | "
            f"{p['requests_per_sim_second']:.3f} | {p['generation_seconds_per_sim_second']:.3f} | {p['fallback_fraction']:.1%} |")
    report += ["", "## 真实异步", "", "| 方法 | 成功/分母 | 平均 Hz | 最低回合≤55ms比例 | 达标回合 |",
               "| --- | --- | --- | --- | --- |"]
    for p in async_cells:
        report.append(f"| {p['method']} | {p['successes']}/{p['episodes']} | {p['mean_control_hz']:.2f} | "
            f"{p['minimum_fraction_under_55ms']:.1%} | {p['realtime_episodes_passed']}/{p['episodes']} |")
    report += ["", "## 机制与边界", "",
        "本轮共同采用静止原生块启动、on_request 新鲜双相机与原物理/动作语义，不能与旧模拟分数直接相减。",
        "硬前缀和 RTC 的覆盖检查避免缺失旧动作的中性填充；迟到和不可持续延迟仍按方法失败计分。",
        "RTC 使用安装版本 guidance 与项目锁定执行器，包含原生 padding guidance；未修改上游公式，不是论文全系统复现。",
        "清洁前缀训练分布偏移、锁定反馈不足与夹爪时序仍属机制假设；结构问题减少不保证任务改善。",
        "启动及尾部生成成本保留；请求和生成成本按仿真秒报告，不预设同预算。生成秒是同步测量墙钟，不是 GPU kernel 利用率。",
        "帧按仿真 20fps 编码，真实采样时刻和帧龄另存；实时门槛通过也不代表持续精确 20Hz。",
        "短失败回合在成功率分母中保留；其 Hz 和单位仿真时间成本不证明持续运行，逐方法终止原因与短失败数见 detailed_metrics.json。",
        "", "## 验真与材料", "", f"原始检查：{validation['counters']}；历史 v2 源码哈希保持一致。",
        f"共享剩余预算 {summary['remaining_budget']}。全部方法参数与 core/async 清单在首次正式得分前同时冻结。",
        "完整分母、Wilson 区间、终止原因和配对区间见 summary.json；逐 trial 数据见 formal_trials.csv。",
        "逐任务成功率与 Wilson 区间见 TASK_RESULTS.md；endpoint、成功时间、切换命令与前缀误差见 detailed_metrics.json。",
        "复现与恢复见 RUNBOOK.md，开发同 K 与旧调度材料见 SAME_PERIOD.md / MECHANISM.md。"]
    (DOCS/"REPORT.md").write_text("\n".join(report)+"\n")
    (DOCS/"VALIDATION.md").write_text("# PLAN-05 验收\n\n```json\n"+__import__("json").dumps(summary,
        indent=2, ensure_ascii=False)+"\n```\n")
    from .delivery import deliver

    deliver(summary)
    publish(RUN/"completion.json", summary)
    return summary
