"""Review a real matched development failure pair without resets or policy calls."""
import json

import numpy as np

from sim_vla.assets import project_root
from sim_vla.latency_v2.experiments import completed_results
from sim_vla.latency_v2.mechanism import trial_metrics
from sim_vla.latency_v2.plots import timeline
from sim_vla.latency_v2.storage import publish, trial_key


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = project_root()
    output = root/"docs/latency_v2/development_pilot"
    output.mkdir(parents=True, exist_ok=True)
    baseline_rows = completed_results(root/"runs/latency_v2_dev_baselines")
    candidate_rows = completed_results(root/"runs/latency_v2_dev_candidates")
    # Check the requested success/failure combination over every common dev init.
    positive_pairs = []
    for baseline in baseline_rows:
        if baseline["config"]["method"] != "B_RAW" or baseline["success"]:
            continue
        for candidate in candidate_rows:
            fields = ("task", "init_index", "seed", "latency")
            if (all(baseline[key] == candidate[key] for key in fields)
                    and baseline["config"]["period"] == candidate["config"]["period"]
                    and candidate["success"]):
                positive_pairs.append([baseline, candidate])
    selection = json.loads((root/"runs/latency_v2_dev_candidates/candidate_selection.json").read_text())
    chosen = selection["latencies"]["20"][selection["C_SELECTED"]]["config"]
    baseline = next(row for row in baseline_rows if row["task"] == "goal_bowl_plate"
                    and row["init_index"] == 0 and row["latency"] == 20
                    and row["config"] == {"method": "B_RAW", "period": chosen["period"]})
    candidate = next(row for row in candidate_rows if row["task"] == baseline["task"]
                     and row["init_index"] == 0 and row["latency"] == 20 and row["config"] == chosen)
    reviewed = positive_pairs[0] if positive_pairs else [baseline, candidate]
    records, raw = [], []
    fields = ("task", "init_index", "seed", "latency", "profile", "config")
    for row, stage in zip(reviewed, ("dev_baselines", "dev_candidates")):
        trial = {key: row[key] for key in fields}
        directory = root/("runs/latency_v2_"+stage)/"trials"/trial_key(trial)
        metrics = trial_metrics(directory)
        method = row["config"]["method"]
        target = output/(method+"_timeline.png")
        timeline(directory, target)
        records.append({"trial": trial, "result": row, "metrics": metrics,
                        "source": str(directory.relative_to(root)),
                        "timeline": str(target.relative_to(root))})
        raw.append((directory, json.loads((directory/"diagnostics.json").read_text())))
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for record, (_, diagnostics) in zip(records, raw):
        obj = np.asarray([row["object_position"] for row in diagnostics])
        eef = np.asarray([row["eef_position"] for row in diagnostics])
        label = record["trial"]["config"]["method"]
        axes[0].plot(obj[:, 2]-obj[0, 2], label=label)
        axes[1].plot(np.linalg.norm(obj-eef, axis=1), label=label)
    axes[0].set(ylabel="Object height change (m)", title="Actual development trajectories; offline diagnostics")
    axes[1].set(ylabel="End-effector/object distance (m)", xlabel="Control tick")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output/"paired_physical_trace.png", dpi=160)
    plt.close(figure)
    # Only display cached candidate frames; original raw camera arrays remain intact.
    directory = raw[1][0]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, name in zip(axes, ("start", "middle", "end")):
        axis.imshow(plt.imread(directory/(name+".png")), origin="lower")
        axis.set_title("Cached candidate frame: "+name)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output/"candidate_cached_frames.png", dpi=160)
    plt.close(figure)
    report = {"scope": "development only; existing real arrays/events, no new resets or model calls",
              "matched_baseline_failure_candidate_success_pairs": len(positive_pairs),
              "selection": "first matched success pair if present; otherwise plate/init0/L20 and selected candidate",
              "same_native_init_seed_latency_period": True, "records": records}
    publish(output/"review.json", report)
    lines = []
    for row in records:
        result, metrics = row["result"], row["metrics"]
        lines.append(f"| {result['config']['method']} | {result['success']} | {result['steps']} | "
                     f"{metrics['fallback_fraction']:.1%} | {metrics['committed_old_buffer_shortage_ticks']} | "
                     f"{metrics['max_executed_prefix_violation']:.3g} |")
    (root/"docs/latency_v2/DEVELOPMENT_PILOT.md").write_text("""# S3 实际开发机制对照

完整扫描的共同 init/seed/L/K 中，B_RAW 失败而候选成功的对数为 """+str(len(positive_pairs))+"""。
按计划审查真实失败对，不虚构改善样例：plate，原生开发 init0，L20，K50，
候选为开发期选定 C_PREFIX/m2。它们均为真实模型与物理环境 rollout，未新开 reset 或调用模型。

| 方法 | 成功 | 物理步数 | 中性后备比例 | 承诺内旧缓冲不足 ticks | 实际前缀最大违约 |
| --- | --- | --- | --- | --- | --- |
"""+"\n".join(lines)+"""

[原始记录与数值](development_pilot/review.json)、[配对物理轨迹](development_pilot/paired_physical_trace.png)、
[基线动作时间线](development_pilot/B_RAW_timeline.png)、[候选动作时间线](development_pilot/C_PREFIX_timeline.png)、
[候选缓存画面](development_pilot/candidate_cached_frames.png)。截图按固定 start/middle/end 抽取，
不是可信接触阶段标签；该轮 K50 基线没有录制相机帧，不能事后制造它的画面。
正式配对视频将为两种方法同时记录，不重复这次开发轨迹作为正式证据。

H50/K50/c22 下，候选每次正常交接的旧有效块末端为 t_obs+50，
到下一承诺窗口会缺少 22 个旧动作；日志直接记录这些中性填充。
这是协议与有限动作块视野的结构关系，不是对象掉落或抓取失败的独立因果证明。
L5/K40 的候选没有该缓冲不足仍多数失败，说明缓冲耗尽不能解释所有失败。
干净前缀逐步投影真实影响模型后缀且没有执行违约；投影下的训练分布偏移、
锁定期内反馈修正受限和夹爪时序仍是机制假设，未通过额外正式调参证明。
""")
    print(json.dumps({"positive_pairs": len(positive_pairs), "review": str(output/"review.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
