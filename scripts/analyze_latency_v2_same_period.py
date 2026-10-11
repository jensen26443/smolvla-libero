"""Offline, same-K development attribution; never changes formal configuration."""
import json

import numpy as np

from sim_vla.assets import project_root
from sim_vla.latency_v2.experiments import completed_results
from sim_vla.latency_v2.storage import publish


def main():
    root = project_root()
    baselines = completed_results(root/"runs/latency_v2_dev_baselines")
    candidates = completed_results(root/"runs/latency_v2_dev_candidates")
    selection = json.loads((root/"runs/latency_v2_dev_candidates/candidate_selection.json").read_text())
    cells, pairs = [], []
    for latency in (5, 10, 20):
        chosen = selection["latencies"][str(latency)]
        period = chosen["C_PREFIX"]["config"]["period"]
        configurations = [{"method": method, "period": period} for method in ("B_RAW", "B_SHIFT")]
        configurations += [chosen[method]["config"] for method in ("C_PREFIX", "C_BLEND")]
        indexed = {}
        for config in configurations:
            values = [row for row in baselines+candidates if row["latency"] == latency
                      and row["init_index"] < 3 and row["config"] == config]
            assert len(values) == 9
            rates = [np.mean([row["success"] for row in values if row["task"] == task])
                     for task in ("goal_bowl_plate", "goal_bowl_stove", "goal_bowl_cabinet_top")]
            cells.append({"latency": latency, "config": config, "common_native_inits": [0, 1, 2],
                          "episodes": len(values), "successes": sum(row["success"] for row in values),
                          "success_rate_equal_task": float(np.mean(rates)),
                          "mean_generation_s": float(np.mean([row["generation_seconds"] for row in values])),
                          "mean_requests_per_sim_second": float(np.mean([row["requests_per_sim_second"] for row in values]))})
            indexed[config["method"]] = {(row["task"], row["init_index"]): row for row in values}
        for left, right in (("B_SHIFT", "B_RAW"), ("C_PREFIX", "B_SHIFT"), ("C_BLEND", "C_PREFIX")):
            units = []
            for key in sorted(indexed[left]):
                a, b = indexed[left][key], indexed[right][key]
                assert a["seed"] == b["seed"]
                units.append({"task": key[0], "init_index": key[1], "seed": a["seed"],
                              "left_success": a["success"], "right_success": b["success"],
                              "difference": int(a["success"])-int(b["success"])})
            pairs.append({"latency": latency, "period": period, "left": left, "right": right,
                          "difference_equal_task": float(np.mean([row["difference"] for row in units])),
                          "units": units, "formal_hypothesis_test": False})
    directory = root/"docs/latency_v2"
    publish(directory/"development_same_period.json", {
        "scope": "common original development init0-2; same K/seed/task/L; offline descriptive analysis",
        "formal_trials_accessed": False, "parameter_selection_changed": False, "cells": cells, "pairs": pairs,
        "limits": "Small development samples and parameters chosen using development data; no independent significance claim."})
    rows = "\n".join(f"| {row['latency']} | {row['config']['period']} | {row['config']['method']} | "
                     f"{row['successes']}/{row['episodes']} | {row['success_rate_equal_task']:.1%} | "
                     f"{row['mean_requests_per_sim_second']:.3f} | {row['mean_generation_s']:.3f} |" for row in cells)
    differences = "\n".join(f"| {row['latency']} | {row['period']} | {row['left']} − {row['right']} | "
                            f"{100*row['difference_equal_task']:+.2f} pp |" for row in pairs)
    (directory/"DEVELOPMENT_SAME_PERIOD.md").write_text("""# 相同 K 的开发机制对照

仅使用三个原生开发 init0–2 的交集，所有方法共享 task/init/seed/L/K。
前缀和混合采用已经在开发阶段选定的参数；B_SHIFT 在完整 S1 网格中取相同 K，
用于剥离独立调优周期带来的混杂，不代替其正式最优周期或已冻结主比较。
以下是开发期描述性证据，样本少且参数来自同一开发集，不作独立显著性结论。

| L | K | 方法 | 成功/回合 | 等权任务成功率 | 请求/仿真秒 | 生成秒/回合 |
| --- | --- | --- | --- | --- | --- | --- |
"""+rows+"""

| L | K | 增量对照 | 配对等权任务成功率差 |
| --- | --- | --- | --- |
"""+differences+"""

独立调优时的结果差异不能全部归于时间偏移、前缀约束或切换平滑。
同 K 仍无法消除“成功更早导致更少调用”、状态轨迹分叉与模块联合作用；
请求频率与实际生成耗时一并列出。[完整逐 init 配对数据](development_same_period.json)。
复现：`.venv/bin/python scripts/analyze_latency_v2_same_period.py`，不调用模型或环境，不读取正式得分。
""")
    print(json.dumps({"cells": len(cells), "paired_contrasts": len(pairs), "formal_scores_accessed": False}))


if __name__ == "__main__":
    main()
