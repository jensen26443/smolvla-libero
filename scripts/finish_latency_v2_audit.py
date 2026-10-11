#!/usr/bin/env python3
"""Rebuild the S0 delivery from completed real audit artifacts only."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    directory = root / "runs/latency_v2_audit"
    report = json.loads((directory / "audit.json").read_text())
    if report.get("last_completed_probe") != "complete":
        raise SystemExit("S0 is incomplete; refusing to publish final materials")
    if not report["interface_gate_passed"] or not report["legacy_preserved"]:
        raise SystemExit("interface/integrity audit did not pass")
    docs = root / "docs/latency_v2"
    docs.mkdir(exist_ok=True)
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = report["model_probe"]
    envs = report["env_probes"]
    tasks = report["initializations"]["tasks"]
    samples = np.asarray([s["end_to_end_s"] for s in model["timing_samples"]])
    fig, axis = plt.subplots(figsize=(6, 3.5))
    axis.hist(samples * 1000, bins=12)
    axis.set(xlabel="End-to-end inference (ms)", ylabel="Calls", title="S0: 10 warmup + 50 measured calls")
    fig.tight_layout()
    fig.savefig(figures / "inference_latency.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 3.5))
    axis.boxplot([np.asarray(v["env_step_seconds_raw"]) * 1000 for v in envs.values()],
                 tick_labels=[k.removeprefix("goal_bowl_") for k in envs])
    axis.axhline(55, color="red", linestyle="--", label="20 Hz acceptance limit (55 ms)")
    axis.set(ylabel="Environment step with observation (ms)", title="S0 environment-only control cost")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "env_step_latency.png", dpi=160)
    plt.close(fig)
    with (directory / "initialization_eligibility.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "index", "state_sha256", "historically_quarantined"])
        writer.writeheader()
        for task, info in tasks.items():
            for index, digest in enumerate(info["row_sha256"]):
                writer.writerow({"task": task, "index": index, "state_sha256": digest,
                                 "historically_quarantined": index in info["quarantined_indices"]})
    with (directory / "inference_timing.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(model["timing_samples"][0]))
        writer.writeheader()
        writer.writerows(model["timing_samples"])

    from sim_vla.latency_v2.initializations import official_reset_history
    from sim_vla.latency_v2.integrity import file_sha256
    from sim_vla.tasks import TASK_BY_KEY

    historical = {"evidence_kind": "reconstruction from installed source and prior success logs", "sources": {}, "tasks": {}}
    scored_union = {TASK_BY_KEY[task].suite_task_id: set() for task in tasks}
    for name in ("e0_formal", "e1_formal", "e2_020pct_formal", "e2_050pct_formal", "e5_formal"):
        path = root / "results" / name / "eval_info.json"
        original = json.loads(path.read_text())
        historical["sources"][name] = {"path": str(path), "sha256": file_sha256(path), "tasks": {}}
        for task in original["per_task"]:
            reconstructed = official_reset_history(task["metrics"]["successes"], 50)
            historical["sources"][name]["tasks"][str(task["task_id"])] = reconstructed
            scored_union[task["task_id"]].update(e["init_index"] for e in reconstructed["scored_starts_reconstructed"])
    for task in tasks:
        seen = scored_union[TASK_BY_KEY[task].suite_task_id]
        historical["tasks"][task] = {
            "scored_union_reconstructed": sorted(seen),
            "preferred_30_49_scored_conflicts_reconstructed": sorted(seen & set(range(30, 50))),
        }
    historical["termination_source"] = {
        "path": str(root / ".venv/lib/python3.12/site-packages/libero/libero/envs/bddl_base_domain.py"),
        "semantics": "BDDLBaseDomain.step overrides done with _check_success(); failure reaches evaluator timeout",
    }
    (directory / "historical_official_init_reconstruction.json").write_text(json.dumps(historical, indent=2))

    timing_rows = "\n".join(
        f"| {label} | {model['timing'][key]['p50'] * 1000:.3f} | {model['timing'][key]['p95'] * 1000:.3f} |"
        for key, label in (("preprocess_s", "预处理（扣除 H2D）"), ("h2d_s", "H2D device processor"),
                           ("generation_s", "模型生成（含噪声创建）"),
                           ("postprocess_d2h_s", "后处理及 D2H"), ("end_to_end_s", "端到端结果可消费"))
    )
    env_rows = "\n".join(
        f"| {task} | {v['env_step_seconds']['p50'] * 1000:.1f} | {v['env_step_seconds']['p95'] * 1000:.1f} |"
        for task, v in envs.items()
    )
    init_rows = "\n".join(
        f"| {task} | {v['shape'][0]}×{v['shape'][1]} | {len(v['quarantined_indices'])} | {len(v['eligible_native_indices'])} |"
        for task, v in tasks.items()
    )
    results = [attempt["result"] for attempt in report["rollout_attempts"]]
    trial_rows = "\n".join(
        f"| {r['task']} | L{r['latency_ticks']}, K={r['period_ticks']} | {r['success']} | {r['steps']} | "
        f"{r['n_requests']} | {r['wall_seconds']:.1f} | {r['failure_type'] or '—'} |"
        for r in results
    )
    model_hash = next(x["sha256"] for x in report["checkpoint"]["files"] if x["path"] == "model.safetensors")
    budget = report["budget"]
    old_directory = directory / "trials/goal_bowl_stove__B_RAW_k40__L5__i00"
    events = [json.loads(line) for line in (old_directory / "events.jsonl").read_text().splitlines()]
    ticks = [e for e in events if "tick" in e]
    requests = [e["request"] for e in events if "request" in e]
    activations = {e["activated_plan_id"]: e["tick"] for e in ticks if e["activated_plan_id"] is not None}
    explicit = []
    for request in requests:
        explicit.append({
            "request_id": request["request_id"], "t_obs": request["input_capture_tick"],
            "t_req": request["request_tick"], "t_ready_simulated": request["completion_tick"],
            "t_act": activations.get(request["request_id"]),
            "actual_gpu_call_wall_seconds": request["inference_seconds"],
            "activation_start_index": 0 if request["request_id"] in activations else None,
        })
    (directory / "legacy_request_timeline.json").write_text(json.dumps(explicit, indent=2))

    audit_text = f"""# PLAN-04 S0 实测审计

2026-10-01。状态：**接口审计通过；正式初始化资产阻塞，G0 未通过。**
主记录：[audit.json](../../runs/latency_v2_audit/audit.json)。这是实际执行的 S0 交付，S1–S7 未开始。

## 仓库与资产

实际仓库 `{root}`；RTX 3090 与已有 `.venv` 可用。未发现适用 AGENTS.md；Git 元数据不可用，未新建或重置 Git。
现有源码、配置、清单、文档及 E1 文件在审计前后逐文件哈希相同；历史 runs/results 保留。
E1：`{report['config']['checkpoint']}`；训练配置与所有 processor safetensors 已核验，模型真实加载并前向。
完整 `model.safetensors` SHA-256：`{model_hash}`。
VLM/tokenizer 本地缓存 revision 为 `7b375e1b73b11138ff12fe22c8f2822d8fe03467`，完整文件摘要见 supplemental_provenance.json。
LIBERO 资产 revision 来源记录为 `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`；本轮另保存资产实际内容摘要，未下载或替换。

## 动作及模型接口

两相机为 `image`/`image2`，256×256×3 uint8；官方 processor 翻转图像并构造 8 维 state。
模型配置内部 resize 为 512×512；没有修改分辨率或采样步数。原始输出 `(1,50,7)`，模型内部 padding `(1,50,32)`。
环境 action 为 7 维 Box[-1,1]：前三维是基坐标系平移增量，后三维是 axis-angle 旋转增量，末维为夹爪命令。
OSC 的输出缩放为平移每维 ±0.05m、旋转每维 ±0.5rad；旋转以 `R_delta @ R_current` 更新目标，不能按不明角度表示直接插值。
Panda 夹爪通过 sign(action) 积分，-1 开、+1 闭；0 保持 actuator 当前值。后续混合必须保留事件语义。
旧 fallback 为零机械臂增量及 -1 夹爪，可能打开夹爪；PLAN-04 的当前目标保持规则尚未实现，不能视为已完成。

整块/单步 postprocessor 一致、同种子输出逐位一致；归一化往返最大误差 `{model['action_roundtrip_max_abs_error']:.3g}`。
7 个真实维度有效，其余 25 个 padding 维度为零；正负夹爪命令往返符号一致。flow 积分 10 步，dt=-0.1。
这里验证的是原 sampler 和处理器；C_PREFIX/C_BLEND 尚未实现或评估。

## 当前机器推理计时

10 次 warmup 后 50 次有效调用；仅审计路径同步 GPU，计时原值保存在 inference_timing.csv。
下表单位为 ms；同步 wall time 包含 Python 调度，不能当作 GPU kernel 利用时间。

| 阶段 | p50 | p95 |
| --- | ---: | ---: |
{timing_rows}

显存峰值 {model['maximum_cuda_allocated_bytes'] / 2**30:.3f} GiB（torch allocated）；torch 线程数 {model['torch_num_threads']}。
端到端 p95 相当于 {int(np.ceil(model['timing']['end_to_end_s']['p95'] / .05))} 个目标控制 tick。

## 环境与开发 rollout

三个任务各两次 init 0/seed 42 reset，状态和首图摘要一致；各真实 step 20 次，均使用现有相对控制接口。

| 任务 | step p50 ms | step p95 ms |
| --- | ---: | ---: |
{env_rows}

仅环境 step 已超过 50ms；当前观测/仿真路径无法通过稳定 20Hz 验收。这里没有真实异步控制频率结果。
以下四次为真实 LIBERO rollout，全部开发 init 0、seed 42。L 是模拟释放时延；R0 每 tick 重规划。
单个开发初始化不能估计任务总体成功率，也不能据此删除第三任务。

| 任务 | 条件 | success | steps | 请求数 | 墙钟秒 | failure |
| --- | --- | --- | ---: | ---: | ---: | --- |
{trial_rows}

## 旧执行器真实时间线

请求周期从上次实际提交 tick 计时；忙时不积累 backlog，空闲后按 elapsed>=K 判断。每 tick 仅一次外层 env.step。
开发 L5/K40 真实记录在 [events.jsonl](../../runs/latency_v2_audit/trials/goal_bowl_stove__B_RAW_k40__L5__i00/events.jsonl)、
[timeline.png](../../runs/latency_v2_audit/trials/goal_bowl_stove__B_RAW_k40__L5__i00/timeline.png) 与 legacy_request_timeline.json。
首请求 t_obs=t_req=0，模拟 ready=5，激活=5，从新块索引 0 消费；ticks 0–4 使用 fallback。
下次提交 40、ready/activate 45，重新从新块索引 0 开始，等待期仍消费旧块。GPU 真实调用时间与模拟释放 tick 分开记录。
终止成功后旧 LiberoEnv 会额外 reset；本轮成功 rollout 的 native cursor 实测由 1 变为 2，确认了隐式初始化消费。

## 正式初始化阻塞

| 任务 | 原生数组 | 历史应隔离行数 | 可核验未暴露行数 |
| --- | --- | ---: | ---: |
{init_rows}

历史 E1 官方评测每任务 30 episodes，batch_size=1；每次 rollout 入口 reset，成功终止又在 LiberoEnv.step 内 reset。
连续成功时 scored init 依次是 0,2,4,…，内部 reset 消费 1,3,5,…；50 行数组随后 modulo 回绕。
每任务 30 次入口 reset 加至少 29 次成功 reset 已超过 50，整个原生池均需隔离。stove 30/30，其他任务 29/30。
这些是当前固定版本源码和历史成功日志的重建，不冒充历史逐次 init telemetry。
进一步重建 E0/E1/E2-20%/E2-50%/E5 的官方评分起点，三个任务的 30–49 均在历史评分起点并集中。
这项结论无需将未评分内部 reset 算作正式 episode；逐次重建和源文件摘要在 historical_official_init_reconstruction.json。
PLAN-03 的显式 init 10–29 记录另行纳入隔离。逐行物理状态 SHA-256 在 initialization_eligibility.csv。
30–49 全部冲突。按相同原生状态换 seed/index 仍会被实际数组摘要校验器拒绝；拒绝证据在 rejected_native_manifest/check.json。
本轮未 reset、推理或调参任何新正式状态，也未创建 experiment_spec/正式矩阵。

## 历史证据差异与预算

800 原始 trial 全部存在、unique=800、attempt 全 1。实际为 R0 40 + 主条件 480 + 长周期消融 160 + 补充 120。
旧 REPORT 将消融描述为 80，冻结 notes 写 planned total 760；本轮记录差异，未更改原文件。
旧 checkpoint 指纹在累计 64MiB 限制处跳过 1.2GB 权重，本轮全内容哈希补足证据。
旧配对 bootstrap 合并 task/latency/init 条件；PLAN-04 必须按任务内 init 重采样并保留该 init 所有时延记录。
历史官方按 episode ordinal 声称 dev/final 物理状态隔离不足，应作为限制保留，不能当作新盲测来源。

实际 scored rollout {budget['scored_rollout_attempts']} 次；未评分 reset probe 按 {budget['unscored_reset_probes']} 次保守计入。
合计 {budget['attempts_charged_conservatively']}/1800，剩余 {budget['remaining']}；未评分审计失败无方法第二次机会。
attempt01 在首次 reset 前因非交互配置顺序失败；attempt02 做两次开发 reset 后因 wrapper 字段路径失败，原报告/日志均保留。
53 项测试通过，ruff 通过。恢复条件和实际命令见 RECOVERY.md / RUNBOOK.md。
"""
    (docs / "AUDIT.md").write_text(audit_text)
    (docs / "REPORT.md").write_text(f"""# PLAN-04 当前交付结果

**S0 接口与 GPU 审计已完成；缺少可核验的新正式物理初始化资产，G0 未通过。S1–S7 尚未执行。**
这不是方法负结果；候选和强基线还没有进入新评测，不能声称提升或无提升。

实际 E1、处理器和三个 LIBERO 任务可用，原资产保留。已实现并运行独立审计 CLI、完整模型哈希与物理状态隔离校验，53 项测试通过。
10 次预热、50 次实测：端到端推理 p50={model['timing']['end_to_end_s']['p50']:.3f}s，p95={model['timing']['end_to_end_s']['p95']:.3f}s。
环境 step p50 约 201–205ms，20Hz 真实异步验收存在环境开销限制；目前没有异步实验结果。

三个原生初始化文件均只有 50 行。历史官方 E1 的入口 reset 加成功后的内部 reset 使整个池暴露；30–49 不能作为本轮盲测。
当前每任务可核验 fresh states=0，需要有来源记录、与现有任务兼容且未历史暴露的真实状态数组。
没有用模拟分数、换 seed、换编号或删掉困难条件代替正式实验。

本轮 {len(results)} 次真实开发 rollout 的条件、步数、成本和失败原因见 [AUDIT.md](AUDIT.md)；完整记录在 [audit.json](../../runs/latency_v2_audit/audit.json)。
历史 800-episode 证据完整；矩阵文字、权重哈希和初始化语义的差异单独记录，未覆盖原成果。
预算保守计费 {budget['attempts_charged_conservatively']}/1800，剩余 {budget['remaining']}。恢复所缺资产与验证方式见 [RECOVERY.md](RECOVERY.md)。

当前可交付材料为审计报告、开发时间线、计时图表及可运行校验代码。
正式差值/区间、K 扫描、前缀 sampler、机制消融、真实异步、演示视频及新简历成果均未取得，不填入虚构数字。
""")
    (docs / "RUNBOOK.md").write_text("""# PLAN-04 S0 已验证命令

工作目录 `sim_vla/`；复用现有 `.venv`。本页不把后续未实现命令写成可运行入口。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/sim_vla/latency_v2 tests/test_latency_v2_isolation.py scripts/finish_latency_v2_audit.py
```

pytest 自动加载 ROS 插件会因 lark 缺失失败；关闭插件自动加载后项目测试全部通过，未重装依赖。

实际执行的 GPU 审计：

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_HUB_DISABLE_PROGRESS_BARS=1 \
.venv/bin/python -m sim_vla.latency_v2.cli audit --config configs/latency_v2/audit.yaml
```

输出 `runs/latency_v2_audit/audit.json`；本次完整审计退出 2 表示初始化资产门未通过。已有 run_dir 不会被覆盖；再审计需复制配置并指定新目录。
两次早期接入失败原样保存在 `runs/latency_v2_audit_attempt01`、`attempt02`。GPU 计时和 rollout 均来自第三次运行。

物理状态复用拒绝检查（实际执行，退出 2）：

```bash
.venv/bin/python -m sim_vla.latency_v2.cli check-formal-inits \
  --manifest runs/latency_v2_audit/rejected_native_manifest/manifest.json \
  --audit runs/latency_v2_audit/audit.json
```

时间线重建与材料生成（实际执行）：

```bash
.venv/bin/python -m sim_vla.latency_v2.cli timeline \
  --trial-dir runs/latency_v2_audit/trials/goal_bowl_stove__B_RAW_k40__L5__i00
.venv/bin/python scripts/finish_latency_v2_audit.py
```

材料生成器只接受已完整完成且接口/资产完整性检查通过的 S0 记录；不会把中间报告发布成最终材料。
图表在 `runs/latency_v2_audit/figures/`，CSV 与 explicit legacy request timeline 在 audit 目录。
""")
    (docs / "RECOVERY.md").write_text("""# S0 阻塞与恢复条件

当前缺少的是可核验未被历史评测暴露的任务兼容物理初始化资产。三个原生池各 50 行，已全部隔离；换采样 seed 不能增加新状态。

恢复默认矩阵需要每个任务 20 个有来源记录的新状态，每行对应现有 MuJoCo 模型的 79 维 flattened state。
资产必须包含真实数组、产生方式/源 revision、历史使用审查及文件 SHA-256；不能只提供新的编号或 seed。
如果从环境重新采样，需要单独记录采样分布、实际数组和 reset 合法性，明确评测范围变化，再在任何正式结果之前冻结；它不是原生 init 30–49 的盲测。
当前交付没有这份经审计的新资产，未自行换测试分布或复用已暴露状态。

`check-formal-inits` 支持非 pickle `.npy` bank，manifest 格式：

```json
{
  "format": "latency-v2-init-bank-v1",
  "provenance": {"source": "实际来源、版本、采样/使用审查"},
  "tasks": {
    "goal_bowl_plate": {
      "states_path": "真实数组文件.npy",
      "states_sha256": "完整文件摘要",
      "indices": [0, 1]
    }
  }
}
```

上例只展示结构，实际默认检查必须覆盖全部三个任务、各 20 行。
相对数组路径以 manifest 所在目录解析；检查器核验实际文件内容、维度、有限值、索引边界、重复物理状态和历史摘要冲突。
通过摘要门还需要任务兼容性/来源与 reset 合法性审计；检查器不宣称通过摘要即完成 benchmark 有效性。

恢复后复用已有 S0 证据，在新 run_dir 完成资产门；遵循原 PLAN-04 的 G0→G1→G2→G3→G4 顺序，正式前冻结所有参数。
新执行器必须消除终止内部 reset 导致的隐式下一状态读取，并在显式 reset 时拒绝 modulo 越界。
控制 loop 需记录实际周期；当前环境单步约 0.2s，应先解决/量化观测与仿真开销，不能把模拟 20Hz 写成真实稳定 20Hz。
不存在后台正式队列，当前不会消耗更多 episode 预算。
""")
    (docs / "PROGRESS.md").write_text(f"""# PLAN-04 执行进度

2026-10-01：实际仓库、RTX 3090、E1 和 processor 已找到；复用环境，无适用 AGENTS.md，无可用 Git 元数据。
既有 47 测试重跑通过；新增 6 项物理初始化隔离回归测试，总计 53 通过，ruff 通过。
S0 checkpoint/全权重/processor/资产摘要、三任务 reset/step、10 warmup+50 GPU 调用与 4 次开发 rollout 已完成。
历史 800 trials 分母无遗漏/重复；记录旧矩阵文字和哈希覆盖差异，原证据未修改。

| 检查点 | 状态 | 证据 |
| --- | --- | --- |
| S0 接口审计 | 通过 | audit.json model_probe/env_probes + 开发 trace |
| G0 全部必要资产 | 阻塞 | 每任务 native fresh=0；初始化摘要门拒绝 30–49 |
| S1/G1 强基线扫描 | 未开始 | 无新扫描结果 |
| S2/S3/G2 候选方法 | 未开始 | 无前缀 sampler/消融结果 |
| S4/G3 真实异步 | 未开始 | 环境 step 超过 20Hz 目标的诊断证据已保存 |
| S5/G4 正式冻结 | 未开始 | 无新正式 spec、manifest 或 trial |
| S6/S7/G5 正式分析材料 | 未开始 | 当前只有 S0 审计交付，未伪造算法结论 |

attempt01：非交互配置在 import 前未建立，0 rollout、0 显式 reset；已修复。
attempt02：读取 wrapper 控制频率层级错误，0 rollout、2 未评分开发 reset；已修复并计预算。
attempt03：完整 S0 执行；最终退出 2 表示必要正式初始化资产不足。
预算保守计费 {budget['attempts_charged_conservatively']}/1800，剩余 {budget['remaining']}；开发 init 均为 0，无正式集调参。
""")
    (docs / "DECISIONS.md").write_text("""# PLAN-04 已执行决定

- 2026-10-01：将 S0 代码放在 `sim_vla.latency_v2`，复用旧 env/policy/runner/timeline；不改 PLAN-03 的执行语义或产物。
- Git 元数据不可用，记录全内容摘要；未初始化空 Git、重置文件或覆盖旧分支。
- 使用现有 E1/venv/cache，模型以离线缓存加载；没有重训、重装 CUDA/PyTorch 或更新模型 revision。
- 隔离依据是物理状态数组摘要，包含成功后隐式 reset 暴露；不将 episode ordinal/new seed 视作未见物理状态。
- init 30–49 资产门拒绝后，完成独立 S0 检查并停止 G0 之后实验；没有用历史分数填充新正式矩阵。
- 严格保留历史报告；800 trial 分母、长周期消融数量、checkpoint 指纹截断与旧官方 init 语义的差异另记 AUDIT。
- 环境 wall step 超过控制周期是实测诊断；不提前声称异步达到 20Hz。后续必须报告实际 wall frequency。
- 未评估候选即不作正/负结果判断，也不产出带假提升的简历和演示。两次审计接入失败保留完整日志。
""")
    print(f"Rebuilt S0 materials: {docs}")


if __name__ == "__main__":
    main()
