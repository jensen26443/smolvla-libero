"""Publish evidence-based materials only after all frozen comparisons finish."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from ..assets import project_root
from .experiments import completed_results
from .integrity import inventory, source_inventory
from .layout import load_layout
from .mechanism import analyze
from .plots import build_figures
from .statistics import summarize
from .storage import Budget, publish, trial_key


def build_videos(summary: dict, output: Path, *, spec_path: Path | None = None) -> dict:
    """Fixed min-init pairs and an 80 s demo with a labeled failure illustration."""
    from PIL import Image, ImageDraw, ImageFont

    root = project_root()
    frozen_path, frozen, run_dirs = load_layout(root, spec_path)
    trials = json.loads((frozen_path.parent/"core_trials.json").read_text())
    minimum_init = min(row["init_index"] for row in trials)
    output.mkdir(parents=True, exist_ok=True)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 23)
    comparisons = []
    for task in frozen["physical_states"]["files"]:
        pair = []
        for method in (frozen["B_SELECTED"], frozen["C_SELECTED"]):
            trial = next(row for row in trials if row["task"] == task and row["init_index"] == minimum_init
                         and row["latency"] == 20 and row["config"]["method"] == method)
            directory = run_dirs["core"]/"trials"/trial_key(trial)
            outcome = json.loads((directory/"result.json").read_text())
            pair.append((directory/"episode.mp4", outcome))
        target = output/(task+f"_init{minimum_init}_L20.mp4")
        # Numeric/ASCII identifiers are controlled repo values; use structured argv,
        # no shell parsing, and retain literal percent signs outside interpolation.
        labels = [f"{item[1]['config']['method']} | success={int(item[1]['success'])}" for item in pair]
        graph = (f"[0:v]vflip,scale=480:480,drawtext=fontfile={font_path}:text='{labels[0]}':x=8:y=8:fontsize=18:"
                 "fontcolor=white:box=1:boxcolor=black@0.7,tpad=stop_mode=clone:stop_duration=15[a];"
                 f"[1:v]vflip,scale=480:480,drawtext=fontfile={font_path}:text='{labels[1]}':x=8:y=8:fontsize=18:"
                 "fontcolor=white:box=1:boxcolor=black@0.7,tpad=stop_mode=clone:stop_duration=15[b];"
                 "[a][b]hstack=inputs=2:shortest=1,pad=960:540:0:60:black,"
                 f"drawtext=fontfile={font_path}:text='{task} | bank init 0 | simulated L20 | video clock 20 Hz':"
                 "x=8:y=12:fontsize=19:fontcolor=white[v]")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-filter_threads", "1", "-filter_complex_threads", "1", "-i", str(pair[0][0]), "-i", str(pair[1][0]),
                        "-filter_complex", graph, "-map", "[v]", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-threads", "2", "-t", str(max(item[1]["steps"] for item in pair)*.05), str(target)], check=True)
        comparisons.append({"task": task, "path": str(target.relative_to(root)),
                            "baseline_success": pair[0][1]["success"], "candidate_success": pair[1][1]["success"]})
    segments = []

    def title(name, lines, seconds):
        canvas = Image.new("RGB", (960, 540), (24, 30, 43))
        draw = ImageDraw.Draw(canvas)
        for index, line in enumerate(lines):
            draw.text((30, 45+index*58), line, font=font, fill="white")
        image_path = output/(name+".png")
        canvas.save(image_path)
        target = output/(name+".mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-filter_threads", "1", "-filter_complex_threads", "1", "-loop", "1", "-i", str(image_path), "-t", str(seconds),
                        "-r", "20", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-threads", "2", str(target)], check=True)
        segments.append(target)

    main = summary["main"]
    title("intro", ["PLAN-04: frozen SmolVLA / LIBERO Goal", "Three supported tasks; one RTX 3090",
                    f"{frozen['B_SELECTED']} vs {frozen['C_SELECTED']}; unchanged 10-step sampler",
                    "Paired examples: bank index 0, simulated delay 20 ticks",
                    "Videos show simulation time at 20 Hz; failures retained"], 10)
    for index, comparison in enumerate(comparisons):
        target = output/(f"demo_pair_{index}.mp4")
        # A short episode is held on its actual final frame, never replaced by a success.
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-filter_threads", "1", "-filter_complex_threads", "1", "-i", str(root/comparison["path"]),
                        "-vf", "tpad=stop_mode=clone:stop_duration=10", "-t", "10", "-r", "20", "-an",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-threads", "2", str(target)], check=True)
        segments.append(target)
    async_trials = json.loads((frozen_path.parent/"async_trials.json").read_text())
    pair = []
    for method in (frozen["B_SELECTED"], frozen["C_SELECTED"]):
        trial = next(row for row in async_trials if row["task"] == "goal_bowl_plate" and row["init_index"] == minimum_init
                     and row["config"]["method"] == method)
        directory = run_dirs["async"]/"trials"/trial_key(trial)
        pair.append((directory/"episode.mp4", json.loads((directory/"result.json").read_text())))
    target = output/"demo_real_async_plate.mp4"
    labels = [f"{row['config']['method']} success={int(row['success'])} wall Hz={row['actual_control_hz']:.1f}" for _, row in pair]
    graph = (f"[0:v]vflip,scale=480:480,tpad=stop_mode=clone:stop_duration=10,drawtext=fontfile={font_path}:"
             f"text='{labels[0]}':x=8:y=8:fontsize=17:fontcolor=white:box=1:boxcolor=black@0.7[a];"
             f"[1:v]vflip,scale=480:480,tpad=stop_mode=clone:stop_duration=10,drawtext=fontfile={font_path}:"
             f"text='{labels[1]}':x=8:y=8:fontsize=17:fontcolor=white:box=1:boxcolor=black@0.7[b];"
             "[a][b]hstack=inputs=2:shortest=1,pad=960:540:0:60:black,"
             f"drawtext=fontfile={font_path}:text='plate init{minimum_init} | real GPU latency | cached video simulation clock 20 Hz':"
             "x=8:y=12:fontsize=19:fontcolor=white[v]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-filter_threads", "1", "-filter_complex_threads", "1", "-i", str(pair[0][0]), "-i", str(pair[1][0]),
                    "-filter_complex", graph, "-map", "[v]", "-t", "10", "-an", "-c:v", "libx264", "-r", "20",
                    "-pix_fmt", "yuv420p", "-threads", "2", str(target)], check=True)
    segments.append(target)
    hz = ", ".join(f"{row['method']}={row['mean_actual_control_hz']:.1f}" for row in summary["async_metrics"])
    title("async", ["Real asynchronous deployment in the simulator", "Spawned GPU owner; one request in flight",
                     "CPU-ready results; causal completed-request p95 estimator",
                     "Measured control Hz: "+hz, "Target 20 Hz; measured deadline misses reported",
                     "No hardware robot or forward-speedup claim"], 5)
    failure = None
    task_order = list(frozen["physical_states"]["files"])
    for method in (frozen["C_SELECTED"], frozen["B_SELECTED"]):
        ordered = sorted((row for row in trials if row["latency"] == 20 and row["config"]["method"] == method),
                         key=lambda row: (task_order.index(row["task"]), row["init_index"]))
        for trial in ordered:
            directory = run_dirs["core"]/"trials"/trial_key(trial)
            outcome = json.loads((directory/"result.json").read_text())
            if not outcome["success"]:
                failure = {"trial": trial, "outcome": outcome, "source": str((directory/"episode.mp4").relative_to(root)),
                           "rule": frozen["video_rule"]["failure_example"], "outcome_selected_for_illustration_only": True}
                break
        if failure:
            break
    if failure:
        trial, outcome = failure["trial"], failure["outcome"]
        target = output/"demo_failure_example.mp4"
        pair = []
        for method in (frozen["B_SELECTED"], frozen["C_SELECTED"]):
            matching = next(row for row in trials if row["task"] == trial["task"] and row["latency"] == 20
                            and row["init_index"] == trial["init_index"] and row["config"]["method"] == method)
            directory = run_dirs["core"]/"trials"/trial_key(matching)
            pair.append((directory, json.loads((directory/"result.json").read_text())))
        failure["paired_sources"] = [str((directory/"episode.mp4").relative_to(root)) for directory, _ in pair]
        failure["paired_successes"] = [row["success"] for _, row in pair]
        labels = [f"{row['config']['method']} | {row['termination_reason']}" for _, row in pair]
        graph = (f"[0:v]vflip,scale=480:480,tpad=stop_mode=clone:stop_duration=10,drawtext=fontfile={font_path}:"
                 f"text='{labels[0]}':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.7[a];"
                 f"[1:v]vflip,scale=480:480,tpad=stop_mode=clone:stop_duration=10,drawtext=fontfile={font_path}:"
                 f"text='{labels[1]}':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.7[b];"
                 "[a][b]hstack=inputs=2:shortest=1,pad=960:540:0:60:black,"
                 f"drawtext=fontfile={font_path}:text='Failure illustration | {trial['task']} init {trial['init_index']} L20':"
                 "x=8:y=6:fontsize=18:fontcolor=white,"
                 f"drawtext=fontfile={font_path}:text='Lowest failed init under frozen rule | illustration only | video clock 20 Hz':"
                 "x=8:y=31:fontsize=16:fontcolor=white[v]")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-filter_threads", "1", "-filter_complex_threads", "1", "-i", str(pair[0][0]/"episode.mp4"),
                        "-i", str(pair[1][0]/"episode.mp4"), "-filter_complex", graph, "-map", "[v]",
                        "-t", "10", "-r", "20", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-threads", "2",
                        str(target)], check=True)
        failure["paired_video"] = str(target.relative_to(root))
        segments.append(target)
    else:
        title("no_failure", ["No failed core L20 episode under either main method",
                             "A failed formal example is unavailable; none is invented",
                             "All raw episodes and fixed index-0 comparisons are retained"], 10)
    title("conclusion", ["Result: "+summary["classification"].replace("_", " ")[:65],
                         f"High-latency paired gain: {100*main['gain']:+.1f} percentage points",
                         f"95% paired bootstrap: [{100*main['ci95'][0]:+.1f}, {100*main['ci95'][1]:+.1f}] pp",
                         f"Generation cost ratio: {main['generation_cost_ratio']:.2f}",
                         "All formal comparisons frozen before scoring",
                         "Reports include action prefixes, late results and failures"], 15)
    concat = output/"demo_concat.txt"
    concat.write_text("".join(f"file '{path.name}'\n" for path in segments))
    demo = output/"demo_annotated.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-filter_threads", "1", "-filter_complex_threads", "1", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-c", "copy", str(demo)], check=True)
    duration = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                              "-of", "default=noprint_wrappers=1:nokey=1", str(demo)], text=True))
    if not 60 <= duration <= 90:
        raise ValueError("demo duration outside 60–90 seconds")
    return {"comparisons": comparisons, "demo": str(demo.relative_to(root)), "demo_seconds": duration,
            "failure_illustration": failure,
            "sample_rule": f"fixed minimum bank init {minimum_init}, L20, each task; no outcome selection"}


def finish(*, spec_path: Path | None = None, output_dir: Path | None = None) -> dict:
    root = project_root()
    frozen_path, frozen, run_dirs = load_layout(root, spec_path)
    directory = root/(output_dir or Path(frozen.get("documentation_dir", "docs/latency_v2")))
    summary = summarize(directory, spec_path=spec_path)
    core_mechanism = analyze(run_dirs["core"], directory/"core_mechanism.json")
    analyze(run_dirs["jitter"], directory/"jitter_mechanism.json")
    analyze(run_dirs["async"], directory/"async_mechanism.json")
    analyze(root/"runs/latency_v2_dev_candidates", directory/"development_mechanism.json")
    figures = build_figures(summary, directory/"figures", spec_path=spec_path)
    materials = root/frozen.get("materials_dir", "runs/latency_v2_materials")
    videos = build_videos(summary, materials/"videos", spec_path=spec_path)
    if videos["failure_illustration"]:
        from .plots import timeline

        for source in videos["failure_illustration"]["paired_sources"]:
            trial_directory = (root/source).parent
            row = json.loads((trial_directory/"result.json").read_text())
            path = directory/"figures"/("failure_pair_"+row["config"]["method"]+"_timeline.png")
            timeline(trial_directory, path)
            figures.append(str(path.relative_to(root)))
    publish(directory/"videos.json", videos)
    publish(directory/"video_inventory.json", inventory(root,
            [path for path in materials.rglob("*") if path.is_file()]))
    publish(directory/"physical_bank_inventory.json", inventory(root,
            [path for path in (root/"runs/latency_v2_physical_bank").rglob("*") if path.is_file()]))
    legacy = source_inventory(root, legacy_only=True)
    audit = json.loads((root/"runs/latency_v2_audit/audit.json").read_text())
    if legacy["sha256"] != audit["legacy_inventory_before"]["sha256"]:
        raise ValueError("existing project artifacts changed; inspect before final delivery")
    all_rows = []
    for name in ("core", "jitter", "async"):
        run_dir = run_dirs[name]
        stage_rows = completed_results(run_dir)
        all_rows.extend(stage_rows)
        metrics = json.loads((directory/(name+"_mechanism.json")).read_text())["trial_metrics"]
        fields = ("task", "init_index", "seed", "latency", "profile", "config")
        indexed_metrics = {trial_key({key: item[key] for key in fields}): item for item in metrics}
        for row in stage_rows:
            trial = {key: row[key] for key in fields}
            target = run_dir/"trials"/trial_key(trial)
            attempt = json.loads((target/"attempt.json").read_text())
            metric = indexed_metrics[trial_key(trial)]
            publish(target/"trial.json", {"unique_key": trial_key(trial), "stage": name, **row,
                "attempt_number": attempt["attempt_number"], "checkpoint": frozen["checkpoint"],
                "model_sha256": frozen["model_sha256"], "formal_spec_sha256": frozen["spec_sha256"],
                "physical_bank_manifest_sha256": frozen["physical_manifest_sha256"],
                "failure_type": row["termination_reason"], "mean_action_observation_age_ticks": metric["mean_observation_age_ticks"],
                "max_executed_prefix_violation": metric["max_executed_prefix_violation"]})
        publish(directory/(name+"_raw_inventory.json"), inventory(root,
                [path for path in (run_dir/"trials").rglob("*") if path.is_file()]))
    with (directory/"formal_trials.csv").open("w") as file:
        fields = ["task", "init_index", "latency", "profile", "method", "success", "steps", "n_requests",
                  "generation_seconds", "inference_seconds", "wall_seconds", "fallback_ticks", "late_results", "termination_reason",
                  "requests_per_sim_second", "generation_seconds_per_sim_second", "actual_control_hz",
                  "realtime_20hz_passed", "timing_provenance_complete"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({**{key: row.get(key) for key in fields if key != "method"}, "method": row["config"]["method"]})
    main = summary["main"]
    classification = {"positive_within_fixed_tasks": "固定三任务范围内正结果",
                      "limited_positive_not_meeting_full_threshold": "有限结果：点估计为正，未达到完整提升标准",
                      "negative_no_improvement_over_selected_strong_baseline": "负结果：未超过开发期选择的强基线"}[summary["classification"]]
    async_lines = "\n".join(f"| {row['method']} | {row['success_rate']:.1%} | {row['mean_actual_control_hz']:.2f} | "
                     f"{row['mean_deadline_missed_fraction']:.1%} | {row['mean_fraction_cycles_under_55ms']:.1%} | "
                     f"{row['mean_late_results']:.2f} | {row['mean_fallback_fraction']:.1%} |" for row in summary["async_metrics"])
    core_lines = "\n".join(f"| {row['latency']} | {row['method']} | {row['success_rate_equal_task']:.1%} | "
                    f"{row['mean_requests']:.2f} | {row['mean_generation_s']:.2f} | "
                    f"{row['mean_requests_per_sim_second']:.3f} | {row['mean_generation_s_per_sim_second']:.3f} |"
                    for row in summary["core_aggregate"])
    mechanism_lines = "\n".join(f"| {row['latency']} | {row['method']} | {row['mean_observation_age_ticks']:.2f} | "
                        f"{row['switch_translation_mean_action_norm']:.3f} | {row['mean_fallback_fraction']:.1%} | "
                        f"{row['mean_committed_old_buffer_shortage_ticks']:.2f} | {row['mean_late_results']:.2f} |" for row in core_mechanism["aggregate"])
    resource_lines = "\n".join(f"| {row['prefix_length']} | {row['median_endpoint_s']:.3f} | "
                        f"{row['median_endpoint_ratio_to_native']:.3f} | "
                        f"{row['max_peak_allocated_bytes']/2**20:.1f} | "
                        f"{row['additional_peak_allocated_bytes']/2**20:+.1f} |"
                        for row in frozen["resource_probe"]["rows"])
    budget = Budget(root/"runs/latency_v2_budget.json")
    remaining = budget.remaining()
    scope = ("本轮是验收复验，沿用原开发阶段选出的强基线、候选身份和参数；原有 324 次基线扫描与 180 次候选筛选未重复。"
             if frozen.get("run_id") else "本轮完成冻结 E1 实现、324 次强基线开发扫描和 180 次候选筛选。")
    report = f"""# PLAN-04 最终实测报告

{scope}
完成真实跨进程异步开发验证，以及正式核心 {summary['counts']['core']}、抖动 {summary['counts']['jitter']}、
异步 {summary['counts']['async']} 个配对 trials。结论：**{classification}**。

- delivery_complete: {summary['delivery_complete']}
- realtime_20hz_passed: {summary['realtime_20hz_passed']}
- timing_provenance_complete: {summary['timing_provenance_complete']}

三项分别判断；时间溯源范围为本规格正式 trial，历史缺失字段仍为未知。

主比较在开发期固定为 {frozen['C_SELECTED']} 对 {frozen['B_SELECTED']}；L10/L20 等权任务成功率差
为 {100*main['gain']:+.2f} pp，10,000 次按任务内 init 配对重采样的 95% CI 为
[{100*main['ci95'][0]:+.2f}, {100*main['ci95'][1]:+.2f}] pp。同一 init 的两个时延和所有方法保留在一起。
生成耗时比为 {main['generation_cost_ratio']:.3f}，真实异步成功率差为 {100*summary['async_gain']:+.2f} pp。
区间包含零时，不能断言显著优于或劣于基线。
这是固定三项已有任务的状态分布评测；不主张未见任务泛化或训练状态无重叠。

## 隔离与冻结

原生 init30–49 已历史暴露。本轮通过同一 BDDL 的真实 placement sampler 保存每任务 20 个新物理状态，
同时保存 79 维状态与模型 body_pos/body_quat；比较 qpos 而非仅比较 seed/time，
使用不同 reset seed 验证完整场景摘要与双相机图像重复一致。状态生成和验证不调用策略、不看 rollout 分数。
新池分布是 simulator placement draws，与原生 held-out 状态列表不同。
本规格实际选择索引：{json.dumps(frozen.get('selected_init_indices', '原规格每任务 0–4'), ensure_ascii=False)}。
正式规格 SHA-256：`{frozen['spec_sha256']}`；全部核心/抖动/异步清单在第一次正式得分前同时冻结。
规模决定记录在 `development_only_scale_decision`：{json.dumps(frozen['development_only_scale_decision'], ensure_ascii=False)}。
所有 formal 结果只用于分析，没有参数回调或赢家替换。完成结果不会在 resume 时重跑。

## 正式核心结果

| L tick | 方法 | 等权任务成功率 | 请求/episode | 生成秒/episode | 请求/仿真秒 | 生成秒/仿真秒 |
| --- | --- | --- | --- | --- | --- | --- |
{core_lines}

各任务条件的分母、Wilson 95% 区间见 [summary.json](summary.json)；主推断使用配对 bootstrap。
完整开发扫描与每时延配置见运行目录的 baseline_selection.json / candidate_selection.json。
固定同一模型与 10 步 sampler；请求频率和生成耗时的差异作为真实成本取舍报告，不预设“同预算”。
成功早停会改变回合长度，因此每回合成本差不能直接解释为调用频率增加或模型前向变慢。

独立开发资源检查在同一开发图像/状态/语言下，对每种前缀长度测量 5 次实际 CUDA 调用：

| 前缀长度（0 为原 sampler） | 中位端到端秒 | 相对原 sampler 耗时比 | Torch 分配峰值 MiB | 额外分配峰值 MiB |
| --- | --- | --- | --- | --- |
{resource_lines}

显存为进程内 PyTorch 分配统计，不包含渲染器、驱动和桌面；不把全卡占用当作策略显存。
采样开销检查不是 rollout 成功证据，原始逐次计时见 `runs/latency_v2_resource_probe/probe.json`。

## 机制证据

每个积分步前后投影真实进入模型速度场；S3 空前缀与原采样器逐位等价，20 次投影误差为零。
同一开发观测、语言与 request noise 下，改变前 7 步平移前缀，归一化后缀 L∞ 差为
{frozen['conditioning_probe']['normalized_suffix_linf_difference']:.6g}；无投影的私有包装路径与原采样器逐位一致。
此检查只验证输入干预与采样接入，不作为任务收益证据；详情见 `runs/latency_v2_conditioning_probe/probe.json`。
正式前还审查了真实同 init/seed/L/K 的开发失败对，见 [DEVELOPMENT_PILOT.md](DEVELOPMENT_PILOT.md)。
正式执行前缀与承诺数组的最大偏差为 {max(row['prefix_violation_max'] for row in core_mechanism['aggregate']):.3g}。
旋转和夹爪沿用新动作；仅平移增量在同一绝对 tick 混合。迟到候选被丢弃，缓冲不足使用零机械臂增量与保持夹爪目标。

| L | 方法 | 平均观测年龄 tick | 切换平移命令范数 | 后备比例 | 承诺中旧缓冲不足 tick/episode | 平均迟到结果 |
| --- | --- | --- | --- | --- | --- | --- |
{mechanism_lines}

观测年龄小、切换平滑或前缀严格满足，均不等价于任务成功。锁定 c 个动作会降低闭环修正频率；
清洁前缀投影可能偏离模型训练时的噪声轨迹分布；B_SHIFT 只是删除等待期动作，没有保证旧执行序列等于新前缀。
前缀/SHIFT 块的有效末端为旧观测 tick+H；RAW 的末端为激活 tick+H。
固定周期前缀方法若 max(K,c)+c>H，会在正常接续时缺少旧动作缓冲；H50/K50/c22 每次后续承诺缺少 22 个旧动作。
该时间关系是执行协议直接推导，表中的不足计数来自实际动作/事件；不足是否导致具体任务失败仍需谨慎解释。
这些是与实现和统计相关的机制解释，未额外用正式集调参或消融来证明单独的因果效应。
观测年龄针对非中性动作的活跃模型计划；复制承诺沿用旧计划的年龄，混合平移还包含更老的分量。
失败阶段图使用对象高度与末端距离的离线几何启发式，不能当作可靠抓取/接触真值。
固定最低 init 的真实成败对照视频及时间线均保留，不只展示成功。

## 真实异步

| 方法 | 成功率 | 实际 Hz | 50ms 超限比例 | ≤55ms 周期比例 | 平均迟到数 | 后备比例 |
| --- | --- | --- | --- | --- | --- | --- |
{async_lines}

控制进程独占环境，spawn worker 独占模型，队列容量 1；worker 后处理到 CPU 后才 ready。
初始化推理单独计时，所有方法协议一致；推理阶段没有注入 sleep，控制时钟不会补步或跳步。
S0 EGL 为 llvmpipe，开发和正式统一改用 WSL D3D12 RTX3090 渲染，仍保持双 256×256 相机与原物理/成功谓词。
是否达到 20Hz 由实测周期判定；该实现仅为仿真异步，不是真机部署或硬实时证明。
按每次异步 episode 至少 95% 周期≤55ms 的规则，20Hz 目标验收：{'通过' if summary['async_target_20hz_gate_passed'] else '未通过，结果仅支持实测频率下的诊断结论'}。
观测模式：{frozen['timing'].get('observation_mode', 'per_tick')}。请求采图、state、快照复制、通信、物理、诊断与日志入队均计入控制周期。
缓存视频仍按仿真 tick 编码；实际采样时间和帧年龄见各 trial 的 frame_sampling.json，不代表相机持续 20Hz 采样。

## 原有成果、成本与材料

保留 PLAN-03 的 800 trials、训练/数据预算/语言成果与 S0 全部审计。
历史 E1 官方路径为 88/90（97.78%），其初始化暴露和自动 reset 的审计见 [AUDIT.md](AUDIT.md)；
不能把该数字与本轮新状态池直接合并。
总尝试预算已使用 {1800-remaining}/1800，剩余 {remaining}；包括 S0、开发、状态验证及正式实际尝试。
GPU 生成耗时是同步墙钟测量，不等价于整段 episode 都在占用 GPU。
原始数组/事件在 `runs/latency_v2_*/trials`，完整汇总见 [formal_trials.csv](formal_trials.csv)、[summary.json](summary.json)。
图表见 [figures](figures)，视频清单见 [videos.json](videos.json)，演示实测时长 {videos['demo_seconds']:.2f}s。
简历与面试材料见 [RESUME.md](RESUME.md)、[INTERVIEW.md](INTERVIEW.md)，复现与恢复见 [RUNBOOK.md](RUNBOOK.md)。
"""
    (directory/"REPORT.md").write_text(report)
    if frozen.get("run_id"):
        report = report.replace("(AUDIT.md)", "(../AUDIT.md)").replace("(DEVELOPMENT_PILOT.md)", "(../DEVELOPMENT_PILOT.md)")
        (directory/"REPORT.md").write_text(report)
        import shutil

        shutil.copy2(root/"docs/latency_v2/INTERVIEW.md", directory/"INTERVIEW.md")
        (directory/"RUNBOOK.md").write_text(f"# 本轮复验与恢复\n\n工作目录：{root}。\n\n"
            f"冻结规格：{frozen_path.relative_to(root)}，哈希 `{frozen['spec_sha256']}`。\n\n"
            "使用保存的 code_snapshot 核查源码；代码变动后不得继续此规格。完整 trial 在 resume 时跳过。\n\n"
            "```bash\n" + "\n".join(f"bash scripts/run_latency_v2.sh run --config {frozen_path.parent.relative_to(root)}/{name}_config.json --resume"
            for name in ("core", "jitter", "async")) + "\n```\n\n"
            f"中断先执行 `bash scripts/run_latency_v2.sh recover --run-id {frozen['run_id']}`；"
            "结束时间未知会保留 null。基础设施错误分类后才能归档并付费重试；方法失败不得重试获取第二次评分。\n")
    (directory/"RESUME.md").write_text(f"""# 可核验的简历项目表述

- 复用冻结 SmolVLA E1，延续三任务监督适配、20%/50%/100% 数据预算与语言验证主线；历史官方评测 88/90，完成全权重和初始化泄漏审计，保留 800 条历史时延试验。
- 实现逐步流采样前缀约束与动作接续，完成 324 次强基线及 180 次候选开发扫描；新物理状态池正式核心 {summary['counts']['core']} 次评测，高时延相对强基线 {100*main['gain']:+.2f} pp，95% CI [{100*main['ci95'][0]:+.2f}, {100*main['ci95'][1]:+.2f}] pp，按实测交付{classification}。
- 在 LIBERO 实现 GPU worker 与控制解耦的单在途异步闭环，正式 {summary['counts']['async']} 次试验；实测控制频率 {min(row['mean_actual_control_hz'] for row in summary['async_metrics']):.2f}–{max(row['mean_actual_control_hz'] for row in summary['async_metrics']):.2f} Hz，记录迟到丢弃、中性后备与时钟超限，提供真实配对视频和 {videos['demo_seconds']:.0f}s 演示。
""")
    publish(directory/"completion.json", {"status": "complete", "classification": summary["classification"],
        "counts": summary["counts"], "formal_spec_sha256": frozen["spec_sha256"], "remaining_budget": remaining,
        "legacy_preserved": True, "delivery_complete": True,
        "realtime_20hz_passed": summary["realtime_20hz_passed"],
        "timing_provenance_complete": summary["timing_provenance_complete"], "figures": figures, "videos": videos})
    if frozen.get("run_id"):
        completion = {"status": "complete", "run_id": frozen["run_id"], "formal_completed": sum(summary["counts"].values()),
            "formal_expected": 345, "delivery_complete": True,
            "realtime_20hz_passed": summary["realtime_20hz_passed"],
            "timing_provenance_complete": summary["timing_provenance_complete"],
            "remaining_budget": remaining, "report": str((directory/"REPORT.md").relative_to(root))}
        publish(root/"runs"/frozen["run_id"]/"progress.json", completion)
        (directory/"PROGRESS.md").write_text("# PLAN-04 验收复验完成\n\n"+
            "\n".join(f"- {key}: {completion[key]}" for key in
            ("delivery_complete", "realtime_20hz_passed", "timing_provenance_complete", "formal_completed", "remaining_budget"))+"\n")
    artifacts = [path for path in directory.rglob("*") if path.is_file() and path.name != "delivery_inventory.json"]
    artifacts += [path for path in frozen_path.parent.rglob("*") if path.is_file()]
    publish(directory/"delivery_inventory.json", inventory(root, artifacts))
    return summary
