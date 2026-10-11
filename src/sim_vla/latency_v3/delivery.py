"""Derived development evidence and fixed-example videos; never launches rollouts."""
import json
from pathlib import Path

import numpy as np

from ..latency_v2.integrity import file_sha256, inventory
from ..latency_v2.storage import publish
from ..tasks import TASK_BY_KEY
from .experiments import DOCS, ROOT, RUN, read, rows, score
from .schema import METHODS


def development_materials():
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    screen = rows("dev_screen")
    legacy = rows("dev_legacy")
    raw = rows("dev_strong_raw")
    stages = ("dev_screen", "dev_confirm", "dev_strong_raw", "dev_strong_shift", "dev_legacy", "dev_async_r1")
    with (DOCS/"development_trials.csv").open("w") as handle:
        fields = ("stage", "task", "init_index", "latency", "method", "period", "success", "steps",
                  "termination_reason", "n_requests", "neutral_prefix_positions", "generation_seconds_per_sim_second")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stage in stages:
            for r in rows(stage):
                writer.writerow({k: stage if k == "stage" else r["config"][k] if k in ("method", "period") else r[k]
                                 for k in fields})
    lines = ["# PLAN-05 相同 K 开发比较", "", "全部为原生开发 init0，历史已暴露；样本小，只作描述性机制比较。",
        "同 K 不等于同实际请求预算，锁定期与单在途仍改变请求频率。", "",
        "| L | K | 方法 | 成功/回合 | 请求/仿真秒 | 生成秒/仿真秒 |",
        "| --- | --- | --- | --- | --- | --- |"]
    for L in (10, 20):
        for K in (20, 25):
            for method in METHODS:
                subset = [r for r in screen if r["latency"] == L and r["config"]["period"] == K
                          and r["config"]["method"] == method]
                s = score(subset)
                lines.append(f"| {L} | {K} | {method} | {sum(r['success'] for r in subset)}/{len(subset)} | "
                    f"{s['request_frequency']:.3f} | {s['generation_s_per_sim_s']:.3f} |")
    (DOCS/"SAME_PERIOD.md").write_text("\n".join(lines)+"\n")
    lines = ["# PLAN-05 覆盖修正机制", "", "旧 K50 与新 K20/25 使用共同静止启动、相同模型和观测协议。",
        "改变 K 同时改变请求预算和状态轨迹，因此不能把任务成功差全归因于缓存覆盖。", "",
        "| L | K | 方法 | 成功/回合 | 承诺中性填充/回合 | 后备比例 |",
        "| --- | --- | --- | --- | --- | --- |"]
    data = []
    for L in (10, 20):
        subsets = [([r for r in legacy if r["latency"] == L], "P_HARD legacy", 50),
            ([r for r in raw if r["latency"] == L and r["init_index"] == 0], "B_RAW", 50)]
        subsets += [([r for r in screen if r["latency"] == L and r["config"]["period"] == K
                      and r["config"]["method"] == "P_HARD"], "P_HARD", K) for K in (20, 25)]
        for subset, method, K in subsets:
            missing = np.mean([r["neutral_prefix_positions"] for r in subset])
            fallback = np.mean([r["fallback_ticks"]/max(r["steps"], 1) for r in subset])
            lines.append(f"| {L} | {K} | {method} | {sum(r['success'] for r in subset)}/{len(subset)} | {missing:.1f} | {fallback:.1%} |")
            data.append({"L": L, "K": K, "method": method, "missing": float(missing), "fallback": float(fallback)})
    lines += ["", "模型采样接入、投影误差与 RTC 干预见 audit/mechanism_probe.json；它们验证机制实现，不证明任务收益。",
        "正式结果与成本见 REPORT.md；夹爪时序、训练分布偏移与反馈修正受限仍作为未独立验证的假设。"]
    (DOCS/"MECHANISM.md").write_text("\n".join(lines)+"\n")
    publish(DOCS/"development_mechanism.json", data)
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = [f"L{r['L']} {r['method']} K{r['K']}" for r in data]
    ax.bar(labels, [r["missing"] for r in data])
    ax.set(ylabel="Neutral-filled commitment ticks / episode", title="Predeclared native-init0 development ablation")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    (DOCS/"figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(DOCS/"figures/coverage_ablation.png", dpi=160)
    plt.close(fig)
    # Fixed first task, L20, init0, feasible K25: actual action and request timeline.
    expected = read(RUN/"dev_screen/expected_trials.json")
    sample = next(t for t in expected if t["task"] == next(iter(TASK_BY_KEY)) and t["latency"] == 20
                  and t["config"] == {"method": "P_HARD", "period": 25})
    directory = RUN/"dev_screen/trials"/sample["key"]
    actions = np.load(directory/"arrays.npz")["actions"]
    requests = read(directory/"requests.json")
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axes[0].plot(actions[:, :3])
    axes[0].set(ylabel="Translation commands")
    for request in requests:
        if request["prefix_length"]:
            axes[1].axvspan(request["t_req"], request["t_req"]+request["prefix_length"], alpha=.2)
        axes[1].axvline(request["t_req"], color="C0", lw=.8)
        if request["t_act"] is not None:
            axes[1].plot(request["t_act"], request["remaining_old"], "o", color="C1")
    axes[1].set(xlabel="Control tick", ylabel="Old actions at request", ylim=(0, 51))
    fig.tight_layout()
    fig.savefig(DOCS/"figures/development_timeline.png", dpi=160)
    plt.close(fig)


def fixed_video(summary):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    directory = RUN/"materials"
    directory.mkdir(parents=True, exist_ok=True)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    writer = imageio.get_writer(directory/"demo_annotated.mp4", fps=20, codec="libx264",
                                quality=7, macro_block_size=1)
    examples = []
    try:
        for stage, duration, methods in (("formal_core", 15, METHODS),
                ("formal_async", 10, ("B_RAW", "P_HARD", "RTC_GUIDED"))):
            expected = read(RUN/stage/"expected_trials.json")
            for task in TASK_BY_KEY:
                init = min(t["init_index"] for t in expected if t["task"] == task)
                panels = []
                for method in methods:
                    trial = next(t for t in expected if t["task"] == task and t["init_index"] == init
                                 and t["config"]["method"] == method and t["latency"] == 20)
                    source = RUN/stage/"trials"/trial["key"]
                    video_path = source/"episode.mp4"
                    frames = list(imageio.mimread(video_path)) if video_path.exists() else [
                        imageio.imread(source/"initial_no_control.png")]
                    events = [json.loads(s) for s in (source/"events.jsonl").read_text().splitlines()]
                    steps = [e for e in events if e["event"] == "step"]
                    row = read(source/"result.json")
                    sampling = read(source/"frame_sampling.json")
                    panels.append((method, trial["config"]["period"], frames, steps, row, sampling))
                    examples.append({"stage": stage, "task": task, "init_index": init, "method": method,
                        "source": str(source.relative_to(ROOT)), "success": row["success"],
                        "video_sha256": file_sha256(video_path) if video_path.exists() else None,
                        "initial_scene_only": not video_path.exists()})
                for frame_idx in range(duration*20):
                    canvas = Image.new("RGB", (960, 640), (15, 20, 28))
                    draw = ImageDraw.Draw(canvas)
                    for panel_idx, (method, K, frames, steps, row, sampling) in enumerate(panels):
                        x, y = (panel_idx % 2)*480, (panel_idx//2)*320
                        index = min(frame_idx, len(frames)-1)
                        canvas.paste(Image.fromarray(frames[index]).resize((280, 280)), (x+100, y+28))
                        ended = frame_idx >= len(frames)
                        event = steps[min(index, len(steps)-1)] if steps else {
                            "tick": "none", "source": "no control steps", "remaining_old": 50}
                        label = f"{method} K{K} | init{init} | " + (
                            f"actual {row['actual_control_hz']:.2f}Hz" if stage == "formal_async" else "L20 simulated")
                        draw.text((x+8, y+5), label, font=font, fill="white")
                        lower = f"tick {event['tick']} | {event['source']} | remaining {event.get('remaining_old', 0)}"
                        draw.text((x+8, y+300), lower, font=font, fill="white")
                        age = sampling[index]["frame_age_ticks"] if sampling else 0
                        draw.text((x+100, y+278), f"frame age {age} ticks",
                                  font=font, fill=(255, 220, 70))
                        if ended or not steps:
                            draw.text((x+105, y+140), "END: "+row["termination_reason"], font=font,
                                      fill=(90, 255, 150) if row["success"] else (255, 110, 100))
                    if len(panels) == 3:
                        draw.text((490, 355), task, font=font, fill="white")
                        draw.text((490, 385), "Actual GPU/control overlap", font=font, fill="white")
                        draw.text((490, 415), "Frames encoded at simulation 20fps", font=font, fill="white")
                        draw.text((490, 445), "Committed actions / buffer shown above", font=font, fill="white")
                    writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
    publish(DOCS/"videos.json", {"recipe": "fixed smallest formal init, L20, all tasks/methods; hold last frame labelled END",
        "examples": examples, "demo": str((directory/"demo_annotated.mp4").relative_to(ROOT)),
        "expected_duration_s": 75, "encoding_fps": 20, "real_control_rates": summary["async"]})


def deliver(summary):
    development_materials()
    fixed_video(summary)
    (DOCS/"METHOD.md").write_text("# PLAN-05 方法\n\n"
        "共同静止原生块启动。RAW 从返回块首动作执行，SHIFT 按观测年龄裁前缀；HARD 与 RTC 在等待期锁定复制的旧动作，"
        "到承诺边界从索引 c 接续，迟到丢弃。HARD 每 flow 步前后投影，RTC 调用 LeRobot 0.5.1 原生 guidance。\n\n"
        "H50、c=L+2。固定请求满足 max(K,c)+c≤H，且每次实际 old remaining≥c；异步仅用已完成 endpoint 的窗口10/p95估计，"
        "必要时 K=min(K_selected,H-c)，2c>H 作为不可持续的方法失败。没有未来延迟输入、H扩展、末动作重复或中性前缀伪装。\n\n"
        "RTC guidance weight10、EXP、软重叠3，使用归一化旧动作和32维 padding。安装源码在计算速度场后才设置 x_t.requires_grad，"
        "本轮冻结权重并保留这一公式与顺序；不把它声称为含完整速度场Jacobian的论文复现。"
        "空前缀走未修改原 sampler；局部梯度使用 no_grad/enable_grad，未在 inference_mode 中运行。\n\n"
        "原生接口资料：[LeRobot RTC](https://huggingface.co/docs/lerobot/rtc)、[RTC原论文](https://arxiv.org/abs/2506.07339)。"
        "安装源码和项目执行器快照均保存在 runs/latency_v3_plan05/frozen/code_snapshot。\n")
    (DOCS/"RUNBOOK.md").write_text("# PLAN-05 复现与恢复\n\n工作目录：/home/lzx/VLA/embodied-learning/sim_vla。\n\n"
        "```bash\nbash scripts/run_latency_v3.sh audit\nbash scripts/run_latency_v3.sh develop\n"
        "bash scripts/run_latency_v3.sh freeze\nbash scripts/run_latency_v3.sh dry-run\n"
        "bash scripts/run_latency_v3.sh formal-core\nbash scripts/run_latency_v3.sh formal-async\n"
        "bash scripts/run_latency_v3.sh finish\n```\n\n完整恢复入口：`bash scripts/run_latency_v3.sh pipeline`；"
        "已完成结果跳过，执行规格和源文件哈希必须一致。中断后先 `bash scripts/run_latency_v3.sh recover`，"
        "只按基础设施错误分类付费重试；不因方法失败给第二次评分。旧 v2 路径保持可用。\n\n"
        "共享账本 cap=1800，本run cap=360。源代码变化不能混入已冻结矩阵；报告/视频离线重建不调用模型或环境。\n")
    m = summary["main"]
    rates = [r["mean_control_hz"] for r in summary["async"]]
    (DOCS/"RESUME.md").write_text("# 可核验的简历表述\n\n"
        "项目名称：SmolVLA 监督适配、时延评测与异步闭环优化。\n\n"
        "- 在三项 LIBERO Goal 任务上完成 SmolVLA 监督适配；历史官方路径 E1 88/90，20%/50% 数据预算为82/90、84/90，"
        "未见措辞评测84/90。上述结果按各自历史协议报告，与本轮新物理状态评测分开说明。\n"
        "- 实现带缓存覆盖检查的动作接续，并接入 LeRobot 原生 RTC guidance；完成同K开发对照与旧调度消融，"
        f"正式核心 {summary['counts']['core']} 次。主候选相对强RAW {m['gain']*100:+.2f}pp，"
        f"95% CI [{m['ci95'][0]*100:.2f},{m['ci95'][1]*100:.2f}]pp，按实测报告收益或负结果。\n"
        f"- 在单卡3090实现控制/推理分离，正式真实异步 {summary['counts']['async']} 次，方法平均频率"
        f"{min(rates):.2f}–{max(rates):.2f}Hz；逐方法成败与持续运行边界见REPORT，保留迟到、覆盖不可行、后备与成本证据。\n\n"
        "RTC 属于已有方法适配，三任务均为已有支持任务；实时阈值通过不等于持续精确20Hz。\n\n"
        "## 简历正文建议\n\n"
        "**SmolVLA 操作策略适配与异步闭环评测｜PyTorch、LeRobot、LIBERO、MuJoCo**\n\n"
        "- 基于双相机视觉、语言与机器人状态完成三任务监督适配；在既有官方协议下成功率88/90，完成20%/50%训练数据预算和措辞泛化对照。\n"
        "- 构建控制进程与GPU推理进程分离的评测链路，记录观测、请求、完成、激活与动作时间线，支持冻结配置、配对状态恢复和断点续跑。\n"
        "- 实现动作缓存覆盖检查与前缀接续，接入LeRobot原生RTC guidance，与强RAW/SHIFT基线完成新物理状态比较，定位长时延下的接续与吞吐边界。\n\n"
        "具体正负结果、统计区间与异步频率以本报告为准；未超过强基线时不写算法成功率提升，面试时主动解释该边界。\n")
    (DOCS/"INTERVIEW.md").write_text("# PLAN-05 面试问答\n\n"
        "1. 为什么原K50不可行？新块以观测tick为时间原点，下一次承诺跨越其H50末端；K50/c22缺22步真实旧动作。\n"
        "2. 修正是否证明性能提升？覆盖缺口减少是直接日志证据，任务成功要看冻结比较；K改变也改变调用成本。\n"
        "3. HARD和RTC差异？HARD在每步固定干净前缀；RTC在flow速度场上加原生guidance，生成前缀可以有偏差。\n"
        "4. 为什么RTC不能放在inference_mode？本地guidance需要局部autograd；no_grad允许内部enable_grad。\n"
        "5. 复现边界是什么？保存安装源码的公式、padding和梯度顺序，加项目执行器；不声称论文全系统复现。\n"
        "6. 怎么防未来泄漏？请求只使用新鲜观测、真实旧动作和已完成endpoint；当前未完成推理不进入延迟估计。\n"
        "7. 为什么模拟高分不等于部署效果？模拟释放tick可以小于真实GPU端到端耗时，真实单卡吞吐与迟到必须实测。\n"
        "8. 统计单元是什么？task内物理init，保留两个L和全部方法一起重采样；三任务不支持广泛泛化。\n"
        "9. 为什么换seed不算新状态？bank注入可覆盖reset随机性，固定设施还有model body位姿，需完整场景恢复证明。\n"
        "10. 负结果怎么交付？保留强基线、主身份、分母、成本和固定样例，区分验证的结构问题与未证实的机制假设。\n")
    readme = ROOT/"README.md"
    text = readme.read_text()
    if "<!-- PLAN05 -->" not in text:
        first, rest = text.split("\n", 1)
        readme.write_text(first+"\n\n<!-- PLAN05 -->\n最新动作接续研究：[PLAN-05报告](docs/latency_v3/REPORT.md)、"
            "[复现入口](docs/latency_v3/RUNBOOK.md)、[验收](docs/latency_v3/VALIDATION.md)。"
            "历史复验见 [PLAN-04](docs/latency_v2/acceptance_r1/REPORT.md)。\n<!-- /PLAN05 -->\n"+rest)
    publish(DOCS/"delivery_inventory.json", inventory(DOCS,
        (p for p in DOCS.rglob("*") if p.name != "delivery_inventory.json")))
