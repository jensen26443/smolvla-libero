"""Durable execution queue from verified development to final evidence, no score callbacks."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback

from ..assets import project_root
from .experiments import select_baselines, select_candidates
from .freeze import async_development_trials, freeze_formal, verify_frozen
from .locking import run_lock
from .storage import Budget, publish


def write_progress(stage: str, status: str, detail: dict | None = None) -> None:
    root = project_root()
    stages = [("dev_baselines", 324), ("dev_candidates", 180), ("dev_async", 9),
              ("formal_core", None), ("formal_jitter", None), ("formal_async", None)]
    counts = []
    for name, default in stages:
        directory = root/("runs/latency_v2_"+name)
        count = len(list((directory/"trials").glob("*/result.json")))
        expected = default
        if (directory/"expected_trials.json").exists():
            expected = len(json.loads((directory/"expected_trials.json").read_text()))
        counts.append((name, count, expected))
    budget = Budget(root/"runs/latency_v2_budget.json")
    state = {"stage": stage, "status": status, "updated_unix": time.time(), "pid": os.getpid(),
             "counts": counts, "remaining_budget": budget.remaining(), "detail": detail or {}}
    publish(root/"runs/latency_v2_pipeline/progress.json", state)
    lines = "\n".join(f"| {name} | {count} | {expected if expected is not None else '正式前冻结'} |" for name, count, expected in counts)
    text = f"""# PLAN-04 执行进度

当前阶段：{stage}；状态：{status}。更新时间 UTC epoch：{state['updated_unix']:.0f}。
本文件由实际运行队列更新，完成分母取磁盘 result.json；不把排队任务计为完成。

| 阶段 | 完成 trials | 预期 trials |
| --- | --- | --- |
{lines}

预算已使用 {1800-budget.remaining()}/1800，剩余 {budget.remaining()}。S0 14 次与后续 4 个开发探针
保守计入基数，之后所有 attempt 在 reset/rollout 前持久化计费；中断或失败不会扣回预算。

S0 接口、模型、环境与隔离审计证据保留于 runs/latency_v2_audit。
空前缀/真实模型逐位等价、20 次逐步投影零误差已验证；新增控制器不会终止后隐式 reset。
原生正式候选状态历史耗尽不影响开发。只有 S1–S4 完成后才采样新物理状态并冻结全部正式规格。
执行顺序：强基线→候选开发→真实异步开发→物理状态验证→同时冻结三份正式矩阵→核心→抖动→异步→分析材料。
worker 队列有界、单在途，模型只在 worker 内执行；实测频率和失败会如实交付。
所有训练/语言/PLAN-03 资产及历史结果保留；正式评分不能改变配置、矩阵或主比较身份。

细节：{json.dumps(detail or {}, ensure_ascii=False)}
若 status 为 error，查看 runs/latency_v2_pipeline/error.json 和阶段日志；不会自动重跑成功结果。
"""
    (root/"docs/latency_v2/PROGRESS.md").write_text(text)


def run_child(stage: str, arguments: list[str], *, run_dir: str | None = None) -> None:
    root = project_root()
    log = root/("runs/latency_v2_pipeline/"+stage+".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as file:
        process = subprocess.Popen([sys.executable, *arguments], cwd=root, stdout=file, stderr=subprocess.STDOUT)
        write_progress(stage, "running", {"child_pid": process.pid, "log": str(log.relative_to(root))})
        last_publish = time.monotonic()
        while process.poll() is None:
            if time.monotonic()-last_publish >= 30:
                detail = {"child_pid": process.pid, "log": str(log.relative_to(root))}
                if run_dir and (root/run_dir/"progress.json").exists():
                    detail["latest"] = json.loads((root/run_dir/"progress.json").read_text())
                write_progress(stage, "running", detail)
                last_publish = time.monotonic()
            time.sleep(1)
    if process.returncode:
        raise RuntimeError(f"{stage} failed with exit {process.returncode}; retained log {log}")


def pipeline(*, baseline_pid: int | None = None, candidate_pid: int | None = None, resume: bool = False) -> dict:
    root = project_root()
    directory = root/"runs/latency_v2_pipeline"
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory/"owner.lock"
    import fcntl

    owner = lock.open("a")
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        baseline_dir = root/"runs/latency_v2_dev_baselines"
        if baseline_pid is not None and not (baseline_dir/"complete.json").exists():
            last_publish = 0.
            with run_lock(baseline_dir):
                while not (baseline_dir/"complete.json").exists():
                    try:
                        os.kill(baseline_pid, 0)
                    except ProcessLookupError as error:
                        raise RuntimeError("attached baseline process exited without completing its full scan") from error
                    if time.monotonic()-last_publish > 30:
                        write_progress("dev_baselines", "running", {"attached_pid": baseline_pid})
                        last_publish = time.monotonic()
                    time.sleep(1)
        elif not (baseline_dir/"complete.json").exists():
            args = ["-m", "sim_vla.latency_v2.cli", "run", "--config", "configs/latency_v2/dev_baselines.yaml"]
            if resume:
                args.append("--resume")
            run_child("dev_baselines", args, run_dir="runs/latency_v2_dev_baselines")
        baselines = select_baselines(baseline_dir)
        write_progress("dev_candidates", "running", {"baseline_selection_complete": True})
        candidate_dir = root/"runs/latency_v2_dev_candidates"
        if candidate_pid is not None:
            while not (candidate_dir/"complete.json").exists():
                try:
                    os.kill(candidate_pid, 0)
                except ProcessLookupError as error:
                    raise RuntimeError("attached candidate process exited before completing development") from error
                from pathlib import Path

                command = Path(f"/proc/{candidate_pid}/cmdline").read_bytes()
                if b"sim_vla.latency_v2.cli" not in command or b"dev_candidates.yaml" not in command:
                    raise RuntimeError("attached candidate PID exited or belongs to a different process")
                write_progress("dev_candidates", "running", {"attached_pid": candidate_pid})
                time.sleep(30)
        if not (candidate_dir/"complete.json").exists():
            args = ["-m", "sim_vla.latency_v2.cli", "run", "--config", "configs/latency_v2/dev_candidates.yaml"]
            if resume and (candidate_dir/"spec.json").exists():
                args.append("--resume")
            run_child("dev_candidates", args, run_dir="runs/latency_v2_dev_candidates")
        candidates = select_candidates(candidate_dir)
        if not (root/"runs/latency_v2_conditioning_probe/complete.json").exists():
            run_child("conditioning_probe", ["scripts/probe_latency_v2_conditioning.py"])
        if not (root/"runs/latency_v2_resource_probe/complete.json").exists():
            run_child("resource_probe", ["scripts/probe_latency_v2_resources.py"])
        run_child("development_mechanism", ["-m", "sim_vla.latency_v2.cli", "analyze-mechanism", "--run-dir",
            "runs/latency_v2_dev_candidates", "--output", "docs/latency_v2/development_mechanism.json"])
        # All production runtime tests are required before real async development or asset reads.
        run_child("verification", ["-m", "pytest", "-q"])
        async_config = {"stage": "dev_async", "run_dir": "runs/latency_v2_dev_async", "backend": "async",
            "checkpoint": "runs/train_100pct_formal/checkpoints/010000/pretrained_model", "max_ticks": 300,
            "trials_path": "runs/latency_v2_pipeline/dev_async_trials.json"}
        publish(directory/"dev_async_trials.json", async_development_trials(baselines, candidates))
        publish(directory/"dev_async_config.json", async_config)
        if not (root/async_config["run_dir"]/"complete.json").exists():
            args = ["-m", "sim_vla.latency_v2.cli", "run", "--config", str(directory/"dev_async_config.json")]
            if resume and (root/async_config["run_dir"]/"spec.json").exists():
                args.append("--resume")
            run_child("dev_async", args, run_dir=async_config["run_dir"])
        # Formal states are generated only now, after all tuning and GPU-worker smoke.
        manifest = root/"runs/latency_v2_physical_bank/manifest.json"
        if not manifest.exists():
            run_child("physical_bank", ["-m", "sim_vla.latency_v2.cli", "make-bank", "--output-dir",
                                       "runs/latency_v2_physical_bank"])
        frozen_dir = root/"runs/latency_v2_frozen"
        if not (frozen_dir/"experiment_spec.json").exists():
            frozen = freeze_formal(baselines, candidates, manifest, directory=frozen_dir)
        else:
            frozen = json.loads((frozen_dir/"experiment_spec.json").read_text())
        for name in ("core", "jitter", "async"):
            config_path = frozen_dir/(name+"_config.json")
            config = json.loads(config_path.read_text())
            verify_frozen(config)
            # Dry-run all three specifications before the first formal episode.
            run_child("dry_run_"+name, ["-m", "sim_vla.latency_v2.cli", "run", "--config", str(config_path), "--dry-run"])
        write_progress("formal_freeze", "passed", {"spec_sha256": frozen["spec_sha256"], "counts": frozen["counts"]})
        for name in ("core", "jitter", "async"):
            config = json.loads((frozen_dir/(name+"_config.json")).read_text())
            if (root/config["run_dir"]/"complete.json").exists():
                continue
            args = ["-m", "sim_vla.latency_v2.cli", "run", "--config", str(frozen_dir/(name+"_config.json"))]
            if resume and (root/config["run_dir"]/"spec.json").exists():
                args.append("--resume")
            run_child("formal_"+name, args, run_dir=config["run_dir"])
        run_child("final_materials", ["-m", "sim_vla.latency_v2.cli", "finish"])
        completion = json.loads((root/"docs/latency_v2/completion.json").read_text())
        publish(directory/"complete.json", completion)
        write_progress("S7", "complete", completion)
        from .integrity import inventory

        materials = root/"docs/latency_v2"
        artifacts = [path for path in materials.rglob("*") if path.is_file() and path.name != "delivery_inventory.json"]
        artifacts += [path for path in frozen_dir.rglob("*") if path.is_file()]
        publish(materials/"delivery_inventory.json", inventory(root, artifacts))
        return completion
    except Exception:
        error = {"error": traceback.format_exc(), "failed_unix": time.time(), "pid": os.getpid()}
        publish(directory/"error.json", error)
        write_progress("pipeline", "error", error)
        raise
    finally:
        owner.close()
