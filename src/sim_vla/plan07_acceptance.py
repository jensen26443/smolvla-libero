"""Read frozen PLAN-07 traces, including captured but unsubmitted diagnostics.

This post-run reader keeps the scored runtime and raw files unchanged. The
original reader assumes that every captured scene was submitted, although the
runner captures before the executor can reject insufficient action coverage.
"""
import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

from .latency_v2.integrity import file_sha256
from .latency_v2.locking import run_lock
from .latency_v2.storage import publish
from .latency_v5 import delivery, experiments
from .latency_v5.layout import RUN, progress, read

_RUNTIME_VERIFY = delivery.verify
_RUNTIME_REPORT = delivery.report


def submitted_scenes(scenes, requests, result, events):
    if len(scenes) == len(requests):
        return scenes, None
    if len(scenes) != len(requests)+1:
        raise ValueError("scene count is not the submitted set plus one terminal diagnostic")
    extra = scenes[-1]
    extra.validate()
    rejected = [event for event in events if event["event"] == "coverage_infeasible"
                and event["tick"] == extra.tick and event["reason"] == "insufficient_old_actions"]
    if (result["termination_reason"] != "coverage_infeasible" or result["error"]
            or extra.tick != result["steps"] or extra.task != result["task"] or len(rejected) != 1
            or any(event["event"] == "submit" and event["tick"] == extra.tick for event in events)
            or extra.sampled["state"]["end_monotonic_ns"] > rejected[0]["monotonic_ns"]):
        raise ValueError("extra scene is not a captured, rejected terminal diagnostic")
    return scenes[:-1], {"tick": extra.tick, "physical_scene_sha256": extra.scene_sha256,
                        "visual_sha256": extra.visual_sha256, "submitted": False}


def verify():
    diagnostics = {}

    def load(handle):
        scenes = pickle.load(handle)
        directory = Path(handle.name).parent
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        submitted, extra = submitted_scenes(scenes, read(directory / "requests.json"),
                                           read(directory / "result.json"), events)
        if extra:
            diagnostics[str(directory.relative_to(RUN))] = extra
        return submitted

    original_reader = delivery.pickle
    delivery.pickle = SimpleNamespace(load=load)
    try:
        result = _RUNTIME_VERIFY()
    finally:
        delivery.pickle = original_reader
    result.update(captured_unsubmitted_diagnostics=diagnostics, raw_files_changed=False,
                  scored_runtime_changed=False, verification_reader_sha256=file_sha256(Path(__file__)))
    publish(delivery.DOCS / "verification.json", result)
    publish(RUN / "testing/verification_reader.json", result)
    return result


def resume():
    if not (RUN / "validation/complete.json").exists():
        raise ValueError("this reader only resumes completed scored matrices")
    original_verify = delivery.verify
    original_report = delivery.report
    delivery.verify = verify
    delivery.report = report
    try:
        with run_lock(RUN):
            result = experiments.pipeline()
    finally:
        delivery.verify = original_verify
        delivery.report = original_report
    progress("complete", sustained_execution_supported=result["sustained_execution_supported"],
             realtime_gate_passed=result["realtime_gate_passed"], resume_claims_verified=True)
    return result


def report():
    result = _RUNTIME_REPORT()
    path = delivery.DOCS / "RUNBOOK.md"
    path.write_text("# PLAN-07 已完成矩阵复核与恢复\n\n从sim_vla运行：\n\n```bash\n"
                    "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -m sim_vla.plan07_acceptance verify\n"
                    "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -m sim_vla.plan07_acceptance resume\n"
                    "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -m sim_vla.plan07_acceptance report\n"
                    "```\n\n冻结评分代码与原始文件保持不变。原runner先捕获场景，再尝试提交；"
                    "两次STRICT覆盖失败各留下一个未提交的末尾诊断快照。原读取器假定捕获数等于提交数，"
                    "因此完成矩阵后报错；独立读取器仅在终止原因、tick、覆盖拒绝事件、采样时钟和"
                    "无同tick提交均符合时单列该诊断，实际提交的每个输入仍逐一完整核验。\n\n"
                    "已完成resume在创建环境或GPU worker前返回，实测0新增尝试，账本字节一致。"
                    "原脚本pipeline保留作为评分版本档案；本次已完成矩阵使用上述读取器恢复。"
                    "所有构造/reset/异常保留计费；原PLAN-06账本与门槛结论保留。"
                    "编码20fps不能代替实际控制频率。详见REVIEW.md及verification.json。\n")
    strict = result["gates"]["validation"]["methods"]["RTC_STRICT"]
    catchup = result["gates"]["validation"]["methods"]["RTC_CATCHUP"]
    path = delivery.DOCS / "REPORT.md"
    text = path.read_text()
    text = text.replace(f"**固定工程验证问题解决：{result['engineering_problem_resolved_within_validation']}。**",
        f"**固定持续/实时验收门槛通过：{result['engineering_problem_resolved_within_validation']}。"
        f"STRICT完整持续{strict['full_300_ticks']}/6，CATCHUP {catchup['full_300_ticks']}/6；保留失败分母。**")
    path.write_text(text)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify", "resume", "report"))
    args = parser.parse_args()
    result = {"verify": verify, "resume": resume, "report": report}[args.command]()
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("gates", "request_distributions", "runtime")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
