"""PLAN-04 development and isolated formal experiment commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    import signal

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description="PLAN-04 audit and physical initialization isolation")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="real environment, checkpoint, GPU and historical-init audit")
    audit.add_argument("--config", required=True)
    timeline = commands.add_parser("timeline", help="render a real S0 legacy executor timeline")
    timeline.add_argument("--trial-dir", required=True)
    check = commands.add_parser("check-formal-inits", help="validate an actual physical initialization bank")
    check.add_argument("--manifest", required=True)
    check.add_argument("--audit", required=True)
    run = commands.add_parser("run", help="run a frozen real development/formal matrix")
    run.add_argument("--config", required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    select = commands.add_parser("select", help="select configurations from complete development results")
    select.add_argument("--run-dir", required=True)
    select.add_argument("--stage", choices=["baselines", "candidates"], required=True)
    queue = commands.add_parser("pipeline", help="execute development, fresh assets, formal comparisons and materials")
    queue.add_argument("--baseline-pid", type=int)
    queue.add_argument("--candidate-pid", type=int)
    queue.add_argument("--resume", action="store_true")
    bank = commands.add_parser("make-bank", help="generate policy-blind physical initialization arrays after development")
    bank.add_argument("--output-dir", required=True)
    finish_command = commands.add_parser("finish", help="produce analysis, figures, video and materials from complete frozen trials")
    finish_command.add_argument("--spec")
    finish_command.add_argument("--output-dir")
    summary = commands.add_parser("summarize", help="verify complete formal denominators and paired inference")
    summary.add_argument("--output-dir")
    summary.add_argument("--spec")
    profile = commands.add_parser("profile-env", help="paired, budgeted environment component timing")
    profile.add_argument("--output-dir", required=True)
    profile.add_argument("--observation-mode", choices=["per_tick", "on_request"], default="per_tick")
    commands.add_parser("freeze-acceptance", help="freeze the acceptance rerun only after development gates pass")
    recovery = commands.add_parser("recover", help="reconcile dead attempts without inventing end clocks")
    recovery.add_argument("--run-id", required=True)
    recovery.add_argument("--classify", choices=["infrastructure_error"])
    mechanism = commands.add_parser("analyze-mechanism", help="verify actual executed prefixes and single-flight provenance")
    mechanism.add_argument("--run-dir", required=True)
    mechanism.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze-acceptance":
        from .acceptance import freeze_acceptance

        result = freeze_acceptance()
        print(json.dumps({"spec_sha256": result["spec_sha256"], "counts": result["counts"]}))
        return 0
    if args.command == "recover":
        from ..assets import project_root
        from .storage import Budget

        print(json.dumps(Budget(project_root()/"runs/latency_v2_budget.json").recover(
            run_id=args.run_id, classification=args.classify), ensure_ascii=False))
        return 0
    if args.command == "profile-env":
        from .acceptance import profile_paths
        from .locking import run_lock

        with run_lock(Path(args.output_dir)):
            result = profile_paths(Path(args.output_dir), observation_mode=args.observation_mode)
        print(json.dumps({"profiles": len(result["results"]), "remaining_budget": result["remaining_budget"]}))
        return 0
    if args.command == "analyze-mechanism":
        from .mechanism import analyze

        result = analyze(Path(args.run_dir), Path(args.output))
        print(json.dumps({"trials": len(result["trial_metrics"]), "aggregate": result["aggregate"]}, ensure_ascii=False))
        return 0
    if args.command == "pipeline":
        from .pipeline import pipeline

        print(json.dumps(pipeline(baseline_pid=args.baseline_pid, candidate_pid=args.candidate_pid,
                                  resume=args.resume), ensure_ascii=False))
        return 0
    if args.command == "make-bank":
        from .bank import make_bank

        root = Path.cwd()
        if not (root/"runs/latency_v2_dev_async/complete.json").exists():
            parser.error("fresh formal states may only be created after completed asynchronous development")
        print(make_bank(Path(args.output_dir)))
        return 0
    if args.command == "finish":
        from .delivery import finish

        result = finish(spec_path=Path(args.spec) if args.spec else None,
                        output_dir=Path(args.output_dir) if args.output_dir else None)
        print(json.dumps({"classification": result["classification"], "counts": result["counts"]}, ensure_ascii=False))
        return 0
    if args.command == "summarize":
        from .statistics import summarize

        result = summarize(Path(args.output_dir) if args.output_dir else None,
                           spec_path=Path(args.spec) if args.spec else None)
        print(json.dumps({"classification": result["classification"], "main": result["main"]}, ensure_ascii=False))
        return 0
    if args.command == "run":
        import yaml

        from .experiments import development_baselines, development_candidates, run_matrix

        config = yaml.safe_load(Path(args.config).read_text())
        if config["stage"] == "dev_baselines":
            trials = development_baselines()
        elif config["stage"] == "dev_candidates":
            trials = development_candidates(json.loads(Path(config["baseline_selection"]).read_text()))
        else:
            trials = json.loads(Path(config["trials_path"]).read_text())
        if config["stage"].startswith("formal"):
            from .freeze import verify_frozen

            verify_frozen(config)
        from contextlib import nullcontext

        from ..assets import project_root
        from .locking import run_lock

        with nullcontext() if args.dry_run else run_lock(project_root()/config["run_dir"]):
            if config.get("backend") == "async":
                from .runner_async import run_async_matrix

                result = run_async_matrix(config, trials, resume=args.resume, dry_run=args.dry_run)
            else:
                result = run_matrix(config, trials, resume=args.resume, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "select":
        from .experiments import select_baselines, select_candidates

        result = (select_baselines if args.stage == "baselines" else select_candidates)(Path(args.run_dir))
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "audit":
        import yaml

        from ..assets import project_root
        from .audit import run_audit

        config = yaml.safe_load(Path(args.config).read_text())
        if config["warmup_calls"] < 10 or config["timing_calls"] < 50:
            parser.error("S0 requires at least 10 warmup and 50 effective GPU calls")
        report = run_audit(config, root=project_root())
        print(json.dumps({"status": report["status"], "run_dir": config["run_dir"],
                          "blockers": report.get("blockers", [])}, ensure_ascii=False))
        return 0 if report["status"] == "passed" else 2
    if args.command == "check-formal-inits":
        from .initializations import InitializationIsolationError, check_formal_manifest

        report = json.loads(Path(args.audit).read_text())
        try:
            result = check_formal_manifest(Path(args.manifest), report["initializations"])
        except (InitializationIsolationError, OSError) as error:
            print(json.dumps({"historical_isolation_passed": False, "error": str(error)}, ensure_ascii=False))
            return 2
        print(json.dumps(result, ensure_ascii=False))
        return 0
    directory = Path(args.trial_dir)
    if (directory/"requests.json").exists():
        from .plots import timeline as draw_timeline

        draw_timeline(directory, directory/"timeline.png")
        print(directory/"timeline.png")
        return 0
    from ..latency_eval.timeline import plot_rows, rows_from_events_file, write_rows_csv

    rows = rows_from_events_file(directory / "events.jsonl")
    write_rows_csv(rows, directory / "timeline.csv")
    plot_rows(rows, directory / "timeline.png", title=f"S0 real development trace: {directory.name}")
    print(directory / "timeline.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
