"""CLI for the latency evaluation project (PLAN-03).

Subcommands map to the plan's stages: ``selftest`` (§8 mock evidence),
``audit`` (§3 environment audit), ``run``/``resume`` (§9–§10 matrix with
resume), ``summarize`` (§12 statistics + figures), ``timeline`` (per-trial
event chart). Every command writes machine-readable artifacts under runs/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _selftest(output_dir: str) -> int:
    """Generate mock-timeline evidence artifacts; no GPU, no real model."""

    from sim_vla.latency_eval.config import load_config
    from sim_vla.latency_eval.experiment_matrix import ExperimentRunner, expand_trials
    from sim_vla.latency_eval.mocks import MockEnv, MockPolicy
    from sim_vla.latency_eval.timeline import plot_rows, rows_from_events_file, write_rows_csv

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config = load_config(Path(__file__).resolve().parents[3] / "configs" / "latency" / "selftest.yaml")

    runner = ExperimentRunner(
        config,
        output_root=root / "matrix",
        env_factory=lambda task_key: MockEnv(),
        policy_factory=lambda: MockPolicy(mode="capture_echo"),
    )
    summary = runner.run()
    n_charts = 0
    for trial in expand_trials(config):
        events_path = runner.trial_dir(trial) / "events.jsonl"
        if not events_path.is_file():
            continue
        rows = rows_from_events_file(events_path)
        stem = f"timeline_{trial.method_name}_{trial.latency_name}"
        write_rows_csv(rows, root / f"{stem}.csv")
        plot_rows(rows, root / f"{stem}.png", title=f"{trial.method_name} / {trial.latency_name} (mock)")
        n_charts += 1
        if n_charts >= 4:
            break
    (root / "selftest.json").write_text(
        json.dumps({"matrix": summary, "timeline_charts": n_charts}, indent=1), encoding="utf-8"
    )
    print(f"selftest complete: {summary['completed']}/{summary['trials_total']} trials, {n_charts} timelines")
    return 0


def _audit(args: argparse.Namespace) -> int:
    from sim_vla.latency_eval.audit import run_audit

    report = run_audit(
        checkpoint=args.checkpoint,
        tasks=tuple(args.tasks),
        output=args.output,
        device=args.device,
        timing_calls=args.timing_calls,
    )
    probe = report["model_probe"]
    print(
        "audit complete: chunk="
        f"{probe['raw_chunk_shape']} broadcast_ok="
        f"{probe['postprocessor_chunk_broadcast_verified']} "
        f"inference_p50={probe['measured_inference_seconds_p50']:.3f}s "
        f"(~{probe['equivalent_control_ticks_p50']:.0f} ticks)"
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    from sim_vla.latency_eval.config import load_config
    from sim_vla.latency_eval.experiment_matrix import run_experiment

    config = load_config(args.config)
    summary = run_experiment(
        config,
        output_root=args.output_root,
        resume=not args.no_resume,
        limit=args.limit,
    )
    print(
        f"experiment {config.experiment_id}: {summary['completed']}/{summary['trials_total']} "
        f"completed, {summary['failed_infrastructure']} infra failures, "
        f"{summary['executed_this_run']} executed this run in {summary['elapsed_seconds']:.0f}s"
    )
    return 0


def _summarize(args: argparse.Namespace) -> int:
    from sim_vla.latency_eval.metrics import write_results_csv, write_summary
    from sim_vla.latency_eval.plots import generate_all_figures

    summary_path = write_summary(args.experiment_dir)
    csv_path = write_results_csv(args.experiment_dir)
    figures = []
    if not args.no_figures:
        figures = generate_all_figures(args.experiment_dir)
    print(f"summary: {summary_path}")
    print(f"csv: {csv_path}")
    for figure in figures:
        print(f"figure: {figure}")
    return 0


def _timeline(args: argparse.Namespace) -> int:
    from sim_vla.latency_eval.timeline import plot_rows, rows_from_events_file, write_rows_csv

    trial_dir = Path(args.trial_dir)
    rows = rows_from_events_file(trial_dir / "events.jsonl")
    csv_path = write_rows_csv(rows, trial_dir / "timeline.csv")
    png_path = plot_rows(rows, trial_dir / "timeline.png", title=trial_dir.name)
    print(f"timeline: {csv_path} {png_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="latency evaluation CLI (PLAN-03)")
    sub = parser.add_subparsers(dest="command", required=True)

    selftest = sub.add_parser("selftest", help="mock-based timing evidence, no GPU")
    selftest.add_argument("--output-dir", default="runs/latency_selftest")

    audit = sub.add_parser("audit", help="stage-0 environment/model audit probes")
    audit.add_argument("--checkpoint", required=True)
    audit.add_argument("--tasks", nargs="+", default=["goal_bowl_stove", "goal_bowl_plate"])
    audit.add_argument("--output", default="runs/latency_audit/audit.json")
    audit.add_argument("--device", default="cuda")
    audit.add_argument("--timing-calls", type=int, default=30)

    run = sub.add_parser("run", help="run (or resume) an experiment matrix from a config")
    run.add_argument("--config", required=True)
    run.add_argument("--output-root", default="runs")
    run.add_argument("--limit", type=int, help="max trials to execute this invocation")
    run.add_argument("--no-resume", action="store_true")

    summarize = sub.add_parser("summarize", help="aggregate trial records into summary/figures")
    summarize.add_argument("--experiment-dir", required=True)
    summarize.add_argument("--no-figures", action="store_true")

    timeline = sub.add_parser("timeline", help="render the event timeline of one trial")
    timeline.add_argument("--trial-dir", required=True)

    args = parser.parse_args(argv)
    if args.command == "selftest":
        return _selftest(args.output_dir)
    if args.command == "audit":
        return _audit(args)
    if args.command == "run":
        return _run(args)
    if args.command == "summarize":
        return _summarize(args)
    if args.command == "timeline":
        return _timeline(args)
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
