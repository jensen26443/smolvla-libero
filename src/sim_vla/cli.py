"""Command-line entry points for protocol and experiment operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assets import prepare_libero_assets, prepare_model_snapshot
from .data import build_data_manifest_main, prepare_data_main
from .language_alignment import language_alignment_main
from .lerobot_commands import build_eval_command, build_train_command
from .paired_language_eval import paired_language_eval_main
from .protocol import ExperimentProtocol, collect_environment, load_json
from .smoke import smoke_libero_main
from .splits import load_episode_index, make_nested_manifests, write_manifests
from .statistics import load_trial_records, summarize_lerobot_eval, summarize_trials
from .tasks import all_task_metadata


def _output(value: object, path: str | None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def check_environment_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="collect read-only host facts")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    _output(collect_environment(), args.output)
    return 0


def make_manifests_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="make nested, per-task LIBERO episode manifests")
    parser.add_argument("--episode-index", required=True, help="JSON/JSONL/CSV/parquet episode metadata")
    parser.add_argument("--output-dir", default="manifests")
    parser.add_argument("--dataset-repo", default="HuggingFaceVLA/libero")
    args = parser.parse_args(argv)
    protocol = ExperimentProtocol()
    episodes = load_episode_index(args.episode_index)
    manifests = make_nested_manifests(episodes, protocol=protocol, source=args.dataset_repo)
    paths = write_manifests(manifests, args.output_dir)
    print(json.dumps({"written": [str(path) for path in paths]}, ensure_ascii=False, indent=2))
    return 0


def summarize_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="summarize evaluation JSONL")
    parser.add_argument("records")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    _output(summarize_trials(load_trial_records(args.records)), args.output)
    return 0


def summarize_lerobot_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="convert and summarize official LeRobot eval_info.json")
    parser.add_argument("records", help="official eval_info.json")
    parser.add_argument("--condition", default="unspecified")
    parser.add_argument("--final-start-index", type=int)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    task_map = {str(item["suite_task_id"]): str(item["key"]) for item in all_task_metadata()}
    result = summarize_lerobot_eval(
        args.records,
        condition=args.condition,
        task_key_by_id=task_map,
        final_start_index=args.final_start_index,
    )
    _output(result, args.output)
    return 0


def build_command_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="print a verified LeRobot command")
    sub = parser.add_subparsers(dest="kind", required=True)
    train = sub.add_parser("train")
    train.add_argument("--output-dir", required=True)
    train.add_argument("--manifest")
    train.add_argument("--steps", type=int, default=10_000)
    train.add_argument("--eval-freq", type=int, default=2_000)
    train.add_argument("--save-freq", type=int, default=2_000)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--dataset-root")
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--episodes", type=int, default=20)
    evaluate.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.kind == "train":
        episodes = None
        if args.manifest:
            manifest = load_json(args.manifest)
            episodes = [item for values in manifest["train_episodes"].values() for item in values]
        spec = build_train_command(
            output_dir=args.output_dir,
            episodes=episodes,
            steps=args.steps,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            seed=args.seed,
            dataset_root=args.dataset_root,
        )
    else:
        spec = build_eval_command(
            checkpoint=args.checkpoint, output_dir=args.output_dir, episodes=args.episodes, seed=args.seed
        )
    print(spec.shell())
    return 0


def prepare_assets_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="materialize pinned Hub model/LIBERO assets")
    parser.add_argument("--kind", choices=("model", "libero", "all"), default="all")
    parser.add_argument("--model-root", help="optional project root for the model snapshot")
    parser.add_argument("--libero-assets", help="optional destination for LIBERO assets")
    args = parser.parse_args(argv)
    result: dict[str, str] = {}
    if args.kind in {"model", "all"}:
        result["model"] = str(prepare_model_snapshot(args.model_root))
    if args.kind in {"libero", "all"}:
        result["libero_assets"] = str(prepare_libero_assets(args.libero_assets))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="sim-vla protocol utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-environment").add_argument("--output")
    sub.add_parser("show-protocol")
    manifests = sub.add_parser("make-manifests")
    manifests.add_argument("--episode-index", required=True)
    manifests.add_argument("--output-dir", default="manifests")
    manifests.add_argument("--dataset-repo", default="HuggingFaceVLA/libero")
    summary = sub.add_parser("summarize")
    summary.add_argument("records")
    summary.add_argument("--output")
    lerobot_summary = sub.add_parser("summarize-lerobot")
    lerobot_summary.add_argument("records")
    lerobot_summary.add_argument("--condition", default="unspecified")
    lerobot_summary.add_argument("--final-start-index", type=int)
    lerobot_summary.add_argument("--output")
    assets = sub.add_parser("prepare-assets")
    assets.add_argument("--kind", choices=("model", "libero", "all"), default="all")
    data = sub.add_parser("prepare-data")
    data.add_argument("--episode-source", default="manifests/libero_episode_index.json")
    data.add_argument("--split", choices=("all", "train", "validation"), default="all")
    data.add_argument("--file-manifest", default="manifests/libero_data_files.json")
    data.add_argument("--root", default=".cache/lerobot/libero")
    data.add_argument("--output", default="runs/data_prepare.json")
    data.add_argument("--max-workers", type=int, default=4)
    data.add_argument("--no-download", action="store_true")
    data.add_argument("--no-verify", action="store_true")
    data_manifest = sub.add_parser("build-data-manifest")
    data_manifest.add_argument("--episode-source", default="manifests/libero_episode_index.json")
    data_manifest.add_argument("--output", default="manifests/libero_data_files.json")
    data_manifest.add_argument("--max-workers", type=int, default=8)
    smoke = sub.add_parser("smoke-libero")
    smoke.add_argument("--output", default="runs/libero_smoke.json")
    smoke.add_argument("--video-dir", default="runs/libero_smoke")
    smoke.add_argument("--resets", type=int, default=5)
    smoke.add_argument("--steps", type=int, default=1)
    smoke.add_argument("--assets")
    language_alignment = sub.add_parser("language-alignment")
    language_alignment.add_argument("--output", default="results/language_alignment_audit.json")
    language_alignment.add_argument("--seed", type=int, default=42)
    language_alignment.add_argument("--assets")
    language_alignment.add_argument("--no-runtime", action="store_true")
    paired_language = sub.add_parser("paired-language-eval")
    paired_language.add_argument("--checkpoint", required=True)
    paired_language.add_argument("--output-root", default="results")
    paired_language.add_argument("--phase", choices=("development", "formal", "all"), default="formal")
    paired_language.add_argument("--episodes", type=int)
    paired_language.add_argument("--seed", type=int, default=42)
    paired_language.add_argument("--device", default="cuda")
    paired_language.add_argument("--assets")
    paired_language.add_argument("--overwrite", action="store_true")
    paired_language.add_argument("--conditions", choices=("all", "e3", "e4"), default="all")
    paired_language.add_argument("--start-seed", type=int)
    paired_language.add_argument("--start-initialization-index", type=int)
    model_smoke = sub.add_parser("model-smoke")
    model_smoke.add_argument("--model-path")
    model_smoke.add_argument("--output", default="runs/model_smoke.json")
    model_smoke.add_argument("--task-id", type=int, default=8)
    model_smoke.add_argument("--seed", type=int, default=42)
    model_smoke.add_argument("--device", default="cuda")
    model_smoke.add_argument("--no-save-load", action="store_true")
    builder = sub.add_parser("build-command")
    builder.add_argument("kind", choices=("train", "eval"))
    builder.add_argument("--output-dir", required=True)
    builder.add_argument("--manifest")
    builder.add_argument("--checkpoint")
    builder.add_argument("--steps", type=int, default=10_000)
    builder.add_argument("--eval-freq", type=int, default=2_000)
    builder.add_argument("--save-freq", type=int, default=2_000)
    builder.add_argument("--episodes", type=int, default=20)
    builder.add_argument("--seed", type=int, default=42)
    builder.add_argument("--dataset-root")
    args, _rest = parser.parse_known_args(argv)
    if args.command == "check-environment":
        return check_environment_main([*(["--output", args.output] if args.output else [])])
    if args.command == "show-protocol":
        protocol = ExperimentProtocol()
        _output({"protocol": protocol.to_dict(), "tasks": all_task_metadata()}, None)
        return 0
    if args.command == "make-manifests":
        return make_manifests_main(
            ["--episode-index", args.episode_index, "--output-dir", args.output_dir, "--dataset-repo", args.dataset_repo]
        )
    if args.command == "summarize":
        return summarize_main([args.records, *(["--output", args.output] if args.output else [])])
    if args.command == "summarize-lerobot":
        forwarded = [args.records, "--condition", args.condition]
        if args.final_start_index is not None:
            forwarded.extend(["--final-start-index", str(args.final_start_index)])
        if args.output:
            forwarded.extend(["--output", args.output])
        return summarize_lerobot_main(forwarded)
    if args.command == "prepare-assets":
        return prepare_assets_main(["--kind", args.kind])
    if args.command == "build-data-manifest":
        return build_data_manifest_main(
            ["--episode-source", args.episode_source, "--output", args.output, "--max-workers", str(args.max_workers)]
        )
    if args.command == "prepare-data":
        forwarded = [
            "--episode-source",
            args.episode_source,
            "--split",
            args.split,
            "--file-manifest",
            args.file_manifest,
            "--root",
            args.root,
            "--output",
            args.output,
            "--max-workers",
            str(args.max_workers),
        ]
        if args.no_download:
            forwarded.append("--no-download")
        if args.no_verify:
            forwarded.append("--no-verify")
        return prepare_data_main(forwarded)
    if args.command == "smoke-libero":
        forwarded = ["--output", args.output, "--video-dir", args.video_dir, "--resets", str(args.resets), "--steps", str(args.steps)]
        if args.assets:
            forwarded.extend(["--assets", args.assets])
        return smoke_libero_main(forwarded)
    if args.command == "language-alignment":
        forwarded = ["--output", args.output, "--seed", str(args.seed)]
        if args.assets:
            forwarded.extend(["--assets", args.assets])
        if args.no_runtime:
            forwarded.append("--no-runtime")
        return language_alignment_main(forwarded)
    if args.command == "paired-language-eval":
        forwarded = [
            "--checkpoint",
            args.checkpoint,
            "--output-root",
            args.output_root,
            "--phase",
            args.phase,
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--conditions",
            args.conditions,
            "--start-seed",
            str(args.start_seed) if args.start_seed is not None else "",
            "--start-initialization-index",
            str(args.start_initialization_index) if args.start_initialization_index is not None else "",
        ]
        forwarded = [item for item in forwarded if item != ""]
        if args.episodes is not None:
            forwarded.extend(["--episodes", str(args.episodes)])
        if args.assets:
            forwarded.extend(["--assets", args.assets])
        if args.overwrite:
            forwarded.append("--overwrite")
        return paired_language_eval_main(forwarded)
    if args.command == "model-smoke":
        from .model_smoke import model_smoke_main

        forwarded = ["--output", args.output, "--task-id", str(args.task_id), "--seed", str(args.seed), "--device", args.device]
        if args.model_path:
            forwarded.extend(["--model-path", args.model_path])
        if args.no_save_load:
            forwarded.append("--no-save-load")
        return model_smoke_main(forwarded)
    if args.command == "build-command":
        forwarded = [
            args.kind,
            "--output-dir",
            args.output_dir,
            "--seed",
            str(args.seed),
        ]
        if args.kind == "train":
            if args.manifest:
                forwarded.extend(["--manifest", args.manifest])
            forwarded.extend(
                [
                    "--steps",
                    str(args.steps),
                    "--eval-freq",
                    str(args.eval_freq),
                    "--save-freq",
                    str(args.save_freq),
                ]
            )
            if args.dataset_root:
                forwarded.extend(["--dataset-root", args.dataset_root])
        else:
            if not args.checkpoint:
                parser.error("build-command eval requires --checkpoint")
            forwarded.extend(["--checkpoint", args.checkpoint, "--episodes", str(args.episodes)])
        return build_command_main(forwarded)
    parser.error(f"unsupported command: {args.command}; use the dedicated entry point for command builders")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
