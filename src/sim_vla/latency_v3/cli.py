"""PLAN-05 resumable stages; policy failures keep their frozen denominator."""
import argparse
import json

from ..latency_v2.storage import publish, trial_key
from .execution_settings import verify_execution_settings
from .experiments import (
    RUN,
    RUN_ID,
    async_dev_trials,
    budget,
    confirmation_trials,
    ensure_frozen_unchanged,
    finalize_selection,
    freeze,
    legacy_trials,
    progress,
    read,
    run_matrix,
    screen_trials,
    select_short,
    strong_trials,
)


def develop():
    run_matrix("dev_screen", screen_trials())
    short = select_short()
    run_matrix("dev_confirm", confirmation_trials(short))
    run_matrix("dev_strong_raw", strong_trials("B_RAW"))
    run_matrix("dev_strong_shift", strong_trials("B_SHIFT"))
    run_matrix("dev_legacy", legacy_trials())
    selection = finalize_selection(short)
    run_matrix("dev_async_r1", async_dev_trials(selection), asynchronous=True)
    return selection


def pipeline():
    frozen_path = RUN/"frozen/experiment_spec.json"
    if frozen_path.exists():
        ensure_frozen_unchanged(read(frozen_path))
    from .audit import audit, verify_physical_starts
    from .materials import finish

    budget()
    audit()
    selection = develop()
    verify_physical_starts()
    spec = freeze(selection)
    run_matrix("formal_core", spec["core_trials"], physical=True)
    run_matrix("formal_async", spec["async_trials"], physical=True, asynchronous=True)
    summary = finish()
    progress("complete", classification=summary["classification"], counts=summary["counts"])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "develop", "freeze", "formal-core", "formal-async",
                                           "finish", "pipeline", "dry-run", "recover"))
    args = parser.parse_args()
    frozen_path = RUN/"frozen/experiment_spec.json"
    if args.command != "recover" and frozen_path.exists():
        verify_execution_settings(read(frozen_path), RUN)
    if args.command == "audit":
        from .audit import audit

        result = audit()
    elif args.command == "develop":
        result = develop()
    elif args.command == "freeze":
        from .audit import verify_physical_starts

        verify_physical_starts()
        result = freeze(read(RUN/"selection.json"))
    elif args.command in ("formal-core", "formal-async"):
        spec = read(RUN/"frozen/experiment_spec.json")
        asynchronous = args.command == "formal-async"
        result = run_matrix("formal_async" if asynchronous else "formal_core",
                            spec["async_trials" if asynchronous else "core_trials"],
                            physical=True, asynchronous=asynchronous)
    elif args.command == "finish":
        from .materials import finish

        result = finish()
    elif args.command == "recover":
        result = budget().recover(run_id=RUN_ID)
        publish(RUN/"recovery.json", result)
    elif args.command == "dry-run":
        result = run_matrix("dev_screen", screen_trials(), dry_run=True)
        frozen = RUN/"frozen/experiment_spec.json"
        if frozen.exists():
            spec = read(frozen)
            ensure_frozen_unchanged(spec)
            result["formal_core"] = len(spec["core_trials"])
            result["formal_async"] = len(spec["async_trials"])
            result["formal_core_unique"] = len({trial_key(t) for t in spec["core_trials"]})
            result["formal_async_unique"] = len({trial_key(t) for t in spec["async_trials"]})
            result["remaining_budget"] = budget().remaining()
    else:
        result = pipeline()
    # Large evidence stays in files; emit small stage metadata only.
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_inventory", "checkpoint_inventory",
        "historical_source_inventory", "code_inventory", "core_trials", "async_trials", "core", "async")}
        if isinstance(result, dict) else result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
