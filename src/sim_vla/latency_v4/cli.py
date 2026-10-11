"""Resumable PLAN-06 stages; gate failures are completed research outcomes."""
import argparse
import json

COMMANDS = ("audit", "profile", "develop-sampling", "develop-period", "sustain", "prepare-bank", "freeze",
            "dry-run", "formal-async", "formal-delay", "formal-ablation", "report", "verify", "recover", "pipeline")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=COMMANDS)
    args = parser.parse_args()
    if args.command == "audit":
        from .audit import audit

        result = audit()
    elif args.command == "profile":
        from .audit import profile

        result = profile()
    else:
        from .experiments import dispatch

        result = dispatch(args.command)
    print(json.dumps({key: value for key, value in result.items()
        if key not in ("execution_assets", "source_inventory_at_audit", "source_inventory", "measurements", "cells")},
        ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
