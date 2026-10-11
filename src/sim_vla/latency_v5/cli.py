"""Bounded PLAN-07 execution and offline evidence rebuilding."""
import argparse
import json

from . import experiments
from .layout import RUN_ID, budget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "pipeline", "dry-run", "report", "verify", "recover"))
    command = parser.parse_args().command
    if command == "pipeline":
        from ..latency_v2.locking import run_lock
        from .layout import RUN

        with run_lock(RUN):
            result = experiments.pipeline()
    elif command == "audit":
        result = experiments.audit()
    elif command == "dry-run":
        result = {"environment_attempts": 0, "allocated_from_remaining": 36, "planned_construction": 6,
                  "planned_calibration_resets": 3, "planned_development": 4, "planned_validation": 12,
                  "preflight_attempts_retained": 10, "reserve": 1, "remaining": budget().remaining()}
    elif command == "recover":
        result = {"reconciled": budget().recover(run_id=RUN_ID), "refunded": 0}
    else:
        from .delivery import report, verify

        result = report() if command == "report" else verify()
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("code_inventory", "measurements", "runtime", "gates", "request_distributions")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
