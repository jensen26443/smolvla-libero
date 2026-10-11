"""Durable attempts, atomic completed results and immutable experiment specs."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import time
import traceback
from contextlib import contextmanager
from pathlib import Path


def process_start_ticks(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def publish(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


class Budget:
    def __init__(self, path: Path, *, cap: int = 1800, initial: int = 18) -> None:
        self.path, self.cap = path, cap
        if not path.exists():
            publish(path, {"cap": cap, "initial_s0_and_profile_attempts": initial, "attempts": []})

    def claim(self, key: str, stage: str, *, run_id: str | None = None, target: Path | None = None) -> int:
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text())
            used = data["initial_s0_and_profile_attempts"] + len(data["attempts"])
            if used >= data["cap"]:
                raise RuntimeError("episode budget exhausted")
            identity = run_id or stage
            run_cap = data.get("run_attempt_caps", {}).get(identity)
            if run_cap is not None and sum(row.get("run_id") == identity for row in data["attempts"]) >= run_cap:
                raise RuntimeError("this run exhausted its reserved attempt budget")
            data["attempts"].append({"key": key, "trial_key": key, "stage": stage,
                "run_id": run_id or stage, "attempt_id": f"attempt-{used + 1:06d}",
                "generation": used + 1, "started_unix": time.time(),
                "started_monotonic_ns": time.monotonic_ns(), "finished_unix": None,
                "finished_monotonic_ns": None, "end_time_status": "pending", "status": "running",
                "exception": None, "pid": os.getpid(), "hostname": socket.gethostname(),
                "owner_start_ticks": process_start_ticks(os.getpid()),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "target": str(target.resolve()) if target else None, "attempt_number": used + 1})
            publish(self.path, data)
            return used + 1

    def reserve(self, run_id: str, cap: int) -> None:
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text())
            caps = data.setdefault("run_attempt_caps", {})
            if run_id in caps and caps[run_id] != cap:
                raise ValueError("run attempt reservation cannot change silently")
            caps[run_id] = cap
            publish(self.path, data)

    def update(self, number: int, **fields) -> dict:
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text())
            row = next(row for row in data["attempts"] if row["attempt_number"] == number)
            row.update(fields)
            publish(self.path, data)
            return dict(row)

    @contextmanager
    def track(self, key: str, stage: str, target: Path, trial: dict, *, run_id: str):
        number = self.claim(key, stage, run_id=run_id, target=target)
        row = self.update(number)
        try:
            publish(target / "attempt.json", {**row, "trial": trial})
            yield number
            result_path = target / "result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
        except BaseException as failure:
            status = "interrupted" if isinstance(failure, (KeyboardInterrupt, SystemExit)) else "infrastructure_error"
            self.finish(number, target, trial, status=status, exception={
                "type": type(failure).__name__, "message": str(failure), "traceback": traceback.format_exc()})
            raise
        else:
            self.finish(number, target, trial, status="method_error" if result.get("error") else "completed",
                        exception={"type": "MethodError", "traceback": result["error"]} if result.get("error") else None)

    def finish(self, number: int, target: Path, trial: dict, *, status: str, exception=None) -> None:
        row = self.update(number, status=status, exception=exception, finished_unix=time.time(),
                          finished_monotonic_ns=time.monotonic_ns(), end_time_status="known")
        publish(target / "attempt.json", {**row, "trial": trial})

    def recover(self, *, run_id: str, classification: str | None = None) -> list[dict]:
        """Reconcile dead owners only; never invent their end clocks or refund attempts."""
        data = json.loads(self.path.read_text())
        recovered = []
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        for old in data["attempts"]:
            if old.get("run_id") != run_id or old.get("target") is None:
                continue  # Legacy rows retain missing fields.
            if old.get("status") == "running":
                if old.get("hostname") != socket.gethostname():
                    raise RuntimeError("cannot classify an owner on a different host")
                if (old.get("boot_id") == boot and process_start_ticks(old["pid"]) is not None
                        and process_start_ticks(old["pid"]) == old.get("owner_start_ticks")):
                    raise RuntimeError("attempt owner is still alive")
                target = Path(old["target"])
                outcome = target / "result.json"
                fields = {"status": "interrupted_unclassified", "end_time_status": "unknown",
                          "finished_unix": None, "finished_monotonic_ns": None,
                          "recovery_observed_unix": time.time(), "exception": {
                              "type": "OwnerExited", "message": "process ended; actual end time unknown"}}
                if outcome.exists():
                    result = json.loads(outcome.read_text())
                    fields["status"] = "method_error" if result.get("error") else "completed"
                old = self.update(old["attempt_number"], **fields)
            target = Path(old["target"])
            if old.get("status") in {"interrupted_unclassified", "interrupted", "infrastructure_error"} and classification:
                if (target / "result.json").exists():
                    raise ValueError("completed outcomes cannot be reclassified for retry")
                old = self.update(old["attempt_number"], classification=classification,
                                  classification_unix=time.time())
            attempt_path = target / "attempt.json"
            trial = json.loads(attempt_path.read_text()).get("trial") if attempt_path.exists() else None
            if trial is None:
                expected = target.parent.parent / "expected_trials.json"
                if expected.exists():
                    trial = next(({k: v for k, v in item.items() if k != "key"}
                                  for item in json.loads(expected.read_text()) if item["key"] == old["key"]), None)
            publish(attempt_path, {**old, "trial": trial})
            recovered.append(old)
        return recovered

    def remaining(self) -> int:
        data = json.loads(self.path.read_text())
        return data["cap"] - data["initial_s0_and_profile_attempts"] - len(data["attempts"])

    def prepare_retry(self, target: Path) -> None:
        path = target / "attempt.json"
        if not path.exists():
            return
        row = json.loads(path.read_text())
        if row.get("classification") != "infrastructure_error" or row.get("status") not in {
                "interrupted_unclassified", "interrupted", "infrastructure_error"}:
            raise RuntimeError("interrupted attempt must be explicitly classified; preserve evidence before retry")
        archive = target.parent.parent / "interrupted" / (target.name + "_" + row["attempt_id"])
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            raise FileExistsError("interrupted attempt archive already exists")
        target.rename(archive)
        self.update(row["attempt_number"], target=str(archive.resolve()), archived_for_retry=True)


def trial_key(trial: dict) -> str:
    return canonical_hash(trial)[:24]


def prepare_run(directory: Path, spec: dict, trials: list[dict], *, resume: bool = False) -> dict:
    keys = [trial_key(trial) for trial in trials]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate expected trial key")
    frozen = {**spec, "expected_trials": trials, "expected_count": len(trials)}
    frozen["spec_sha256"] = canonical_hash(frozen)
    path = directory / "spec.json"
    if path.exists():
        if not resume:
            raise FileExistsError("run already exists; explicit --resume required")
        if json.loads(path.read_text()) != frozen:
            raise ValueError("resume semantics differ from frozen spec")
    else:
        publish(path, frozen)
        publish(directory / "expected_trials.json", [{"key": key, **trial} for key, trial in zip(keys, trials)])
    return frozen
