"""Captured event clocks and bounded, nonblocking control-loop logging."""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path


class EventLog(list):
    def __init__(self, *, attempt_id: str, generation: int, directory: Path | None = None,
                 capacity: int = 8192) -> None:
        super().__init__()
        self.attempt_id, self.generation = attempt_id, generation
        self.error = None
        self.channel = queue.Queue(maxsize=capacity)
        self.stopping = threading.Event()
        self.writer = None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            self.writer = threading.Thread(target=self._write, args=(directory / "events.partial.jsonl",), daemon=True)
            self.writer.start()

    def append(self, event: dict) -> None:
        row = {"tick": None, "monotonic_ns": time.monotonic_ns(), "attempt_id": self.attempt_id,
               "generation": self.generation, "request_id": None, **event}
        super().append(row)
        if self.writer is not None:
            try:
                self.channel.put_nowait(row)
            except queue.Full:
                self.error = "event writer queue full; partial crash evidence incomplete"

    def _write(self, path: Path) -> None:
        try:
            with path.open("w") as handle:
                last_sync = time.monotonic()
                while not self.stopping.is_set() or not self.channel.empty():
                    try:
                        row = self.channel.get(timeout=.05)
                    except queue.Empty:
                        row = None
                    if row is not None:
                        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    if time.monotonic() - last_sync >= .25:
                        handle.flush()
                        os.fsync(handle.fileno())
                        last_sync = time.monotonic()
                handle.flush()
                os.fsync(handle.fileno())
        except Exception as failure:  # noqa: BLE001 -- reported by the control owner outside its loop
            self.error = f"event writer failed: {type(failure).__name__}: {failure}"

    def close(self) -> None:
        self.stopping.set()
        if self.writer is not None:
            self.writer.join(timeout=5)
            if self.writer.is_alive():
                self.error = "event writer did not finish within five seconds"


def provenance_complete(events: list[dict], requests: list[dict], *, log_error=None) -> bool:
    required = {"tick", "monotonic_ns", "attempt_id", "generation", "request_id"}
    if log_error or not events or any(not required <= row.keys() or row["monotonic_ns"] is None for row in events):
        return False
    return all(row.get("submit_monotonic_ns") is not None
               and row.get("worker_completed_monotonic_ns") is not None
               and (row.get("received_monotonic_ns") is not None or row.get("simulated_unreleased"))
               for row in requests)


def request_record(request) -> dict:
    return {"attempt_id": request.attempt_id, "generation": request.generation,
            "request_id": request.request_id, "t_obs": request.snapshot.tick, "t_req": request.request_tick,
            "submit_wall": request.submit_wall, "submit_monotonic_ns": request.submit_monotonic_ns,
            "prefix_length": len(request.prefix.actions), "prefix_sha256": request.prefix.digest,
            "observation_sha256": request.snapshot.digest, "sampling": request.snapshot.sampled,
            "snapshot_copy_s": request.snapshot.capture_seconds,
            "worker_completed_monotonic_ns": None, "received_monotonic_ns": None,
            "activation_monotonic_ns": None, "t_act": None}


def result_record(result) -> dict:
    return {"ready_wall": result.ready_wall, "worker_completed_monotonic_ns": result.worker_completed_monotonic_ns,
            "received_monotonic_ns": result.received_monotonic_ns,
            "endpoint_s": result.ready_wall-result.request.submit_wall, "stages": result.stages,
            "sampler": result.sampler, "error": result.error}
