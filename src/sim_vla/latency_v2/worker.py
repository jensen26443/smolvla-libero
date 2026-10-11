"""A spawned, single-flight GPU owner with bounded queues and explicit cleanup."""
from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import queue
import time
import traceback
from collections import deque

import numpy as np

from .schema import InferenceResult


def _worker_main(requests, results, factory_name: str, kwargs: dict) -> None:
    started_unix, started_ns = time.time(), time.monotonic_ns()
    try:
        module, attribute = factory_name.rsplit(":", 1)
        factory = getattr(importlib.import_module(module), attribute)
        if factory_name.endswith(":PrefixPolicyAdapter"):
            import torch

            torch.set_num_threads(4)
        policy = factory(**kwargs)
        policy.warmup()
        results.put({"startup": "ready", "pid": os.getpid(), "started_unix": started_unix,
                     "started_monotonic_ns": started_ns})
        generation = None
        while True:
            request = requests.get()
            if request is None:
                break
            try:
                if generation != request.generation:
                    policy.reset()
                    generation = request.generation
                result = policy.infer_request(request)
            except Exception:  # noqa: BLE001 -- transfer original worker failure to the evaluator
                result = InferenceResult(request, None, time.monotonic(), error=traceback.format_exc())
            results.put(result)
        results.put({"shutdown": "complete", "finished_unix": time.time(),
                     "finished_monotonic_ns": time.monotonic_ns()})
    except Exception:  # noqa: BLE001 -- startup errors cannot be swallowed or mistaken for ready
        results.put({"startup": "error", "error": traceback.format_exc()})


class InferenceWorker:
    def __init__(self, checkpoint=None, *, factory_name="sim_vla.latency_v2.policy:PrefixPolicyAdapter",
                 factory_kwargs=None, startup_timeout=60.0) -> None:
        context = mp.get_context("spawn")  # CUDA/GL contexts must never be forked.
        self.requests, self.results = context.Queue(maxsize=1), context.Queue(maxsize=1)
        kwargs = factory_kwargs if factory_kwargs is not None else {"checkpoint": checkpoint, "warmup_calls": 3}
        self.process = context.Process(target=_worker_main, args=(self.requests, self.results, factory_name, kwargs))
        self.process.start()
        self.busy = False
        self.submitted = None
        self.worker_metadata = {"pid": self.process.pid}
        self.shutdown_metadata = None
        try:
            ready = self.results.get(timeout=startup_timeout)
            if ready["startup"] != "ready":
                raise RuntimeError(ready["error"])
            self.worker_metadata.update({key: value for key, value in ready.items() if key != "startup"})
        except (queue.Empty, RuntimeError):
            self.close()
            raise

    def submit(self, request) -> None:
        if self.busy:
            raise RuntimeError("only one inference may be in flight")
        self.requests.put_nowait(request)
        self.busy, self.submitted = True, request

    def poll(self, *, timeout: float = 30.0):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            if self.busy and (not self.process.is_alive() or time.monotonic()-self.submitted.submit_wall > timeout):
                raise TimeoutError("inference worker timed out or exited")
            return None
        self.busy = False
        if not isinstance(result, InferenceResult):
            raise TypeError(f"unexpected worker message: {result}")
        result.received_monotonic_ns = time.monotonic_ns()
        return result

    def drain(self, timeout: float = 30.0):
        start = time.monotonic()
        while self.busy:
            result = self.poll(timeout=timeout)
            if result is not None:
                return result
            if time.monotonic()-start > timeout:
                raise TimeoutError("worker drain timeout")
            time.sleep(.005)
        return None

    def close(self) -> dict:
        if self.shutdown_metadata is not None:
            return self.shutdown_metadata
        mode, acknowledgement = "graceful", None
        if self.process.is_alive():
            try:
                self.requests.put_nowait(None)
            except queue.Full:
                pass
            self.process.join(timeout=5)
        if self.process.is_alive():
            mode = "terminate"
            self.process.terminate()
            self.process.join(timeout=5)
        if self.process.is_alive():
            mode = "kill"
            self.process.kill()
            self.process.join(timeout=2)
        if mode == "graceful":
            try:
                message = self.results.get_nowait()
            except queue.Empty:
                pass
            else:
                if isinstance(message, dict) and message.get("shutdown") == "complete":
                    acknowledgement = message
        self.shutdown_metadata = {**self.worker_metadata, "termination_mode": mode,
            "event": "worker_shutdown_observed", "tick": None, "monotonic_ns": time.monotonic_ns(),
            "finished_unix": acknowledgement.get("finished_unix") if acknowledgement else None,
            "finished_monotonic_ns": acknowledgement.get("finished_monotonic_ns") if acknowledgement else None,
            "end_time_status": "known" if acknowledgement else "unknown",
            "exit_observed_unix": time.time(), "exit_observed_monotonic_ns": time.monotonic_ns(),
            "attempt_id": self.submitted.attempt_id if self.submitted else None,
            "generation": self.submitted.generation if self.submitted else None,
            "request_id": self.submitted.request_id if self.submitted else None}
        for channel in (self.requests, self.results):
            channel.cancel_join_thread()
            channel.close()
        return self.shutdown_metadata


class CausalDelayEstimator:
    def __init__(self, initial_p95: float = 1.10448796484, window: int = 10, tick_seconds: float = .05) -> None:
        self.initial, self.tick_seconds = initial_p95, tick_seconds
        self.completed = deque(maxlen=window)

    def observe(self, result: InferenceResult) -> None:
        if result.error is None:
            self.completed.append(max(0, result.ready_wall-result.request.submit_wall))

    def ticks(self) -> int:
        seconds = float(np.percentile(self.completed, 95)) if self.completed else self.initial
        return int(np.ceil(seconds/self.tick_seconds))
