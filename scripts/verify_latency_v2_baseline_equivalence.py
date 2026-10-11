"""Compare current B_RAW/B_SHIFT timing/actions against the preserved running S1 code."""
import importlib.util
import json

import numpy as np

from sim_vla.assets import project_root
from sim_vla.latency_v2.alignment import Executor
from sim_vla.latency_v2.integrity import file_sha256
from sim_vla.latency_v2.schema import InferenceResult, MethodConfig, ObservationSnapshot
from sim_vla.latency_v2.storage import publish


def trace(executor_type, method, period, latency):
    executor = executor_type(MethodConfig(method, period), generation=1, seed=42, version="same-E1")
    pending, actions = None, []
    for tick in range(300):
        if pending is not None and pending[1] == tick:
            executor.receive(pending[0], tick)
            pending = None
        executor.advance(tick)
        raw = {"pixels": {"image": np.full((4, 4, 3), tick % 256, dtype=np.uint8)}, "robot_state": {}}
        request = executor.submit(ObservationSnapshot.capture("test", tick, raw, "same task"), latency)
        if request is not None:
            chunk = np.zeros((50, 7))
            chunk[:, 0] = ((np.arange(50) + request.request_id) % 50)/50
            chunk[:, 6] = 1 if request.request_id % 2 else -1
            result = InferenceResult(request, chunk, 100.)
            if latency:
                pending = (result, tick+latency)
            else:
                executor.receive(result, tick)
                executor.advance(tick)
        action, _ = executor.action(tick)
        actions.append(action)
    return np.asarray(actions), executor.events, executor.ordinal


def main():
    root = project_root()
    folder = root/"runs/latency_v2_dev_baselines"
    manifest = json.loads((folder/"code_snapshot_manifest.json").read_text())
    path = folder/"code_snapshot/src/sim_vla/latency_v2/alignment.py"
    assert file_sha256(path) == manifest["semantic_code"]["src/sim_vla/latency_v2/alignment.py"]
    spec = importlib.util.spec_from_file_location("sim_vla.latency_v2._s1_snapshot_alignment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    comparisons = []
    for method in ("B_RAW", "B_SHIFT"):
        for period in (5, 10, 20, 25, 40, 50):
            for latency in (0, 2, 5, 10, 20):
                previous = trace(module.Executor, method, period, latency)
                current = trace(Executor, method, period, latency)
                np.testing.assert_array_equal(previous[0], current[0])
                assert previous[1:] == current[1:]
                comparisons.append({"method": method, "period": period, "latency": latency,
                                    "ticks": 300, "exact_actions_and_events_equal": True})
    publish(root/"runs/latency_v2_model_validation/baseline_equivalence.json", {
        "comparison_kind": "mock clock/action-source regression, not additional scored episodes",
        "original_S1_code_sha256": manifest["semantic_code_sha256"], "cases": comparisons,
        "total_case_ticks": 300*len(comparisons), "baseline_behavior_unchanged": True})
    print(json.dumps({"cases": len(comparisons), "baseline_behavior_unchanged": True}))


if __name__ == "__main__":
    main()
