"""Finite engineering validation, frozen before scores, using one live job."""
import pickle
import time

import numpy as np

from ..latency_v2.integrity import file_sha256
from ..latency_v2.schema import CommittedPrefix, ObservationSnapshot, digest_arrays, frozen_copy
from ..latency_v2.storage import canonical_hash, publish, trial_key
from ..latency_v2.worker import InferenceWorker
from ..latency_v4.layout import MODEL_SHA
from ..latency_v4.schema import MethodConfig, Request
from ..tasks import TASK_BY_KEY
from .env import StateOnlyEnv
from .layout import (
    CHECKPOINT,
    CONFIG,
    METHODS,
    PLAN,
    ROOT,
    RUN,
    RUN_ID,
    RUNTIME,
    budget,
    immutable_bytes,
    immutable_json,
    progress,
    read,
    seed,
    snapshot_sources,
    sources,
)
from .runner import run_episode

STAGES = ("development", "validation")
DEVELOPMENT_TASKS = tuple(TASK_BY_KEY)[:2]


def audit():
    budget()
    target = RUN / "audit_v3.json"
    if target.exists():
        result = read(target)
        if result["plan_sha256"] != file_sha256(PLAN) or sources() != result["code_inventory"]:
            raise ValueError("audited plan or runtime source changed")
        return result
    if file_sha256(ROOT / CHECKPOINT / "model.safetensors") != MODEL_SHA:
        raise ValueError("the pinned E1 model changed")
    inventory = snapshot_sources(RUN / "audit_v3/code_snapshot")
    result = {"run_id": RUN_ID, "plan_sha256": file_sha256(PLAN), "model_sha256": MODEL_SHA,
              "code_inventory": inventory, "code_sha256": canonical_hash(inventory),
              "allocation_sha256": file_sha256(RUN / "allocation.json"), "environment_attempts": 0,
              "original_protocol_complete": False,
              "scope": "engineering follow-up; original PLAN-06 failed gates retained"}
    immutable_json(target, result)
    return result


def config(method, probe):
    return MethodConfig(method, 5, 25, probe["initial_endpoint_p95_s"]["rtc"],
                        late_policy="catchup" if method == "RTC_CATCHUP" else "strict").resolved()


def matrices(probe):
    values = {}
    for stage, indices in (("development", (0,)), ("validation", (2, 3))):
        values[stage] = []
        for index in indices:
            for task in (DEVELOPMENT_TASKS if stage == "development" else tuple(TASK_BY_KEY)):
                offset = index % 2
                for method in METHODS[offset:] + METHODS[:offset]:
                    values[stage].append({"task": task, "init_index": index, "seed": seed(task, index),
                        "state_id": f"previously_exposed_native_init{index}", "latency": 0,
                        "profile": "actual_async", "config": config(method, probe), "runtime": dict(RUNTIME)})
    return values


def rows(stage, *, complete=True):
    path = RUN / "frozen/experiment_spec.json"
    if not path.exists():
        return []
    expected = read(path)["matrices"][stage]
    output = []
    for trial in expected:
        result = RUN / stage / "trials" / trial_key(trial) / "result.json"
        if not result.exists():
            if complete:
                raise ValueError("incomplete stage: " + stage)
            continue
        value = read(result)
        if (any(value.get(key) != expected_value for key, expected_value in trial.items())
                or value["effective_config"] != trial["config"] or value["effective_runtime"] != RUNTIME):
            raise ValueError("saved result differs from the declared trial")
        output.append(value)
    return output


def freeze(probe):
    target = RUN / "frozen/experiment_spec.json"
    if target.exists():
        return verify_frozen()
    code = snapshot_sources(RUN / "frozen/code_snapshot")
    if code != probe["code_inventory"]:
        raise ValueError("measured sampler or scene renderer changed after its production probe")
    if not all(read(RUN / "calibration_v3" / task / "result.json")["pixel_equivalence_verified"] for task in TASK_BY_KEY):
        raise ValueError("all three tasks require same-scene pixel equivalence")
    calibration = {str(path.relative_to(ROOT)): file_sha256(path) for path in (RUN / "calibration_v3").rglob("*")
                   if path.is_file() and path.suffix != ".lock"}
    immutable_bytes(RUN / "frozen/production_probe.json", (RUN / "profile_v3/production_probe.json").read_bytes())
    spec = {"format": "latency-v5-engineering-freeze-v3", "run_id": RUN_ID,
        "frozen_unix": time.time(), "frozen_monotonic_ns": time.monotonic_ns(),
        "code_inventory": code, "code_sha256": canonical_hash(code), "model_sha256": MODEL_SHA,
        "runtime": dict(RUNTIME), "probe_sha256": file_sha256(RUN / "profile_v3/production_probe.json"),
        "initial_endpoint_p95_s": probe["initial_endpoint_p95_s"], "calibration_inventory": calibration,
        "matrices": matrices(probe), "allocation_sha256": file_sha256(RUN / "allocation.json"),
        "plan_sha256": file_sha256(PLAN),
        "sustain_gate": {"minimum_full_300_per_method": 4, "n_per_method": 6,
                         "minimum_prefix_activations_per_task": 3},
        "development_diagnostic": {"n_per_method": 2, "minimum_full_300_per_method": 2,
                                   "minimum_prefix_activations_per_task": 3,
                                   "tasks": list(DEVELOPMENT_TASKS)},
        "realtime_gate": {"required_cycle_fraction": .95, "cycle_limit_seconds": .055,
                          "step_complete": True},
        "method_order": "balanced rotation by init index, same paired seed within task/init",
        "error_scoring": "retain method failure; never give completed method outcomes a second score",
        "state_scope": "fixed previously exposed simulator native states; no training-novelty claim"}
    spec["spec_sha256"] = canonical_hash(spec)
    immutable_json(CONFIG, spec)
    immutable_json(target, spec)
    progress("frozen", code_sha256=spec["code_sha256"], counts={stage: len(items) for stage, items in spec["matrices"].items()})
    return verify_frozen()


def verify_frozen():
    spec = read(RUN / "frozen/experiment_spec.json")
    required = {"format", "run_id", "frozen_monotonic_ns", "code_inventory", "code_sha256", "runtime",
                "probe_sha256", "initial_endpoint_p95_s", "calibration_inventory", "matrices",
                "allocation_sha256", "plan_sha256", "spec_sha256", "sustain_gate", "realtime_gate", "model_sha256",
                "development_diagnostic"}
    if not required <= spec.keys() or spec["format"] != "latency-v5-engineering-freeze-v3" or spec["run_id"] != RUN_ID:
        raise ValueError("incomplete frozen runtime; missing values cannot use defaults")
    if canonical_hash({key: value for key, value in spec.items() if key != "spec_sha256"}) != spec["spec_sha256"]:
        raise ValueError("frozen specification changed")
    if read(CONFIG) != spec or spec["runtime"] != RUNTIME:
        raise ValueError("reviewable configuration or runtime differs from the frozen version")
    if (spec["model_sha256"] != MODEL_SHA
            or spec["sustain_gate"] != {"minimum_full_300_per_method": 4, "n_per_method": 6,
                                        "minimum_prefix_activations_per_task": 3}
            or spec["realtime_gate"] != {"required_cycle_fraction": .95, "cycle_limit_seconds": .055,
                                         "step_complete": True}
            or spec["development_diagnostic"] != {"n_per_method": 2, "minimum_full_300_per_method": 2,
                                                   "minimum_prefix_activations_per_task": 3,
                                                   "tasks": list(DEVELOPMENT_TASKS)}):
        raise ValueError("frozen model or acceptance thresholds changed")
    for relative in ("profile_v3/production_probe.json", "frozen/production_probe.json"):
        if file_sha256(RUN / relative) != spec["probe_sha256"]:
            raise ValueError("frozen production probe changed")
    probe = read(RUN / "frozen/production_probe.json")
    if (spec["initial_endpoint_p95_s"] != probe["initial_endpoint_p95_s"]
            or probe["warmups_per_combination"] != 5 or probe["effective_calls_per_combination"] != 30
            or probe["fine_profiler_used"]):
        raise ValueError("frozen delay estimates differ from the production measurements")
    current = sources()
    if current != spec["code_inventory"] or canonical_hash(current) != spec["code_sha256"]:
        raise ValueError("frozen runtime or statistics source changed")
    for name, digest in current.items():
        if file_sha256(RUN / "frozen/code_snapshot" / name) != digest:
            raise ValueError("frozen source snapshot changed")
    for name, digest in spec["calibration_inventory"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError("frozen calibration scene or reference image changed")
    if (file_sha256(PLAN) != spec["plan_sha256"]
            or file_sha256(RUN / "allocation.json") != spec["allocation_sha256"]
            or matrices(probe) != spec["matrices"]):
        raise ValueError("frozen authorization, plan, matrix, seed or method configuration changed")
    budget()
    return spec


def create_control_env(task):
    target = RUN / "construction/control" / str(time.monotonic_ns()) / task
    with budget().track(f"control:{task}", "control_construction", target, {"task": task}, run_id=RUN_ID):
        env = StateOnlyEnv(task_key=task)
        publish(target / "result.json", {"task": task, "constructed": True, "explicit_resets": 0})
    return env


def calibrate(envs, engine):
    target = RUN / "profile_v3/production_probe.json"
    if target.exists():
        return read(target)
    snapshots, references = {}, {}
    for task in TASK_BY_KEY:
        directory = RUN / "calibration_v3" / task
        if task not in envs:
            envs[task] = create_control_env(task)
        if not (directory / "result.json").exists():
            with budget().track(f"calibration:{task}", "calibration_reset", directory,
                                {"task": task, "init_index": 0, "seed": seed(task, 0)}, run_id=RUN_ID):
                envs[task].start_episode(0, seed(task, 0))
                snapshot = envs[task].freeze_snapshot(0, "calibration-" + task)
                reference = envs[task].reference_observation(0)
                with (directory / "scene.pkl").open("wb") as handle:
                    pickle.dump(snapshot, handle, protocol=5)
                np.savez_compressed(directory / "reference_images.npz", **reference["pixels"])
                reference_packet = ObservationSnapshot.capture(snapshot.episode, 0, reference, snapshot.language)
                publish(directory / "result.json", {"task": task, "scene_sha256": snapshot.scene_sha256,
                    "visual_sha256": snapshot.visual_sha256,
                    "scene_file_sha256": file_sha256(directory / "scene.pkl"),
                    "reference_images_sha256": file_sha256(directory / "reference_images.npz"),
                    "reference_observation_sha256": reference_packet.digest,
                    "pixel_equivalence_verified": False, "state_capture_seconds": snapshot.capture_seconds})
        with (directory / "scene.pkl").open("rb") as handle:
            snapshots[task] = pickle.load(handle)
        with np.load(directory / "reference_images.npz", allow_pickle=False) as data:
            references[task] = {key: data[key].copy() for key in data.files}
    code = snapshot_sources(RUN / "profile_v3/code_snapshot")
    records, guides = [], {}
    options = [(task, method) for task in TASK_BY_KEY for method in ("RAW", "RTC_STRICT")]
    ordinal = 0
    started = time.monotonic()
    for round_index in range(35):
        offset = round_index % len(options)
        for task, method in options[offset:] + options[:offset]:
            snapshot = snapshots[task]
            actions = np.zeros((0, 7)) if method == "RAW" else guides[task][:20]
            prefix = CommittedPrefix(0, frozen_copy(actions), (), digest_arrays(actions))
            request = Request(-1, ordinal, snapshot, 0, prefix, seed(task, 0), time.monotonic(),
                              sampler_mode=method, sampling_steps=5,
                              guide_actions=None if method == "RAW" else frozen_copy(guides[task][:23]))
            ordinal += 1
            engine.submit(request)
            result = engine.drain()
            if result.error or result.sampler.get("steps") != 5 or result.sampler.get("parameter_gradients_present"):
                raise RuntimeError(result.error or "sampler or gradient verification failed")
            if method == "RAW" and task not in guides:
                guides[task] = result.actions.copy()
                differences = {key: int(np.max(np.abs(result.rendered_images[key].astype(int) - value.astype(int))))
                               for key, value in references[task].items()}
                directory = RUN / "calibration_v3" / task
                np.savez_compressed(directory / "worker_images.npz", **result.rendered_images)
                source = directory / "result.json"
                value = read(source)
                value.update(max_pixel_error=differences,
                             worker_images_sha256=file_sha256(directory / "worker_images.npz"),
                             worker_observation_sha256=result.sampler["rendered_observation_sha256"])
                publish(source, value)
                if any(differences.values()):
                    raise ValueError("worker cameras differ from the same-scene reference: " + str(differences))
                if value["reference_observation_sha256"] != result.sampler["rendered_observation_sha256"]:
                    raise ValueError("worker altered the frozen model input state or image layout")
                value.update(pixel_equivalence_verified=True)
                publish(source, value)
            records.append({"task": task, "method": method, "round": round_index, "warmup": round_index < 5,
                "endpoint_s": result.ready_wall - request.submit_wall, "stages": result.stages,
                "sampler": result.sampler, "control_receive_s":
                (result.received_monotonic_ns - result.worker_completed_monotonic_ns) / 1e9})
        publish(RUN / "profile_v3/measurements.partial.json", records)
        progress("production_probe", completed_calls=ordinal, total_calls=210)
    estimates = {label: float(np.percentile([row["endpoint_s"] for row in records
                 if row["method"] == method and not row["warmup"]], 95))
                 for label, method in (("native", "RAW"), ("rtc", "RTC_STRICT"))}
    result = {"format": "latency-v5-production-probe-v1", "measurements": records,
              "initial_endpoint_p95_s": estimates, "warmups_per_combination": 5,
              "effective_calls_per_combination": 30, "fine_profiler_used": False,
              "code_inventory": code, "total_wall_s": time.monotonic() - started,
              "environment_resets": 3, "model_calls": 210, "prefix_probe_length": 20,
              "scope": "existing frozen calibration scenes; renderer+model in the real worker endpoint"}
    immutable_json(target, result)
    return result


def run_stage(stage, envs, engine):
    spec = verify_frozen()  # Always before billing or reset, including resume.
    for trial in spec["matrices"][stage]:
        target = RUN / stage / "trials" / trial_key(trial)
        if (target / "result.json").exists():
            continue
        task = trial["task"]
        if task not in envs:
            envs[task] = create_control_env(task)
        budget().prepare_retry(target)
        with budget().track(trial_key(trial), stage, target, trial, run_id=RUN_ID) as generation:
            result = run_episode(envs[task], engine, trial, target, asynchronous=True, sustain=True,
                                 generation=generation, version=spec["code_sha256"],
                                 video=stage == "validation" and trial["init_index"] == 2)
        progress(stage, completed=len(rows(stage, complete=False)), total=len(spec["matrices"][stage]), task=task,
                 method=trial["config"]["method"], steps=result["steps"], reason=result["termination_reason"],
                 realtime_gate=result["realtime_20hz_passed"])
        if result["error"]:
            raise RuntimeError("method error retained; stop before modifying frozen control semantics")
    immutable_json(RUN / stage / "complete.json", {"completed": len(rows(stage)), "code_sha256": spec["code_sha256"]})
    return {"stage": stage, "completed": len(rows(stage))}


def pipeline():
    audit()
    from .delivery import report, verify

    if (RUN / "validation/complete.json").exists():
        ledger_before = file_sha256(budget().path)
        verify_frozen()
        verify()
        if file_sha256(budget().path) != ledger_before:
            raise RuntimeError("completed resume changed the attempt ledger")
        immutable_json(RUN / "testing/resume_check.json", {"passed": True, "attempts_added": 0,
            "ledger_before_and_after_sha256": ledger_before, "all_results_reused": True})
        result = report()
        return {"skipped_completed": True, "environment_attempts_added": 0, **result}
    expected = 16 if not (RUN / "frozen/experiment_spec.json").exists() else sum(
        len(read(RUN / "frozen/experiment_spec.json")["matrices"][stage]) - len(rows(stage, complete=False))
        for stage in STAGES)
    startup_resets = 9 if not (RUN / "profile_v3/production_probe.json").exists() else 6
    if budget().remaining() < expected + startup_resets:
        raise RuntimeError("remaining fixed trials and charged environment construction cannot fit the allocated balance")
    engine = InferenceWorker(factory_name="sim_vla.latency_v5.policy:RenderPolicyAdapter",
        factory_kwargs={"checkpoint": ROOT / CHECKPOINT, "sampling_steps": 5, "warmup_calls": 3}, startup_timeout=180.)
    envs = {}
    started = time.monotonic_ns()
    try:
        probe = calibrate(envs, engine)
        freeze(probe)
        for stage in STAGES:
            run_stage(stage, envs, engine)
    finally:
        for env in envs.values():
            env.focus()
            env.close()
        publish(RUN / "worker_shutdown.json", engine.close())
        publish(RUN / "job_accounting.json", {"wall_s": (time.monotonic_ns() - started) / 1e9,
                                               "scope": "calibration, reset, startup/control/tail, serialization, shutdown"})
    result = report()
    verify()
    progress("complete", sustained_execution_supported=result["sustained_execution_supported"],
             realtime_gate_passed=result["realtime_gate_passed"])
    return result
