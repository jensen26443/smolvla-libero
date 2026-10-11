"""Bind every formal parameter before worker startup, reset or attempt billing."""
import hashlib
import math
import time

from ..latency_v2.integrity import file_sha256
from ..latency_v2.storage import canonical_hash
from .continuation import AUTHORIZATION, formal_eligibility
from .experiments import formal_trials
from .layout import (
    BANK,
    CHECKPOINT,
    CONFIG,
    ROOT,
    RUN,
    RUN_ID,
    budget,
    execution_assets,
    immutable_bytes,
    immutable_json,
    progress,
    read,
    source_inventory,
)
from .schema import CONTROL, ESTIMATOR, MethodConfig

FORMAT = "latency-v4-plan06-formal-v1"
PROBE_SNAPSHOT = "frozen/production_probe.json"


def probe_estimates(probe):
    if probe.get("format") != "latency-v4-production-probe-v1" or probe.get("fine_profiler_used") is not False:
        raise ValueError("delay initialization requires an unprofiled production probe")
    if probe.get("effective_calls_per_combination") != 30 or probe.get("warmups_per_combination") != 5:
        raise ValueError("production probe has insufficient or different measurement protocol")
    sources = probe.get("source_inventory")
    if not isinstance(sources, dict) or not sources or any(
            not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for path, digest in sources.items()):
        raise ValueError("production probe source inventory is missing or invalid")
    estimates = probe.get("initial_endpoint_p95_s")
    if not isinstance(estimates, dict) or set(estimates) != {"10", "7", "5"}:
        raise ValueError("missing sampler/step initial endpoint p95 values")
    for values in estimates.values():
        if not isinstance(values, dict) or set(values) != {"native", "rtc"}:
            raise ValueError("incomplete native/RTC probe measurements")
        for value in values.values():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError("probe endpoint p95 must be finite and positive")
    return estimates


def verify_binding(spec, run):
    required = {"format", "run_id", "spec_sha256", "frozen_unix", "frozen_monotonic_ns", "execution_settings",
        "code_inventory", "code_sha256", "execution_assets", "bank_inventory", "bank_manifest_sha256",
        "selection", "selection_sha256", "development_gate_sha256", "matrices", "bootstrap_draws",
        "bootstrap_seed", "main_candidate", "main_baseline", "video_selection", "error_scoring"}
    if not required <= spec.keys() or spec.get("format") != FORMAT:
        raise ValueError("incomplete frozen configuration; missing fields cannot use defaults")
    if canonical_hash({key: value for key, value in spec.items() if key != "spec_sha256"}) != spec["spec_sha256"]:
        raise ValueError("frozen manifest digest mismatch")
    settings = spec["execution_settings"]
    if not isinstance(settings, dict) or set(settings) != {
            "production_probe_sha256", "production_probe_snapshot", "initial_endpoint_p95_s", "delay_estimator", "control"}:
        raise ValueError("incomplete frozen execution settings")
    if (settings["production_probe_snapshot"] != PROBE_SNAPSHOT or settings["delay_estimator"] != ESTIMATOR
            or settings["control"] != CONTROL):
        raise ValueError("unsupported estimator, control or probe snapshot binding")
    for relative in ("profile/production_probe.json", PROBE_SNAPSHOT):
        try:
            content = (run/relative).read_bytes()
        except FileNotFoundError as failure:
            raise ValueError("frozen production probe is missing") from failure
        if hashlib.sha256(content).hexdigest() != settings["production_probe_sha256"]:
            raise ValueError("frozen production probe changed")
        import json

        if probe_estimates(json.loads(content)) != settings["initial_endpoint_p95_s"]:
            raise ValueError("frozen endpoint p95 values disagree with measured bytes")
    for matrix in spec["matrices"].values():
        for trial in matrix:
            cfg = MethodConfig.from_resolved(trial["config"])
            path = "rtc" if cfg.prefix_method else "native"
            if cfg.initial_endpoint_p95_s != settings["initial_endpoint_p95_s"][str(cfg.sampling_steps)][path]:
                raise ValueError("formal method endpoint p95 differs from its frozen sampler probe")
    if (spec["main_candidate"], spec["main_baseline"], spec["bootstrap_draws"], spec["bootstrap_seed"]) != (
            "RTC_CATCHUP", "RTC_STRICT", 10000, 20261005):
        raise ValueError("predeclared main hypothesis or statistical settings changed")
    authority = spec.get("formal_eligibility")
    authorization = run/AUTHORIZATION
    continuation_bound = authorization.exists() or isinstance(authority, dict) and authority.get("mode") == "authorized_continuation"
    if continuation_bound and (not isinstance(authority, dict) or set(authority) != {
                "mode", "development_gate_passed", "authorization_sha256"}
                or authority["mode"] != "authorized_continuation" or authority["development_gate_passed"] is not False
                or not authorization.is_file() or file_sha256(authorization) != authority["authorization_sha256"]):
        raise ValueError("frozen continuation authorization is missing or changed")
    return settings


def freeze():
    target = RUN/"frozen/experiment_spec.json"
    if target.exists():
        return verify_frozen()
    selection = read(RUN/"selection.json")
    gate = read(RUN/"development_gate.json")
    if not selection["passed"]:
        raise RuntimeError("formal freeze requires the declared sampling gate")
    eligibility = formal_eligibility(gate, run=RUN, run_id=RUN_ID)
    if not read(RUN/"physical_restore/complete.json")["passed"]:
        raise ValueError("physical scene restoration has not passed")
    from .bank import verify_bank

    manifest = verify_bank()
    probe_content = (RUN/"profile/production_probe.json").read_bytes()
    import json

    probe = json.loads(probe_content)
    estimates = probe_estimates(probe)
    matrices = formal_trials(selection, probe=probe, manifest=manifest)
    b = budget()
    extra_attempts = max(0, 600-b.remaining()-192)
    remaining_reserve = max(0, 36-extra_attempts)
    if sum(map(len, matrices.values()))+remaining_reserve > b.remaining():
        raise RuntimeError("complete frozen formal matrix and remaining reserve do not fit the additional budget")
    code = source_inventory()
    # layout.py's shared ledger fix changes no sampled input, sampler or GPU
    # work. Retain the original probe source snapshot and bind its difference.
    probe_source_changes = {path: {"measured": digest, "current": code.get(path)}
        for path, digest in probe["source_inventory"].items() if code.get(path) != digest}
    if set(probe_source_changes) - {"src/sim_vla/latency_v4/layout.py"}:
        raise ValueError("production sampler or dependency source changed after its probe")
    for relative, digest in probe["source_inventory"].items():
        snapshot = RUN/"profile/code_snapshot"/relative
        if not snapshot.exists() and code.get(relative) == digest:
            immutable_bytes(snapshot, (ROOT/relative).read_bytes())
        if not snapshot.exists() or file_sha256(snapshot) != digest:
            raise ValueError("original production probe source snapshot changed or missing")
    physical_evidence = [*BANK.parent.rglob("*"), *(RUN/"physical_restore").rglob("*")]
    bank_inventory = {str(path.relative_to(ROOT)): file_sha256(path) for path in physical_evidence
                      if path.is_file() and path.suffix != ".lock"}
    spec = {"format": FORMAT, "run_id": RUN_ID, "checkpoint": CHECKPOINT,
        "frozen_unix": time.time(), "frozen_monotonic_ns": time.monotonic_ns(),
        "code_inventory": code, "code_sha256": canonical_hash(code), "execution_assets": execution_assets(),
        "bank_manifest_sha256": file_sha256(BANK), "bank_inventory": bank_inventory,
        "selection": selection, "selection_sha256": file_sha256(RUN/"selection.json"),
        "development_gate_sha256": file_sha256(RUN/"development_gate.json"), "matrices": matrices,
        "execution_settings": {"production_probe_sha256": hashlib.sha256(probe_content).hexdigest(),
            "production_probe_snapshot": PROBE_SNAPSHOT, "initial_endpoint_p95_s": estimates,
            "delay_estimator": dict(ESTIMATOR), "control": dict(CONTROL)},
        "main_candidate": "RTC_CATCHUP", "main_baseline": "RTC_STRICT",
        "bootstrap_draws": 10000, "bootstrap_seed": 20261005,
        "method_order": "fixed balanced rotation within physical state; all four methods",
        "noise_rule": "same task/state noise seed; different request counts limit noise sequence correspondence",
        "video_selection": {"stage": "formal_async", "state_rank": 0, "methods": ["RAW", "SHIFT", "RTC_STRICT", "RTC_CATCHUP"],
                            "task_segment_seconds": 25, "selection_by_score": False},
        "error_scoring": "method/coverage/timeout failures count as failed; no second score; infrastructure retry retains original attempt",
        "fixed_delay_protocol": "real model and physics, release at t_req+L; L=10/20 is simulated tick delay",
        "historical_ledger_start_sha256": read(b.path)["historical_ledger_start_sha256"],
        "remaining_budget_at_freeze": b.remaining(), "remaining_repair_reserve": remaining_reserve}
    spec["production_probe_support_source_changes"] = probe_source_changes
    spec["formal_eligibility"] = eligibility
    spec["spec_sha256"] = canonical_hash(spec)
    immutable_bytes(RUN/PROBE_SNAPSHOT, probe_content)
    for relative in code:
        immutable_bytes(RUN/"frozen/code_snapshot"/relative, (ROOT/relative).read_bytes())
    immutable_json(CONFIG, spec)
    immutable_json(target, spec)
    progress("frozen", counts={stage: len(matrix) for stage, matrix in matrices.items()}, spec_sha256=spec["spec_sha256"])
    return verify_frozen()


def verify_frozen():
    spec = read(RUN/"frozen/experiment_spec.json")
    verify_binding(spec, RUN)
    if spec["run_id"] != RUN_ID or read(CONFIG) != spec:
        raise ValueError("run or reviewable frozen configuration changed")
    current = source_inventory()
    if current != spec["code_inventory"] or canonical_hash(current) != spec["code_sha256"]:
        raise ValueError("frozen execution/statistics source changed")
    for relative, digest in spec["code_inventory"].items():
        path = RUN/"frozen/code_snapshot"/relative
        if not path.exists() or file_sha256(path) != digest:
            raise ValueError("frozen source snapshot changed or missing")
    probe_source = read(RUN/PROBE_SNAPSHOT)["source_inventory"]
    for relative, digest in probe_source.items():
        snapshot = RUN/"profile/code_snapshot"/relative
        if not snapshot.exists() or file_sha256(snapshot) != digest:
            raise ValueError("measured production probe source snapshot changed or missing")
    for relative, digest in spec["bank_inventory"].items():
        if not (ROOT/relative).exists() or file_sha256(ROOT/relative) != digest:
            raise ValueError("frozen physical state or validation record changed")
    if (file_sha256(BANK) != spec["bank_manifest_sha256"]
            or file_sha256(RUN/"selection.json") != spec["selection_sha256"]
            or file_sha256(RUN/"development_gate.json") != spec["development_gate_sha256"]):
        raise ValueError("frozen selection, eligibility or physical manifest changed")
    if execution_assets() != spec["execution_assets"]:
        raise ValueError("frozen weights, processors, assets or runtime versions changed")
    expected = formal_trials(spec["selection"], probe=read(RUN/PROBE_SNAPSHOT), manifest=read(BANK))
    if expected != spec["matrices"]:
        raise ValueError("frozen trial assignments differ from the PLAN-06 full matrix")
    eligibility = formal_eligibility(read(RUN/"development_gate.json"), run=RUN, run_id=RUN_ID)
    if (RUN/AUTHORIZATION).exists() and spec.get("formal_eligibility") != eligibility:
        raise ValueError("frozen continuation authorization changed or is missing")
    if "formal_eligibility" in spec and spec["formal_eligibility"] != eligibility:
        raise ValueError("frozen formal eligibility differs from its original authority")
    budget()
    return spec
