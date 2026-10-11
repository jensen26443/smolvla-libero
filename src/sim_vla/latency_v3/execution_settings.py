"""Bind measured delay estimates and their source bytes to a formal freeze."""
import hashlib
import json
import math

from .schema import METHODS

ESTIMATOR = {"window": 10, "percentile": 95, "tick_seconds": .05}
CONTROL = {"tick_seconds": .05, "cycle_limit_seconds": .055, "required_cycle_fraction": .95}
SNAPSHOT = "frozen/resource_probe.json"


def _initial_estimates(probe):
    if not isinstance(probe, dict):
        raise ValueError("resource probe must contain per-method measurements")  # noqa: TRY004 -- invalid saved schema
    estimates = {}
    for method in METHODS:
        measurement = probe.get(method)
        value = measurement.get("endpoint_p95_s") if isinstance(measurement, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"resource probe requires a finite positive endpoint p95 for {method}")
        estimates[method] = value
    return estimates


def capture_execution_settings(run):
    """Read once; the same bytes supply the hash, snapshot and initial values."""
    content = (run / "audit/resource_probe.json").read_bytes()
    settings = {"format": "latency-v3-execution-settings-v1",
        "resource_probe_sha256": hashlib.sha256(content).hexdigest(),
        "resource_probe_snapshot": SNAPSHOT,
        "initial_endpoint_p95_s": _initial_estimates(json.loads(content)),
        "delay_estimator": dict(ESTIMATOR), "control": dict(CONTROL)}
    return settings, content


def save_probe_snapshot(run, content):
    """Never overwrite evidence left by an earlier freeze."""
    target = run / SNAPSHOT
    if target.exists():
        if target.read_bytes() != content:
            raise ValueError("existing resource probe snapshot differs; use a new run")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(content)


def verify_execution_settings(spec, run):
    settings = spec.get("execution_settings")
    if not isinstance(settings, dict):
        raise ValueError(  # noqa: TRY004 -- missing binding invalidates a persisted specification
            "frozen spec lacks bound execution settings; preserve this historical run and "
            "use SIM_VLA_LATENCY_V3_RUN_ID for a new freeze")
    if (settings.get("format") != "latency-v3-execution-settings-v1"
            or settings.get("resource_probe_snapshot") != SNAPSHOT
            or settings.get("delay_estimator") != ESTIMATOR or settings.get("control") != CONTROL):
        raise ValueError("frozen estimator or control settings differ from the supported protocol")
    for relative in ("audit/resource_probe.json", SNAPSHOT):
        try:
            content = (run / relative).read_bytes()
        except FileNotFoundError as failure:
            raise ValueError(f"frozen resource probe missing: {relative}") from failure
        if hashlib.sha256(content).hexdigest() != settings.get("resource_probe_sha256"):
            raise ValueError(f"frozen resource probe changed: {relative}")
        if _initial_estimates(json.loads(content)) != settings.get("initial_endpoint_p95_s"):
            raise ValueError("frozen initial endpoint p95 differs from the resource probe")
    return settings
