"""Explicit, immutable continuation after a failed gate; never relabel it passed."""
from pathlib import Path

from ..latency_v2.integrity import file_sha256
from ..latency_v2.storage import canonical_hash
from .layout import BUDGET_PATH, ROOT, RUN, RUN_ID, read

FORMAT = "latency-v4-plan06-continuation-v1"
AUTHORIZATION = "continuation/authorization.json"
DEVELOPMENT = {"develop_sampling": 36, "develop_period": 72, "sustain": 18}


def verify_continuation(*, run=RUN, run_id=RUN_ID):
    path = run/AUTHORIZATION
    if not path.exists():
        raise RuntimeError("declared sustained gate failed; no explicit continuation authorization")
    record = read(path)
    required = {"format", "run_id", "parent_run_id", "user_instruction", "authority", "mode",
        "additional_budget_cap", "budget_used_at_authorization", "remaining_matrix", "repair_reserve",
        "selection", "development_gate", "imported_files", "runtime_sources", "record_sha256",
        "budget_snapshot_sha256", "parent_attempt_prefix_sha256"}
    if not required <= record.keys() or record["format"] != FORMAT:
        raise ValueError("incomplete continuation authorization")
    if canonical_hash({key: value for key, value in record.items() if key != "record_sha256"}) != record["record_sha256"]:
        raise ValueError("continuation authorization changed")
    if (record["run_id"] != run_id or record["parent_run_id"] == run_id
            or record["user_instruction"] != "继续完成剩余任务"
            or record["authority"] != "explicit_user_instruction_after_failed_gate"
            or record["mode"] != "fixed_matrix_extension_with_failed_gate_retained"
            or record["additional_budget_cap"] != 600 or record["budget_used_at_authorization"] != 126
            or record["repair_reserve"] != 36
            or record["remaining_matrix"] != {"prepare_bank": 60, "physical_restore": 6,
                                               "formal_async": 240, "formal_delay": 72, "formal_ablation": 60}):
        raise ValueError("continuation scope, identity or budget changed")
    parent = (ROOT/"runs"/record["parent_run_id"]).resolve()
    if parent.parent != (ROOT/"runs").resolve() or record["parent_run_id"] != "latency_v4_plan06":
        raise ValueError("invalid continuation parent run")
    snapshot = run/"continuation/budget_at_authorization.json"
    ledger = read(BUDGET_PATH)
    if (not snapshot.is_file() or file_sha256(snapshot) != record["budget_snapshot_sha256"]
            or len(read(snapshot)["attempts"]) != 126 or ledger["cap"] != 600
            or canonical_hash(ledger["attempts"][:126]) != record["parent_attempt_prefix_sha256"]):
        raise ValueError("continuation changed historical attempts or its shared budget")
    if not record["imported_files"] or not record["runtime_sources"]:
        raise ValueError("continuation requires original measurements and runtime source identity")
    for relative, digest in record["imported_files"].items():
        local = (run/relative).resolve()
        original = (parent/relative).resolve()
        if (not local.is_relative_to(run.resolve()) or not original.is_relative_to(parent)
                or not local.is_file() or not original.is_file()
                or file_sha256(local) != digest or file_sha256(original) != digest):
            raise ValueError("imported original development evidence changed: "+relative)
    for relative, digest in record["runtime_sources"].items():
        source = (ROOT/relative).resolve()
        if not source.is_relative_to(ROOT.resolve()) or not source.is_file() or file_sha256(source) != digest:
            raise ValueError("continuation changed a measured controller or sampler: "+relative)
    selection, gate = read(run/"selection.json"), read(run/"development_gate.json")
    if selection != record["selection"] or gate != record["development_gate"]:
        raise ValueError("continuation changed the original selected parameters or failed gate")
    if (not selection["passed"] or selection["sampling_steps"] != 5
            or selection["periods"] != {"RAW": 50, "SHIFT": 10, "RTC_STRICT": 25, "RTC_CATCHUP": 25}
            or gate["passed"] or gate["methods"]["RTC_STRICT"]["full_300_ticks"] != 3
            or gate["methods"]["RTC_CATCHUP"]["full_300_ticks"] != 6
            or not gate["methods"]["RTC_CATCHUP"]["passed"]
            or any(item["invariant_errors"] for item in gate["methods"].values())):
        raise ValueError("continuation cannot waive implementation errors or alter the observed gate")
    for stage, count in DEVELOPMENT.items():
        expected = read(run/stage/"expected_trials.json")
        spec = read(run/stage/"spec.json")
        if len(expected) != count or spec["run_id"] != record["parent_run_id"] or not all(
                (run/stage/"trials"/item["key"]/"result.json").exists() for item in expected):
            raise ValueError("continuation imports an incomplete or different development stage")
    if (read(run/"profile/production_probe.json")["environment_attempts"] != 0
            or not read(run/"profile/mechanism.json")["passed"]):
        raise ValueError("continuation production probe or mechanism evidence is invalid")
    return record


def formal_eligibility(gate, *, run=RUN, run_id=RUN_ID):
    if (run/AUTHORIZATION).exists():
        verify_continuation(run=run, run_id=run_id)
        return {"mode": "authorized_continuation", "development_gate_passed": False,
                "authorization_sha256": file_sha256(run/AUTHORIZATION)}
    if not gate["passed"]:
        raise RuntimeError("declared sustained gate failed; no explicit continuation authorization")
    return {"mode": "original_gate_passed", "development_gate_passed": True,
            "authorization_sha256": None}


def original_trial_directory(stage, spec, key, *, run=RUN):
    """Imported trials still belong to their original charged run."""
    if (run/AUTHORIZATION).exists() and stage in DEVELOPMENT:
        record = read(run/AUTHORIZATION)
        if spec["run_id"] != record["parent_run_id"]:
            raise ValueError("imported trial source run changed")
    return Path(ROOT/"runs"/spec["run_id"]/stage/"trials"/key).resolve()
