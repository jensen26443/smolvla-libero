"""Timeline evidence artifacts: per-tick CSV plus a rendered event chart.

The CSV/PNG pair is the human-checkable proof that the discrete-event loop
releases observations, requests, completions, activations, and actions at the
right ticks (PLAN-03 §8). Rows are compact; detailed arrays stay in .npz.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .runner import EpisodeTrace

TIMELINE_FIELDS = [
    "tick",
    "arrived_packets",
    "latest_packet",
    "request_id",
    "request_reason",
    "suppressed",
    "completed_requests",
    "activated_plan",
    "action_source",
    "a0",
    "a1",
    "fallback_reason",
    "active_plan",
    "active_plan_age",
    "input_obs_age",
    "in_flight",
]


def trace_rows(trace: EpisodeTrace) -> list[dict[str, Any]]:
    actions = list(trace.actions) if len(trace.actions) else [None] * len(trace.events)
    rows: list[dict[str, Any]] = []
    for event, action in zip(trace.events, actions, strict=False):
        rows.append(
            {
                "tick": event.tick,
                "arrived_packets": ";".join(str(v) for v in event.arrived_packet_ids),
                "latest_packet": event.latest_packet_id,
                "request_id": event.request_id if event.request_submitted else "",
                "request_reason": event.request_reason or "",
                "suppressed": event.trigger_suppressed_reason or "",
                "completed_requests": ";".join(str(v) for v in event.completed_request_ids),
                "activated_plan": event.activated_plan_id if event.activated_plan_id is not None else "",
                "action_source": event.action_source,
                "a0": round(float(action[0]), 4) if action is not None and len(action) else "",
                "a1": round(float(action[1]), 4) if action is not None and len(action) > 1 else "",
                "fallback_reason": event.fallback_reason or "",
                "active_plan": event.active_plan_id if event.active_plan_id is not None else "",
                "active_plan_age": event.active_plan_age_ticks if event.active_plan_age_ticks is not None else "",
                "input_obs_age": event.input_obs_age_ticks if event.input_obs_age_ticks is not None else "",
                "in_flight": int(event.in_flight),
            }
        )
    return rows


def _row_from_event_dict(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "tick": event["tick"],
        "arrived_packets": ";".join(str(v) for v in event.get("arrived_packet_ids", [])),
        "latest_packet": event.get("latest_packet_id"),
        "request_id": event.get("request_id") if event.get("request_submitted") else "",
        "request_reason": event.get("request_reason") or "",
        "suppressed": event.get("trigger_suppressed_reason") or "",
        "completed_requests": ";".join(str(v) for v in event.get("completed_request_ids", [])),
        "activated_plan": event.get("activated_plan_id") if event.get("activated_plan_id") is not None else "",
        "action_source": event.get("action_source"),
        "a0": round(float(event["action_summary"][0]), 4) if event.get("action_summary") else "",
        "a1": round(float(event["action_summary"][1]), 4) if event.get("action_summary") and len(event["action_summary"]) > 1 else "",
        "fallback_reason": event.get("fallback_reason") or "",
        "active_plan": event.get("active_plan_id") if event.get("active_plan_id") is not None else "",
        "active_plan_age": event.get("active_plan_age_ticks") if event.get("active_plan_age_ticks") is not None else "",
        "input_obs_age": event.get("input_obs_age_ticks") if event.get("input_obs_age_ticks") is not None else "",
        "in_flight": int(event.get("in_flight", False)),
    }


def rows_from_events_file(path: str | Path) -> list[dict[str, Any]]:
    """Rebuild timeline rows from a saved events.jsonl (arrays may be absent)."""

    import json

    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if "request" in payload:
            continue
        rows.append(_row_from_event_dict(payload))
    return rows


def write_rows_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMELINE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return target


def plot_rows(rows: list[dict[str, Any]], path: str | Path, *, title: str = "discrete-event timeline") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    request_ticks = [row["tick"] for row in rows if row.get("request_id") != ""]
    ax.eventplot(
        [request_ticks], lineoffsets=4.0, linelengths=0.7, colors=["tab:red"], label="inference request"
    )
    arrivals = [row["tick"] for row in rows if row.get("arrived_packets")]
    ax.eventplot([arrivals], lineoffsets=3.0, linelengths=0.7, colors=["tab:blue"], label="packet arrival")
    completions = [row["tick"] for row in rows if row.get("completed_requests")]
    ax.eventplot([completions], lineoffsets=2.0, linelengths=0.7, colors=["tab:green"], label="result release")
    activations = [row["tick"] for row in rows if row.get("activated_plan") != ""]
    ax.eventplot([activations], lineoffsets=1.0, linelengths=0.7, colors=["tab:purple"], label="plan activation")
    plan_source = [row["tick"] for row in rows if row.get("action_source") == "plan"]
    ax.eventplot([plan_source], lineoffsets=0.0, linelengths=0.7, colors=["tab:gray"], label="action from plan")
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["plan action", "activation", "release", "arrival", "request"])
    ax.set_xlabel("control tick (50 ms each)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return Path(path)


def write_timeline_csv(trace: EpisodeTrace, path: str | Path) -> Path:
    return write_rows_csv(trace_rows(trace), path)


def plot_timeline(trace: EpisodeTrace, path: str | Path, *, title: str = "discrete-event timeline") -> Path:
    return plot_rows(trace_rows(trace), path, title=title)
