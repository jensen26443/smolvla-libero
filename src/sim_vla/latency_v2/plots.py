"""Standalone research plots using real results and fixed, paired example trials."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..assets import project_root
from .layout import load_layout
from .statistics import wilson
from .storage import trial_key


def timeline(directory: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines()]
    steps = [row for row in rows if row["event"] == "step"]
    actions = np.load(directory/"arrays.npz", allow_pickle=False)["actions"]
    requests = json.loads((directory/"requests.json").read_text())
    request_observations = {row["request_id"]: row["t_obs"] for row in requests}
    active_obs, actual_ages = None, []
    for row in rows:
        if row["event"] == "activate":
            active_obs = request_observations[row["request_id"]]
        elif row["event"] == "late_discard":
            active_obs = None
        elif row["event"] == "step":
            actual_ages.append(row["tick"]-active_obs if active_obs is not None and not row["neutral"] else np.nan)
    figure, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    ticks = np.arange(len(actions))
    axes[0].plot(ticks, actual_ages, label="observation age of executed model actions")
    axes[0].plot(ticks, [row["age"] if row["age"] is not None else np.nan for row in steps],
                 ls="--", alpha=.5, label="plan / pending-prefix snapshot age")
    axes[1].plot(ticks, [row["index"] for row in steps], label="executed chunk/prefix index")
    axes[2].plot(ticks, np.linalg.norm(actions[:, :3], axis=1), label="translation increment norm")
    axes[3].step(ticks, np.sign(actions[:, 6]), where="post", label="gripper sign")
    for row in rows:
        if row["event"] == "submit":
            axes[0].axvline(row["tick"], color="black", alpha=.15)
            if row["prefix_length"]:
                axes[1].axvspan(row["tick"], row["prefix_end"], color="orange", alpha=.12)
        elif row["event"] == "activate":
            for axis in axes:
                axis.axvline(row["tick"], color="green", alpha=.3)
        elif row["event"] == "late_discard":
            axes[1].axvline(row["tick"], color="red", alpha=.8)
    for axis in axes:
        axis.legend(loc="upper left")
        axis.grid(alpha=.2)
    axes[-1].set_xlabel("Control tick (one physics control step)")
    outcome = json.loads((directory/"result.json").read_text())
    figure.suptitle(f"{outcome['task']} | {outcome['config']['method']} | init {outcome['init_index']} | "
                   f"success={outcome['success']} | {outcome['backend']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def build_figures(summary: dict, output: Path, *, spec_path: Path | None = None) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = project_root()
    frozen_path, frozen, run_dirs = load_layout(root, spec_path)
    output.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(figure, name):
        figure.tight_layout()
        path = output/(name+".png")
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(str(path.relative_to(root)))

    rows = summary["core_aggregate"]
    fig, axis = plt.subplots(figsize=(8, 5))
    for method in ("B_RAW", "B_SHIFT", "C_PREFIX", "C_BLEND"):
        values = [row for row in rows if row["method"] == method]
        intervals = []
        for value in values:
            subset = [cell for cell in summary["cells"]["core"] if cell["method"] == method
                      and cell["latency"] == value["latency"]]
            interval = wilson(sum(cell["successes"] for cell in subset), sum(cell["episodes"] for cell in subset))
            rate = value["success_rate_equal_task"]
            intervals.append([rate-interval[0], interval[1]-rate])
        axis.errorbar([row["latency"] for row in values], [row["success_rate_equal_task"] for row in values],
                      yerr=np.maximum(np.asarray(intervals).T, 0), fmt="o-", capsize=3, label=method)
    axis.set(xlabel="Injected latency (control ticks)", ylabel="Equal-task success rate", ylim=(0, 1.05))
    axis.legend()
    axis.grid(alpha=.2)
    save(fig, "core_success_vs_latency")

    baselines = json.loads((root/"runs/latency_v2_dev_baselines/baseline_selection.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for axis, latency in zip(axes, (5, 10, 20)):
        for method in ("B_RAW", "B_SHIFT"):
            values = [row for row in baselines["scan"] if row["latency"] == latency and row["config"]["method"] == method]
            axis.scatter([row["mean_generation_seconds"] for row in values],
                         [row["success_rate_equal_task"] for row in values], label=method)
            for row in values:
                axis.annotate(f"K{row['config']['period']}", (row["mean_generation_seconds"], row["success_rate_equal_task"]), fontsize=7)
        axis.set(title=f"Development L={latency}", xlabel="Actual generation seconds / episode", ylim=(-.03, 1.05))
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Equal-task success rate")
    save(fig, "complete_baseline_cost_frontier")

    candidates = json.loads((root/"runs/latency_v2_dev_candidates/candidate_selection.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for axis, latency in zip(axes, (5, 10, 20)):
        values = [row for row in candidates["scan"] if row["latency"] == latency]
        labels = [f"{row['config']['method']}\nm={row['config']['margin']}, b={row['config']['blend']}" for row in values]
        axis.bar(labels, [row["success_rate_equal_task"] for row in values])
        axis.set(title=f"Development L={latency}", ylim=(0, 1.05))
        axis.tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("Equal-task success rate")
    save(fig, "prefix_and_blend_ablation")

    metrics = json.loads((root/"docs/latency_v2/core_mechanism.json").read_text())["aggregate"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for method in ("B_RAW", "B_SHIFT", "C_PREFIX", "C_BLEND"):
        values = [row for row in metrics if row["method"] == method]
        for axis, name in zip(axes, ("mean_observation_age_ticks", "switch_translation_mean_action_norm", "mean_fallback_fraction")):
            axis.plot([row["latency"] for row in values], [row[name] for row in values], "o-", label=method)
            axis.set(xlabel="Injected latency (ticks)", ylabel=name.replace("_", " "))
            axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    save(fig, "age_switch_and_fallback")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method in summary["async_success_rates"]:
        values = [row for row in summary["cells"]["jitter"] if row["method"] == method]
        averages = [np.mean([row["success_rate"] for row in values if row["profile"] == profile]) for profile in ("J1", "J2")]
        axes[0].plot(["J1", "J2"], averages, "o-", label=method)
    axes[0].set(ylabel="Equal-task success rate", ylim=(0, 1.05), title="Exogenous control-tick jitter")
    axes[0].legend()
    values = summary["async_metrics"]
    axes[1].bar([row["method"] for row in values], [row["mean_actual_control_hz"] for row in values])
    axes[1].axhline(20, color="red", ls="--", label="target 20 Hz")
    axes[1].set(ylabel="Measured wall control Hz", title="Real spawned GPU worker")
    axes[1].legend()
    save(fig, "jitter_and_real_async")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    asynchronous = run_dirs["async"]
    cycles, endpoints = {}, {}
    for expected in json.loads((asynchronous/"expected_trials.json").read_text()):
        method = expected["config"]["method"]
        trial_dir = asynchronous/"trials"/expected["key"]
        events = [json.loads(line) for line in (trial_dir/"events.jsonl").read_text().splitlines()]
        cycles.setdefault(method, []).extend(row["tick_wall_s"] for row in events if row["event"] == "step")
        requests = json.loads((trial_dir/"requests.json").read_text())
        endpoints.setdefault(method, []).extend(row["endpoint_s"] for row in requests if not row.get("startup", False))
    for method, cycle_values in cycles.items():
        for axis, values in zip(axes, (cycle_values, endpoints[method])):
            ordered = np.sort(values)
            axis.plot(ordered, np.arange(1, len(ordered)+1)/max(len(ordered), 1), label=method)
    axes[0].axvline(.055, color="red", ls="--", label="55 ms target")
    axes[0].set(xlabel="Measured control cycle (s)", ylabel="Empirical CDF")
    axes[1].set(xlabel="Worker endpoint latency (s), startup excluded", ylabel="Empirical CDF")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    save(fig, "real_async_cycle_and_endpoint_distributions")

    fig, axis = plt.subplots(figsize=(10, 5))
    values = [row for row in metrics if row["latency"] == 20]
    methods = [row["method"] for row in values]
    bottom = np.zeros(len(values))
    for stage in ("approach", "grasp", "transport_contact", "release"):
        heights = np.asarray([row["failure_stage_counts_heuristic"].get(stage, 0) for row in values])
        axis.bar(methods, heights, bottom=bottom, label=stage)
        bottom += heights
    axis.set(ylabel="Failed episodes", title="L20 offline geometric stage heuristics (not causal ground truth)")
    axis.legend(fontsize=8)
    save(fig, "failure_stage_heuristics")

    trials = json.loads((frozen_path.parent/"core_trials.json").read_text())
    minimum_init = min(row["init_index"] for row in trials)
    for method in (frozen["B_SELECTED"], frozen["C_SELECTED"]):
        trial = next(row for row in trials if row["task"] == "goal_bowl_plate" and row["init_index"] == minimum_init
                     and row["latency"] == 20 and row["config"]["method"] == method)
        path = output/("paired_plate_L20_"+method+"_timeline.png")
        timeline(run_dirs["core"]/"trials"/trial_key(trial), path)
        paths.append(str(path.relative_to(root)))
    return paths
