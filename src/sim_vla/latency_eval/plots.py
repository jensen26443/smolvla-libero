"""Report figures regenerated from the experiment summary (no live state)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _cell(summary: dict[str, Any], task: str, method: str, latency: str) -> dict[str, Any] | None:
    return summary["by_task_method_latency"].get(f"{task}/{method}/{latency}")


def _latency_seconds(label: str) -> float:
    """Parse 'Lxx' labels back to inference delay seconds via the results CSV.

    Falls back to tick labels embedded in the condition name (``L20`` means
    20 ticks = 1.0 s at 20 Hz).
    """

    digits = "".join(ch for ch in label if ch.isdigit())
    return int(digits) * 0.05 if digits else 0.0


def plot_success_vs_latency(summary: dict[str, Any], output: str | Path) -> Path:
    tasks = sorted(summary["by_task"])
    methods = sorted(summary["by_method"])
    fig, axes = plt.subplots(1, max(len(tasks), 1), figsize=(6 * max(len(tasks), 1), 4.5), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for method in methods:
            points = []
            for latency in sorted(
                {
                    key.split("/")[2]
                    for key in summary["by_task_method_latency"]
                    if key.startswith(f"{task}/")
                }
            ):
                cell = _cell(summary, task, method, latency)
                if cell and cell["trials"]:
                    points.append(
                        (
                            _latency_seconds(latency),
                            cell["success_rate"],
                            np.diff(np.array(cell["wilson_95"]))[0] / 2,
                        )
                    )
            if points:
                points = sorted(points)
                xs, ys, es = zip(*points)
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=method)
        ax.set_title(task)
        ax.set_xlabel("simulated inference-result delay (s)")
        ax.set_ylabel("success rate (Wilson 95%)")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return Path(output)


def plot_success_vs_compute(summary: dict[str, Any], output: str | Path) -> Path:
    """Compute–success tradeoff: request rate on x, success on y, per latency."""

    tasks = sorted(summary["by_task"])
    fig, axes = plt.subplots(1, max(len(tasks), 1), figsize=(6 * max(len(tasks), 1), 4.5), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for method in sorted(summary["by_method"]):
            points = []
            for key, cell in summary["by_task_method_latency"].items():
                if not key.startswith(f"{task}/"):
                    continue
                rate = cell.get("requests_per_sim_second")
                if cell["trials"] and rate is not None:
                    points.append((rate, cell["success_rate"], key.split("/")[1], key.split("/")[2]))
            for rate, success, _, _ in points:
                ax.scatter(rate, success, s=28)
        # annotate method clusters roughly
        ax.set_title(f"{task}: compute–success tradeoff")
        ax.set_xlabel("inference requests per simulated second")
        ax.set_ylabel("success rate")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return Path(output)


def plot_age_distributions(summary: dict[str, Any], output: str | Path) -> Path:
    distributions = summary.get("distributions", {})
    fields = [
        ("input_obs_age_ticks", "input observation age (ticks, per-episode mean)"),
        ("active_plan_age_ticks", "active plan age (ticks, per-episode mean)"),
        ("fallback_fraction", "fallback-action fraction per episode"),
    ]
    rows = [(label, distributions.get(field)) for field, label in fields if distributions.get(field)]
    if not rows:
        return Path(output)
    fig, axes = plt.subplots(1, len(rows), figsize=(5 * len(rows), 4), squeeze=False)
    for ax, (label, stats) in zip(axes[0], rows):
        stats = stats or {}
        xs = [stats.get("min", 0), stats.get("p50", 0), stats.get("p95", 0), stats.get("max", 0)]
        ax.bar(["min", "p50", "p95", "max"], xs, color="tab:blue")
        ax.set_title(label, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return Path(output)


def generate_all_figures(experiment_dir: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    import json

    experiment_dir = Path(experiment_dir)
    summary_path = experiment_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    out = Path(output_dir) if output_dir else experiment_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_success_vs_latency(summary, out / "success_vs_latency.png"),
        plot_success_vs_compute(summary, out / "success_vs_compute.png"),
        plot_age_distributions(summary, out / "age_distributions.png"),
    ]
    return paths
