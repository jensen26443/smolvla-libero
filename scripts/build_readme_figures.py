"""Build the README figures from committed result files.

Reads results/formal_results.csv, results/task_results.csv and
results/e3_shared_formal/summary.json; writes assets/fig_*.png.
Requires only matplotlib — no simulation stack.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

C_BASE = "#9AA5B1"  # gray: reference / baseline
C_MAIN = "#2563EB"  # blue: this work
C_WARN = "#E11D48"  # red: negative control
GRID = dict(axis="y", linestyle=(0, (4, 4)), linewidth=0.8, alpha=0.45)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p, c - h, c + h


def load_formal() -> dict[str, dict]:
    rows = {}
    with (ROOT / "results/formal_results.csv").open() as f:
        for r in csv.DictReader(f):
            rows[r["condition"]] = dict(
                k=int(r["successes"]),
                n=int(r["trials"]),
                p=float(r["success_rate"]),
                lo=float(r["wilson_low"]),
                hi=float(r["wilson_high"]),
            )
    return rows


def load_task() -> dict[tuple[str, str], dict]:
    rows = {}
    with (ROOT / "results/task_results.csv").open() as f:
        for r in csv.DictReader(f):
            rows[(r["condition"], r["task_key"])] = dict(
                k=int(r["successes"]),
                n=int(r["trials"]),
                p=float(r["success_rate"]),
                lo=float(r["wilson_low"]),
                hi=float(r["wilson_high"]),
            )
    return rows


def style(ax: plt.Axes, ylim: float = 105.0) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(**GRID)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ylim)
    ax.set_ylabel("success rate (%)")
    ax.tick_params(length=0)


def grouped_bars(ax, labels, series, width=0.36):
    """series: list of (name, color, [(p, lo, hi), ...])."""
    import numpy as np

    x = np.arange(len(labels))
    for i, (name, color, vals) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * width
        ps = [v[0] * 100 for v in vals]
        err = [
            [max(0.0, (p - lo)) * 100 for p, lo, hi in vals],
            [max(0.0, (hi - p)) * 100 for p, lo, hi in vals],
        ]
        bars = ax.bar(
            x + off, ps, width * 0.92, yerr=err, color=color, label=name,
            error_kw=dict(capsize=3, elinewidth=1.0, ecolor="#4B5563"), zorder=3,
        )
        for rect, (p, lo, hi) in zip(bars, vals):
            ax.annotate(
                f"{p * 100:.1f}", (rect.get_x() + rect.get_width() / 2, hi * 100 + 2.5),
                ha="center", va="bottom", fontsize=9, color="#111827",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)


def fig_task_success() -> None:
    task = load_task()
    formal = load_formal()
    tasks = ["goal_bowl_cabinet_top", "goal_bowl_plate", "goal_bowl_stove"]
    names = ["bowl →\ncabinet top", "bowl →\nplate", "bowl →\nstove", "overall\n(3 tasks)"]
    base = [task[("E0", t)] for t in tasks] + [formal["E0"]]
    adap = [task[("E1", t)] for t in tasks] + [formal["E1"]]
    vals = lambda rows: [(r["p"], r["lo"], r["hi"]) for r in rows]  # noqa: E731

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    grouped_bars(
        ax, names,
        [("public baseline", C_BASE, vals(base)), ("adapted (this work)", C_MAIN, vals(adap))],
    )
    style(ax, ylim=110)
    ax.set_title("LIBERO Goal bowl-placement tasks, 30 rollouts per task (Wilson 95% CI)",
                 fontsize=11.5, pad=30)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncols=2,
              fontsize=10, columnspacing=1.8, handlelength=1.4)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_task_success.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig_data_efficiency() -> None:
    formal = load_formal()
    budgets = [("E2_020pct", "21 demos\n(20%)"), ("E2_050pct", "56 demos\n(50%)"), ("E1", "113 demos\n(100%)")]
    vals = [(formal[c]["p"], formal[c]["lo"], formal[c]["hi"]) for c, _ in budgets]
    base = formal["E0"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    import numpy as np

    x = np.arange(len(vals))
    err = [
        [max(0.0, (p - lo)) * 100 for p, lo, hi in vals],
        [max(0.0, (hi - p)) * 100 for p, lo, hi in vals],
    ]
    bars = ax.bar(
        x, [v[0] * 100 for v in vals], 0.5, yerr=err, color=C_MAIN,
        error_kw=dict(capsize=3, elinewidth=1.0, ecolor="#4B5563"), zorder=3,
    )
    for rect, (p, lo, hi) in zip(bars, vals):
        ax.annotate(f"{p * 100:.1f}%", (rect.get_x() + rect.get_width() / 2, hi * 100 + 2.5),
                    ha="center", va="bottom", fontsize=10, color="#111827")
    ax.axhline(base["p"] * 100, color=C_BASE, linestyle=(0, (5, 4)), linewidth=1.4,
               zorder=2, label=f"public baseline {base['p'] * 100:.1f}%")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncols=1, fontsize=10, handlelength=2.2)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in budgets])
    ax.set_xlabel("demonstrations in the training pool (10,000 steps each)", fontsize=10)
    style(ax)
    ax.set_title("Data efficiency: success rate vs. demonstration budget (Wilson 95% CI)",
                 fontsize=11.5, pad=30)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_data_efficiency.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig_language_ablation() -> None:
    formal = load_formal()
    e3 = json.loads((ROOT / "results/e3_shared_formal/summary.json").read_text())["by_condition"]["E3_for"]
    conds = [
        ("public baseline", formal["E0"]["k"], formal["E0"]["n"], C_BASE),
        ("paired\ninitialization", e3["successes"], e3["trials"], C_MAIN),
        ("paraphrased\ninstructions", formal["E5"]["k"], formal["E5"]["n"], C_MAIN),
        ("empty\ninstruction", 0, 20, C_WARN),
    ]
    stats = [wilson(k, n) for _, k, n, _ in conds]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    import numpy as np

    x = np.arange(len(conds))
    err = [
        [max(0.0, (p - lo)) * 100 for p, lo, hi in stats],
        [max(0.0, (hi - p)) * 100 for p, lo, hi in stats],
    ]
    bars = ax.bar(
        x, [s[0] * 100 for s in stats], 0.5, yerr=err,
        color=[c for *_, c in conds],
        error_kw=dict(capsize=3, elinewidth=1.0, ecolor="#4B5563"), zorder=3,
    )
    for rect, (p, lo, hi), (_, k, n, _) in zip(bars, stats, conds):
        label = f"{p * 100:.1f}%" if k else f"{k}/{n}"
        ax.annotate(label, (rect.get_x() + rect.get_width() / 2, hi * 100 + 2.5),
                    ha="center", va="bottom", fontsize=10, color="#111827")
    ax.annotate("all 20 rollouts time out", (2.6, 4.5), ha="right", va="bottom",
                fontsize=9, color="#9F1239", style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{name}\n(n={n})" for name, _, n, _ in conds], fontsize=9.5)
    style(ax)
    ax.set_title("Language conditioning: same scenes, different instructions (Wilson 95% CI)",
                 fontsize=11.5, pad=10)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_language_ablation.png", dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    fig_task_success()
    fig_data_efficiency()
    fig_language_ablation()
    for f in sorted(ASSETS.glob("fig_*.png")):
        print(f.relative_to(ROOT), f.stat().st_size)


if __name__ == "__main__":
    main()
