#!/usr/bin/env bash
# One-shot finisher after the formal matrix completes: aggregates, figures,
# and a compact headline table for the report.
set -eu
cd "$(dirname "$0")/.."
EXP="${1:-runs/latency_formal_v1}"

.venv/bin/sim-vla-latency summarize --experiment-dir "$EXP"

.venv/bin/python - "$EXP" <<'EOF'
import json
import sys
from pathlib import Path

exp = Path(sys.argv[1])
summary = json.loads((exp / "summary.json").read_text(encoding="utf-8"))
rows = []
for key, cell in sorted(summary["by_task_method_latency"].items()):
    task, method, latency = key.split("/")
    rate = cell["success_rate"]
    wilson = cell["wilson_95"]
    req = cell.get("requests_per_sim_second")
    rows.append(
        f"{task:<7} {method:<10} {latency:<9} "
        f"{cell['successes']:>2}/{cell['trials']:<2} "
        f"{(rate or 0) * 100:5.1f}%  "
        f"[{(wilson[0]) * 100:4.0f},{(wilson[1]) * 100:4.0f}]  "
        f"{(req or 0):5.2f} req/s"
    )
out = "\n".join(rows)
(exp / "headline_table.txt").write_text(out + "\n", encoding="utf-8")
print(out)
print("\npaired comparisons:")
for name, pair in sorted(summary["paired_comparisons"].items()):
    print(
        f"  {name}: diff={pair['mean_difference']:+.3f} "
        f"ci95=[{pair['difference_ci95_bootstrap'][0]:+.3f},{pair['difference_ci95_bootstrap'][1]:+.3f}] "
        f"n={pair['n_pairs']}"
    )
EOF
