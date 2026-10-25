"""Plot results/implied_window.jsonl: original vs implied-window query time per case (experiment 4).

    uv run python experiments/plot_implied_window.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ORIG, IMPL = "#eb6834", "#2a78d6"  # categorical slots 2 and 1 (same palette as plot.py)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SHORT = {"CollegeMsg": "CollegeMsg", "email-Eu-core-temporal": "email-Eu-core"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 9, "axes.titlecolor": INK,
    "legend.frameon": False, "figure.dpi": 150,
})


def main():
    rows = {(r["g"], r["k"], r["w"], r["implied"]): r for r in map(json.loads, open(RESULTS / "implied_window.jsonl"))}
    cases = [c for c in dict.fromkeys(k[:3] for k in rows) if (*c, True) in rows and (*c, False) in rows]
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(cases) + 1.1))
    labels = []
    for y, (g, k, w) in enumerate(reversed(cases)):
        o, i = rows[(g, k, w, False)], rows[(g, k, w, True)]
        done = o["count"] is not None
        ax.plot([i["secs"], o["secs"]], [y, y], color=GRID, lw=2, zorder=1, solid_capstyle="round")
        ax.scatter(i["secs"], y, s=36, color=IMPL, zorder=3, edgecolors="white", linewidths=1)
        if done:
            ax.scatter(o["secs"], y, s=36, color=ORIG, zorder=3, edgecolors="white", linewidths=1)
            note = f"{o['secs'] / i['secs']:.1f}×"
        else:  # timed out: hollow marker + arrow, the bar is a lower bound
            ax.scatter(o["secs"], y, s=36, facecolors="white", edgecolors=ORIG, linewidths=1.5, zorder=3)
            ax.annotate("", xy=(o["secs"] * 1.6, y), xytext=(o["secs"] * 1.08, y), zorder=2,
                        arrowprops=dict(arrowstyle="-|>", color=ORIG, lw=1.2))
            x = o["secs"] / i["secs"]
            note = f"> {x:.1f}×, original did not finish" if x < 10 else f"> {x:.0f}×, original did not finish"
        ax.text(max(o["secs"], i["secs"]) * (2.2 if not done else 1.25), y, note, va="center", color=INK2,
                fontsize=8)
        labels.append(f"{SHORT[g]}  {'tri' if k == 3 else 'c4'}  {w}")
    ax.set_yticks(range(len(cases)), labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 3e4)
    ax.set_xlabel("query time (s, log scale; 48 cores, in memory)")
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], s=36, color=ORIG, label="original query")
    ax.scatter([], [], s=36, color=IMPL, label="with implied predicate  f_i.ts <= f1.ts + D")
    ax.scatter([], [], s=36, facecolors="white", edgecolors=ORIG, linewidths=1.5, label="original stopped (timeout)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, handletextpad=0.3,
              columnspacing=1.2)
    ax.set_title("Temporal cycles: same counts, the implied window predicate removes the hub blow-up", pad=24)
    fig.tight_layout()
    fig.savefig(RESULTS / "implied_window.png", facecolor="white")
    print("wrote", RESULTS / "implied_window.png")


if __name__ == "__main__":
    main()
