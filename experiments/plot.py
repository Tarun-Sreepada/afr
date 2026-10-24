"""Plot results/rounds.csv + results/joins.csv and derive the adaptive rule (stop after the first round that
removes less than a fraction d of the current rows).

    uv run python experiments/plot.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order, one slot per graph
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
QUERIES = {"tri": "directed triangles", "c4": "directed 4-cycles", "path2": "2-hop paths (acyclic)"}
DS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 1e-4, 0]

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.titlesize": 9, "axes.titlecolor": INK, "lines.linewidth": 2,
    "legend.frameon": False, "figure.dpi": 150,
})


def family(q):
    return "path" if q == "path2" else "cycle"


def totals(rounds, joins, graph, q):
    """Per k: cumulative reduction time, join time, total. Index k = 0..K-1 (K-th round removed nothing)."""
    rr = rounds[(rounds.graph == graph) & (rounds.family == family(q))].sort_values("round")
    jj = joins[(joins.graph == graph) & (joins["query"] == q)].set_index("k").sort_index()
    red = np.concatenate([[0.0], rr.time_s.cumsum().to_numpy()])
    df = pd.DataFrame({"reduce_s": red[: len(jj)], "join_s": jj.join_s.to_numpy()}, index=jj.index)
    df["total_s"] = df.reduce_s + df.join_s
    return rr, df


def adaptive(rr, df, d):
    """Rounds executed under threshold d, and the resulting total time."""
    frac = (rr.removed / rr.rows_before).to_numpy()
    below = np.nonzero(frac < d)[0] if d > 0 else np.nonzero(frac == 0)[0]
    r = int(below[0]) + 1 if len(below) else len(rr)
    k = min(r, len(df) - 1)  # the state after round r; the zero-removal round's state equals the one before it
    return r, rr.time_s.iloc[:r].sum() + df.join_s.iloc[k]


def main():
    rounds = pd.read_csv(RESULTS / "rounds.csv")
    joins = pd.read_csv(RESULTS / "joins.csv")
    graphs = list(dict.fromkeys(rounds.graph))
    color = dict(zip(graphs, SERIES))

    # counts must match the unreduced join at every k
    bad = joins.dropna(subset=["count"]).groupby(["graph", "query"])["count"].nunique()
    assert (bad == 1).all(), bad[bad != 1]

    # 1. reduction curve (cyclic queries)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for g in graphs:
        rr = rounds[(rounds.graph == g) & (rounds.family == "cycle")].sort_values("round")
        n0 = rr.rows_before.iloc[0]
        x = np.concatenate([[0], rr["round"]])
        y = 100 * np.concatenate([[n0], rr.rows_after]) / n0
        ax.plot(x, y, marker="o", ms=4, color=color[g], label=g)
        ax.annotate(f"{y[-1]:.0f}%", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=8)
    ax.set_xlabel("reduction round")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_ylabel("edges remaining (% of original)")
    ax.set_title("Cycle reduction: keep (a,b) if b has an out-edge and a an in-edge", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / "reduction.png")

    # 2. total time vs k, small multiples
    qs = [q for q in QUERIES if q in set(joins["query"])]
    fig, axes = plt.subplots(len(qs), len(graphs), figsize=(3 * len(graphs), 2.4 * len(qs)), squeeze=False)
    summary = []
    for i, q in enumerate(qs):
        for j, g in enumerate(graphs):
            ax = axes[i][j]
            rr, df = totals(rounds, joins, g, q)
            base = df.total_s.iloc[0]
            ax.axhline(base, color=INK2, lw=1, ls="--", label="plain DuckDB (k=0)")
            ax.plot(df.index, df.total_s, marker="o", ms=4, color=color[g], label="reduce k rounds + join")
            ax.plot(df.index, df.join_s, marker="o", ms=3, lw=1, color=color[g], alpha=0.45, label="join only")
            ax.set_ylim(bottom=0)
            ax.set_title(f"{g} · {QUERIES[q]}", loc="left")
            if i == len(qs) - 1:
                ax.set_xlabel("rounds k")
            if j == 0:
                ax.set_ylabel("seconds")
            ax.xaxis.get_major_locator().set_params(integer=True)
            row = {"graph": g, "query": q, "edges": rr.rows_before.iloc[0], "fixpoint_rounds": len(rr) - 1,
                   "kept_at_fixpoint": rr.rows_after.iloc[-1] / rr.rows_before.iloc[0],
                   "plain_s": base, "best_k": int(df.total_s.idxmin()), "best_speedup": base / df.total_s.min()}
            for k in (1, 2):
                row[f"speedup_k{k}"] = base / df.total_s.iloc[min(k, len(df) - 1)]
            row["speedup_fixpoint"] = base / (rr.time_s.sum() + df.join_s.iloc[-1])
            for d in (0.1, 0.01, 0.001):
                r, t = adaptive(rr, df, d)
                row[f"rounds_d{d}"], row[f"speedup_d{d}"] = r, base / t
            summary.append(row)
    axes[0][0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "time_vs_rounds.png")

    # 3. adaptive rule: speedup over plain DuckDB vs threshold d
    fig, axes = plt.subplots(1, len(qs), figsize=(3.6 * len(qs), 3.2), squeeze=False, sharey=True)
    xs = np.arange(len(DS))
    for i, q in enumerate(qs):
        ax = axes[0][i]
        for g in graphs:
            rr, df = totals(rounds, joins, g, q)
            ax.plot(xs, [df.total_s.iloc[0] / adaptive(rr, df, d)[1] for d in DS], marker="o", ms=4,
                    color=color[g], label=g)
        ax.axhline(1, color=INK2, lw=1, ls="--")
        ax.set_xticks(xs, [str(d) if d else "0\n(fixpoint)" for d in DS], rotation=45, fontsize=7)
        ax.set_xlabel("threshold d (stop after a round removing < d)")
        ax.set_title(QUERIES[q], loc="left")
    axes[0][0].set_ylabel("speedup vs plain DuckDB")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center", ncol=len(graphs), fontsize=8)
    fig.subplots_adjust(top=0.82)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(RESULTS / "adaptive.png")

    s = pd.DataFrame(summary)
    s.to_csv(RESULTS / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        print(s)


if __name__ == "__main__":
    main()
