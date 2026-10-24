"""Why is AFR after 1 round often faster than AFR at the fixpoint, although the fixpoint keeps fewer rows?

For one temporal case, reduce with 1 round and to the fixpoint (afr_vs_rpt.afr_reduce), then run the same join
in plain DuckDB 1.3 (RPT_DISABLE=1) with JSON profiling and compare the plans: join order, build/probe sides,
rows out of every operator, operator time. Writes results/anomaly/<dataset>-<query>-<mode>.json (profiles) and
prints a side-by-side summary.

    numactl --cpunodebind=1 --membind=1 uv run python experiments/one_round_anomaly.py sx-askubuntu --window 3600
"""

import argparse
import json
import os
import subprocess
import sys
import statistics
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import afr_vs_rpt as m  # noqa: E402

OUT = m.RESULTS / "anomaly"


def walk(node, depth, rows):
    name = node.get("operator_name") or node.get("operator_type") or node.get("name", "")
    if name:
        info = node.get("extra_info", {})
        cond = info.get("Conditions", "") if isinstance(info, dict) else ""
        rows.append({"depth": depth, "op": name.strip(), "rows_out": node.get("operator_cardinality"),
                     "secs": node.get("operator_timing"), "cond": " ".join(cond) if isinstance(cond, list) else cond})
    for c in node.get("children", []):
        walk(c, depth + 1, rows)


def profile(tables, query, threads, reps, tag, pre=""):
    OUT.mkdir(parents=True, exist_ok=True)
    prof = OUT / f"{tag}.json"
    setup = "\n".join(f"CREATE TABLE {n} AS FROM '{p}';" for n, p in tables.items())
    script = (f"SET threads = {threads};\nSET memory_limit = '96GB';\n{pre}{setup}\n.mode csv\n.headers off\n.timer on\n"
              + (query + ";\n") * reps
              + f"SET enable_profiling = 'json';\nSET profiling_output = '{prof}';\n{query};\n")
    p = subprocess.run([str(m.DUCK13)], input=script, capture_output=True, text=True,
                       env=dict(os.environ, RPT_DISABLE="1"))
    times = [float(x) for x in re.findall(r"Run Time \(s\): real ([0-9.]+)", p.stdout)][:reps]
    rows = []
    walk(json.loads(prof.read_text()), 0, rows)
    return statistics.median(times), times, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--window", type=int, default=3600)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--fixed-plan", action="store_true",
                    help="also run with join ordering and build-side selection off, building the lower join's hash "
                         "table on f2 (as written) and on f1 (FROM order swapped); triangles only")
    args = ap.parse_args()
    lib = m.load_lib()
    path = m.prepare(args.dataset)
    e = duckdb.connect().execute(f"SELECT a, b, ts, w FROM '{path}'").fetchnumpy()
    base = {c: np.ascontiguousarray(e[c]) for c in ["a", "b", "ts", "w"]}
    k, d = args.k, args.window
    gaps = [1] * (k - 1) + [-d]
    domain = int(max(base["a"].max(), base["b"].max())) + 1
    qname = f"{'tri' if k == 3 else 'c4'}-{d}s"
    summary = {}
    for mode, mr in [("k1", 1), ("k2", 2), ("fix", 10_000)]:
        red, rounds, _ = m.afr_reduce(lib, [base] * k, k, gaps, True, mr, args.threads, domain)
        tables = {}
        for i, c in enumerate(red):
            p = OUT / f"tmp-{mode}-f{i + 1}.parquet"
            OUT.mkdir(parents=True, exist_ok=True)
            duckdb.from_df(pd.DataFrame(c)).write_parquet(str(p))
            tables[f"f{i + 1}x"] = p
        q = m.sql("temporal", k, list(tables), d)
        med, times, plan = profile(tables, q, args.threads, args.reps, f"{args.dataset}-{qname}-{mode}")
        sizes = [len(c["a"]) for c in red]
        summary[mode] = (med, sizes, plan)
        print(f"\n== {mode}: {len(rounds)} rounds, copy sizes {sizes}, join median {med:.3f}s "
              f"(runs {', '.join(f'{t:.3f}' for t in times)})")
        print(plan.to_string(index=False))
        if args.fixed_plan and k == 3:
            pre = "SET disabled_optimizers = 'join_order,build_side_probe_side';\n"
            swapped = q.replace("FROM f1x f1 JOIN f2x f2 ON", "FROM f2x f2 JOIN f1x f1 ON", 1)
            assert swapped != q
            for side, sq in [("build-f2", q), ("build-f1", swapped)]:
                med, times, plan = profile(tables, sq, args.threads, args.reps,
                                           f"{args.dataset}-{qname}-{mode}-{side}", pre)
                lower = plan[plan.op == "HASH_JOIN"].iloc[-1]
                print(f"   fixed plan {side}: join median {med:.3f}s, 2-hop rows {lower.rows_out}, "
                      f"lower join {lower.secs:.3f}s (runs {', '.join(f'{t:.3f}' for t in times)})")
        for p in tables.values():
            p.unlink()


if __name__ == "__main__":
    main()
