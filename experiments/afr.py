"""First AFR experiment: repeated semi-join reduction before cyclic (and acyclic) self-joins in DuckDB.

For each graph: load edges(a, b), run reduction rounds to the fixpoint, and at every round k time the join on
the reduced table(s). Writes results/rounds.csv (per-round reduction) and results/joins.csv (per-k join).

    uv run python experiments/afr.py [graph ...] [--threads N] [--reps R] [--timeout S]
"""

import argparse
import statistics
import threading
import time
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
GRAPHS = ["soc-Epinions1", "soc-Slashdot0902", "email-EuAll", "web-Google"]

# Cyclic queries share one reduced table E_k: an edge (a, b) survives a round if b still has an outgoing edge
# and a an incoming one. The 2-hop path is acyclic: L (the f1 copy) and R (the f2 copy) are reduced separately.
CYCLE_QUERIES = {
    "tri": "SELECT count(*) FROM {e} f1, {e} f2, {e} f3 WHERE f1.b = f2.a AND f2.b = f3.a AND f3.b = f1.a",
    "c4": "SELECT count(*) FROM {e} f1, {e} f2, {e} f3, {e} f4 "
    "WHERE f1.b = f2.a AND f2.b = f3.a AND f3.b = f4.a AND f4.b = f1.a",
}
PATH_QUERY = "SELECT count(*) FROM {l} f1, {r} f2 WHERE f1.b = f2.a"


def load(con, name):
    raw = pd.read_csv(RAW / (name + ".txt"), sep="\t", comment="#", header=None, names=["a", "b"], dtype="int32")
    con.register("raw", raw)
    con.execute("CREATE OR REPLACE TABLE e0 AS SELECT DISTINCT a, b FROM raw WHERE a <> b")
    con.unregister("raw")
    return con.execute("SELECT count(*) FROM e0").fetchall()[0][0]


def timed(con, sql, timeout):
    """Run sql, return (seconds, first value); (nan, None) if it exceeds timeout."""
    timer = threading.Timer(timeout, con.interrupt)
    timer.start()
    t = time.perf_counter()
    try:
        val = con.execute(sql).fetchall()  # fetch to the end so the connection takes new work (DuckDB 2.0)
        return time.perf_counter() - t, val[0][0] if val else None
    except duckdb.InterruptException:
        return float("nan"), None
    finally:
        timer.cancel()


def reduce_rounds(con, step_sqls, count_sql, n0):
    """Apply reduction rounds until one removes nothing. step_sqls(k) builds round k from round k-1."""
    rows, k, n = [], 0, n0
    while True:
        k += 1
        t = time.perf_counter()
        for sql in step_sqls(k):
            con.execute(sql)
        dt = time.perf_counter() - t
        m = con.execute(count_sql(k)).fetchone()[0]
        rows.append({"round": k, "rows_before": n, "rows_after": m, "removed": n - m, "time_s": dt})
        n = m
        if rows[-1]["removed"] == 0:
            return rows


def cycle_steps(k):
    p = f"e{k - 1}"
    return [f"CREATE OR REPLACE TABLE e{k} AS SELECT * FROM {p} "
            f"WHERE b IN (SELECT a FROM {p}) AND a IN (SELECT b FROM {p})"]


def path_steps(k):
    lp, rp = ("e0", "e0") if k == 1 else (f"l{k - 1}", f"r{k - 1}")
    return [f"CREATE OR REPLACE TABLE l{k} AS SELECT * FROM {lp} WHERE b IN (SELECT a FROM {rp})",
            f"CREATE OR REPLACE TABLE r{k} AS SELECT * FROM {rp} WHERE a IN (SELECT b FROM l{k})"]


def time_join(con, sql, reps, timeout):
    times, val = [], None
    for _ in range(reps):
        dt, v = timed(con, sql, timeout)
        if dt != dt:  # timed out: don't repeat
            return float("nan"), None
        times.append(dt)
        val = v
    return statistics.median(times), val


def run_graph(name, args):
    con = duckdb.connect()
    con.execute(f"SET threads = {args.threads}")
    con.execute(f"SET memory_limit = '{args.memory}'")
    con.execute("SET preserve_insertion_order = false")
    n0 = load(con, name)
    print(f"{name}: {n0} edges", flush=True)
    round_rows, join_rows = [], []

    for family, steps, count, queries in [
        ("cycle", cycle_steps, lambda k: f"SELECT count(*) FROM e{k}", CYCLE_QUERIES),
        ("path", path_steps, lambda k: f"SELECT (SELECT count(*) FROM l{k}) + (SELECT count(*) FROM r{k})",
         {"path2": PATH_QUERY}),
    ]:
        base = 2 * n0 if family == "path" else n0
        rr = reduce_rounds(con, steps, count, base)
        for r in rr:
            r.update(graph=name, family=family)
            print(f"  {family} round {r['round']}: {r['rows_after']} rows (-{r['removed']}) {r['time_s']:.3f}s",
                  flush=True)
        round_rows += rr
        for q, tmpl in queries.items():
            for k in range(len(rr)):  # the last round removed nothing, so its join would equal round k-1's
                if family == "cycle":
                    sql = tmpl.format(e=f"e{k}")
                else:
                    sql = tmpl.format(l="e0" if k == 0 else f"l{k}", r="e0" if k == 0 else f"r{k}")
                dt, val = time_join(con, sql, args.reps, args.timeout)
                join_rows.append({"graph": name, "query": q, "k": k, "join_s": dt, "count": val})
                print(f"  {q} k={k}: {dt:.3f}s count={val}", flush=True)
    con.close()
    return round_rows, join_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs", nargs="*", default=GRAPHS)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--memory", default="64GB")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    for name in args.graphs:
        rr, jr = run_graph(name, args)
        for rows, f in [(rr, "rounds.csv"), (jr, "joins.csv")]:
            path = RESULTS / f
            df = pd.DataFrame(rows)
            if path.exists():  # replace this graph's previous rows
                old = pd.read_csv(path)
                df = pd.concat([old[old.graph != name], df])
            df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
