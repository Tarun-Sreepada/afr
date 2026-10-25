"""Experiment 4: the implied window predicate on temporal cycles.

The temporal cycle queries of afr_vs_rpt.py require f1.ts < f2.ts < ... < fk.ts <= f1.ts + D, which implies
f_i.ts <= f1.ts + D for every copy. DuckDB does not derive this, so it only applies the window at the last join.
This script runs each query as written and with the implied predicate added to the intermediate joins, and
records both counts (they must match) and wall times.

DuckDB parallelizes a scan by row groups (122,880 rows), so a graph with fewer than THREADS row groups runs on
1-3 threads. For those, f1 is split into THREADS disjoint parts (rowid % THREADS) run concurrently and the
counts summed. Each cycle has exactly one f1 row, so the sum is exact. Spilling is disabled: a query that does
not fit in memory fails instead of writing to disk.
"""
import argparse, itertools, json, statistics, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb

from afr_vs_rpt import TEMPORAL, WINDOWS, prepare

ROOT = Path(__file__).resolve().parent.parent
ROW_GROUP = 122_880


def sql(k, D, implied, part=None, parts=None):
    q = "SELECT count(*) FROM g f1"
    for i in range(1, k):
        on = f"f{i}.b = f{i + 1}.a AND f{i}.ts < f{i + 1}.ts"
        if i == k - 1:
            on += f" AND f{k}.b = f1.a AND f{k}.ts <= f1.ts + {D}"
        elif implied:
            on += f" AND f{i + 1}.ts <= f1.ts + {D}"
        q += f" JOIN g f{i + 1} ON {on}"
    return q if part is None else q + f" WHERE f1.rowid % {parts} = {part}"


def run(con, k, D, implied, parts, timeout):
    """(count, seconds, error); count is None on timeout or error."""
    curs = [con.cursor() for _ in range(parts)]
    t = threading.Timer(timeout, lambda: [c.interrupt() for c in curs])
    t.start()
    s = time.time()
    try:
        if parts > 1:
            with ThreadPoolExecutor(parts) as ex:
                c = sum(ex.map(lambda i: curs[i].execute(sql(k, D, implied, i, parts)).fetchone()[0], range(parts)))
        else:
            c = curs[0].execute(sql(k, D, implied)).fetchone()[0]
        return c, time.time() - s, None
    except Exception as e:
        return None, time.time() - s, f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        t.cancel()
        for cur in curs:
            cur.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs", nargs="*", default=TEMPORAL)
    ap.add_argument("--queries", default="tri,c4")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--memory", default="340GB")
    ap.add_argument("--out", default=str(ROOT / "results" / "implied_window.jsonl"))
    args = ap.parse_args()
    ks = [{"tri": 3, "c4": 4}[q] for q in args.queries.split(",")]
    try:  # resume: skip cases already recorded
        done = {(r["g"], r["k"], r["w"], r["implied"]) for r in map(json.loads, open(args.out))}
    except FileNotFoundError:
        done = set()
    out = open(args.out, "a")
    for g in args.graphs:
        con = duckdb.connect()
        con.execute(f"SET threads={args.threads}; SET memory_limit='{args.memory}'; SET temp_directory=''")
        con.execute(f"CREATE TABLE g AS SELECT a, b, ts FROM '{prepare(g)}'")
        n = con.execute("SELECT count(*) FROM g").fetchone()[0]
        parts = args.threads if n < args.threads * ROW_GROUP else 1
        for k, wn, implied in itertools.product(ks, WINDOWS, (False, True)):
            if (g, k, wn, implied) in done:
                continue
            c, s, err = run(con, k, WINDOWS[wn], implied, parts, args.timeout)
            if err is None and s < 20:
                s = statistics.median([s] + [run(con, k, WINDOWS[wn], implied, parts, args.timeout)[1]
                                             for _ in range(2)])
            r = dict(g=g, rows=n, split=parts > 1, k=k, w=wn, implied=implied, count=c, secs=round(s, 3), err=err)
            out.write(json.dumps(r) + "\n")
            out.flush()
            print(r, flush=True)
        con.close()


if __name__ == "__main__":
    main()
