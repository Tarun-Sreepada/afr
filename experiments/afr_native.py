"""AFR with the native reduction operator (afr_kernel.cpp): reduce in C++ with per-chunk selection vectors and
bitset semi-joins, compact, load the survivors into DuckDB, and time DuckDB's join on them against plain DuckDB.

Datasets: the SNAP graphs, a synthetic trap (fan-in chains of depth d into a hub that fans out into dead-end
chains of depth d, plus 100 real triangles), and a real graph with the trap planted in it ("web-Google+trap4").
Writes results/native.csv (one row per dataset/query/mode) and results/native_rounds.csv.

    uv run python experiments/afr_native.py [dataset ...] [--threads N] [--reps R] [--timeout S]
"""

import argparse
import ctypes
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from afr import CYCLE_QUERIES, PATH_QUERY, RESULTS, load, time_join  # noqa: E402

HERE = Path(__file__).resolve().parent
LIB = HERE / "libafr.so"
DATASETS = ["trap1", "trap2", "trap3", "trap4", "trap6", "trap8", "trap12", "trap16", "soc-Epinions1", "soc-Slashdot0902",
            "email-EuAll", "web-Google", "web-Google+trap4"]
QUERIES = {"tri": (3, True), "c4": (4, True), "path2": (2, False)}
RULES = {"fixed": 0, "rho": 1, "cost": 2, "work": 3, "budget": 4}
# (name, scheme, rule, max_rounds, tau); fixed-1/2 per-copy is the RPT-style baseline
MODES = [("k0", "copy", "fixed", 0, 0), ("k1", "copy", "fixed", 1, 0), ("k2", "copy", "fixed", 2, 0),
         ("rho5%", "copy", "rho", 1000, 0.05), ("cost", "copy", "cost", 1000, 0),
         ("work", "copy", "work", 1000, 0), ("budget10%", "copy", "budget", 1000, 0.1),
         ("fixpoint", "copy", "fixed", 1000, 0),
         ("k1", "shared", "fixed", 1, 0), ("k2", "shared", "fixed", 2, 0),
         ("rho5%", "shared", "rho", 1000, 0.05), ("cost", "shared", "cost", 1000, 0),
         ("work", "shared", "work", 1000, 0), ("budget10%", "shared", "budget", 1000, 0.1),
         ("fixpoint", "shared", "fixed", 1000, 0)]
MAX_ROUNDS = 1000


def build_lib():
    src = HERE / "afr_kernel.cpp"
    if not LIB.exists() or LIB.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(["g++", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", str(src), "-o", str(LIB)],
                       check=True)
    lib = ctypes.CDLL(str(LIB))
    P = ctypes.c_void_p
    lib.afr_reduce.argtypes = [P, P, ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                               ctypes.c_int, P, P, P, P, P, P, P, P]
    lib.afr_reduce.restype = ctypes.c_int
    return lib


def trap_edges(d, fan, base):
    """fan chains of d edges into hub, hub into fan chains of d edges ending at dead ends: fan^2 2-hop paths
    through the hub, none on a cycle. Node ids from base up."""
    hub, width = base, fan
    layer = lambda side, l: base + 1 + (2 * side * d + l) * width + np.arange(fan)  # noqa: E731
    a, b = [], []
    for l in range(d - 1):  # in-chains: layer l -> l+1, then layer d-1 -> hub
        a.append(layer(0, l)), b.append(layer(0, l + 1))
    a.append(layer(0, d - 1)), b.append(np.full(fan, hub))
    a.append(np.full(fan, hub)), b.append(layer(1, 0))  # hub -> out-chains, then d-1 more edges
    for l in range(d - 1):
        a.append(layer(1, l)), b.append(layer(1, l + 1))
    return np.concatenate(a), np.concatenate(b)


def load_dataset(con, name, fan):
    if name.startswith("trap"):
        d = int(name[4:])
        tri = np.arange(100) * 3 + 1
        ta, tb = trap_edges(d, fan, 1000)
        a = np.concatenate([tri, tri + 1, tri + 2, ta]).astype("int32")
        b = np.concatenate([tri + 1, tri + 2, tri, tb]).astype("int32")
        con.register("raw", pd.DataFrame({"a": a, "b": b}))
        con.execute("CREATE OR REPLACE TABLE e0 AS SELECT DISTINCT a, b FROM raw WHERE a <> b")
        con.unregister("raw")
    elif "+trap" in name:
        graph, d = name.split("+trap")
        load(con, graph)
        top = con.execute("SELECT greatest(max(a), max(b)) FROM e0").fetchall()[0][0]
        ta, tb = trap_edges(int(d), fan, top + 1)
        con.register("raw", pd.DataFrame({"a": ta.astype("int32"), "b": tb.astype("int32")}))
        con.execute("INSERT INTO e0 SELECT * FROM raw")
        con.unregister("raw")
    else:
        load(con, name)
    e = con.execute("SELECT a, b FROM e0").fetchnumpy()
    return np.ascontiguousarray(e["a"], dtype="int32"), np.ascontiguousarray(e["b"], dtype="int32")


def reduce(lib, a, b, k, cyclic, shared, rule, max_rounds, tau, c_j, c_w, threads):
    n, domain = len(a), int(max(a.max(), b.max())) + 1
    nc = 1 if shared else k
    outs = [(np.empty(n, "int32"), np.empty(n, "int32")) for _ in range(nc)]
    pa = (ctypes.c_void_p * nc)(*[o[0].ctypes.data for o in outs])
    pb = (ctypes.c_void_p * nc)(*[o[1].ctypes.data for o in outs])
    rows = np.zeros(nc, "int64")
    st_rows, st_secs, st_w = np.zeros(MAX_ROUNDS, "int64"), np.zeros(MAX_ROUNDS), np.zeros(MAX_ROUNDS)
    w0, csecs = ctypes.c_double(), ctypes.c_double()
    rounds = lib.afr_reduce(a.ctypes.data, b.ctypes.data, n, domain, k, int(cyclic), int(shared), RULES[rule],
                            max_rounds, tau, c_j, c_w, threads, pa, pb, rows.ctypes.data, st_rows.ctypes.data,
                            st_secs.ctypes.data, st_w.ctypes.data, ctypes.byref(w0), ctypes.byref(csecs))
    copies = [(o[0][:m], o[1][:m]) for o, m in zip(outs, rows)]
    per_round = [{"round": t + 1, "rows_after": int(st_rows[t]), "secs": st_secs[t], "w_after": st_w[t]}
                 for t in range(rounds)]
    return copies, per_round, w0.value, csecs.value


def to_duckdb(con, copies, k):
    """Load the compacted copies as tables f1..fk (shared: the same table under every name)."""
    t = time.perf_counter()
    for i, (a, b) in enumerate(copies):
        con.register("tmp", pd.DataFrame({"a": a, "b": b}))
        con.execute(f"CREATE OR REPLACE TABLE f{i + 1} AS SELECT * FROM tmp")
        con.unregister("tmp")
    names = [f"f{i + 1}" if len(copies) > 1 else "f1" for i in range(k)]
    return time.perf_counter() - t, names


def join_sql(q, names):
    if q == "path2":
        return PATH_QUERY.format(l=names[0], r=names[1])
    tmpl = CYCLE_QUERIES[q]
    for i, nm in enumerate(names):  # the templates use one table {e}; rename each copy's FROM entry
        tmpl = tmpl.replace("{e} f%d" % (i + 1), f"{nm} f{i + 1}", 1)
    return tmpl


def calibrate(con, lib, threads, reps):
    """Seconds per join input row (c_j) and per first-join intermediate row (c_w) for each query, measured on a
    synthetic random graph unrelated to the test data (so the rules don't peek at the plain join they beat)."""
    rng = np.random.default_rng(0)
    n, nodes = 2_000_000, 400_000
    con.register("raw", pd.DataFrame({"a": rng.integers(0, nodes, n, dtype="int32"),
                                      "b": rng.integers(0, nodes, n, dtype="int32")}))
    con.execute("CREATE OR REPLACE TABLE e0 AS SELECT DISTINCT a, b FROM raw WHERE a <> b")
    con.unregister("raw")
    e = con.execute("SELECT a, b FROM e0").fetchnumpy()
    a, b = np.ascontiguousarray(e["a"], "int32"), np.ascontiguousarray(e["b"], "int32")
    consts = {}
    for q, (k, cyclic) in QUERIES.items():
        _, _, w0, _ = reduce(lib, a, b, k, cyclic, False, "fixed", 0, 0, 0, 0, threads)
        dt, _ = time_join(con, join_sql(q, ["e0"] * k), reps, 600)
        consts[q] = (dt / (k * len(a)), dt / w0)
        print(f"calibrate {q}: {dt:.3f}s  c_j={consts[q][0] * 1e9:.1f} ns/row  c_w={consts[q][1] * 1e9:.2f} ns/2hop",
              flush=True)
    return consts


def run(con, lib, name, args, consts):
    a, b = load_dataset(con, name, args.fan)
    print(f"{name}: {len(a)} edges", flush=True)
    rows, round_rows = [], []
    for q, (k, cyclic) in QUERIES.items():
        plain_s, plain_cnt = time_join(con, join_sql(q, ["e0"] * k), args.reps, args.timeout)
        rows.append({"dataset": name, "query": q, "mode": "plain", "scheme": "", "rounds": 0, "rows": k * len(a),
                     "w": np.nan, "reduce_s": 0, "compact_s": 0, "load_s": 0, "join_s": plain_s, "total_s": plain_s,
                     "count": plain_cnt, "plain_s": plain_s})
        print(f"  {q} plain: {plain_s:.3f}s count={plain_cnt}", flush=True)
        c_j, c_w = consts[q]
        for mode, scheme, rule, mr, tau in MODES:
            shared = scheme == "shared"
            if shared and not cyclic:
                continue
            copies, pr, w0, csecs = reduce(lib, a, b, k, cyclic, shared, rule, mr, tau, c_j, c_w, args.threads)
            load_s, names = to_duckdb(con, copies, k)
            join_s, cnt = time_join(con, join_sql(q, names), args.reps, args.timeout)
            red = sum(r["secs"] for r in pr)
            w = pr[-1]["w_after"] if pr else w0
            n_rows = sum(len(c[0]) for c in copies) * (k if shared else 1)
            rows.append({"dataset": name, "query": q, "mode": mode, "scheme": scheme, "rounds": len(pr),
                         "rows": n_rows, "w": w, "reduce_s": red, "compact_s": csecs, "load_s": load_s,
                         "join_s": join_s, "total_s": red + csecs + join_s, "count": cnt, "plain_s": plain_s})
            for r in pr:
                r.update(dataset=name, query=q, mode=mode, scheme=scheme, w0=w0)
            round_rows += pr
            ok = "" if cnt == plain_cnt or plain_cnt is None else f"  COUNT MISMATCH (plain {plain_cnt})"
            print(f"  {q} {scheme:6s} {mode:8s} rounds={len(pr):3d} rows={n_rows:9d} 2hop={w:.3g} "
                  f"reduce={red:.3f}s join={join_s:.3f}s total={red + csecs + join_s:.3f}s "
                  f"({plain_s / (red + csecs + join_s):.2f}x){ok}", flush=True)
    return rows, round_rows


def save(df, f, name):
    path = RESULTS / f
    if path.exists():  # replace this dataset's previous rows
        old = pd.read_csv(path)
        df = pd.concat([old[old.dataset != name], df])
    df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=DATASETS)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--memory", default="64GB")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--fan", type=int, default=10000, help="trap fan-in = fan-out (fan^2 dead 2-hop paths)")
    args = ap.parse_args()
    lib = build_lib()
    con = duckdb.connect()
    con.execute(f"SET threads = {args.threads}")
    con.execute(f"SET memory_limit = '{args.memory}'")
    con.execute("SET preserve_insertion_order = false")
    consts = calibrate(con, lib, args.threads, args.reps)
    RESULTS.mkdir(exist_ok=True)
    for name in args.datasets:
        rows, rr = run(con, lib, name, args, consts)
        save(pd.DataFrame(rows), "native.csv", name)
        save(pd.DataFrame(rr), "native_rounds.csv", name)


if __name__ == "__main__":
    main()
