"""AFR vs RPT+ (embryo-labs/dynamic-predicate-transfer, DuckDB 1.3) on two kinds of cyclic queries:

  temporal  SNAP temporal graphs; directed triangles / 4-cycles whose edges are strictly increasing in time and
            close within a window D (f_k.ts <= f_1.ts + D). RPT+ only transfers the equality part.
  filtered  static SNAP graphs; equality triangles with a selective filter on an edge attribute w (uniform in
            [0, 1000)), either on f1 only ("one") or on every copy ("all").

Systems: vanilla DuckDB 1.3 (the RPT+ binary with RPT_DISABLE=1), RPT+, DuckDB 2.0 (Python package), and AFR:
reduce in C++ (afr_cycle.cpp: equality bitsets, or per-key min/max timestamps for temporal edges) for 1, 2, 4
rounds or to the fixpoint, then run the same SQL in vanilla DuckDB 1.3 on the reduced copies. AFR time =
filter + reduce + compact + join; writing the reduced copies to Parquet and loading them is not counted.
Writes results/afr_vs_rpt<tag>.csv and results/afr_vs_rpt_rounds<tag>.csv. For parallel jobs, pin each to one
socket (children inherit the binding) and give each its own --tag:

    numactl --cpunodebind=0 --membind=0 uv run python experiments/afr_vs_rpt.py [dataset ...] --tag _s0
"""

import argparse
import ctypes
import re
import subprocess
import os
import statistics
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DUCK13 = ROOT / "systems" / "rptplus" / "build" / "release" / "duckdb"
PARQ = ROOT / "data" / "parquet"
RESULTS = ROOT / "results"
LIB = HERE / "libafr_cycle.so"

TEMPORAL = ["CollegeMsg", "email-Eu-core-temporal", "sx-mathoverflow", "sx-askubuntu", "sx-superuser",
            "wiki-talk-temporal", "sx-stackoverflow"]
STATIC = ["email-EuAll", "soc-Epinions1", "soc-Slashdot0902", "web-Google"]
WINDOWS = {"1h": 3600, "1d": 86400, "1w": 7 * 86400}
FILTERS = {"one1%": ("one", 10), "one10%": ("one", 100), "all5%": ("all", 50), "all20%": ("all", 200),
           "all50%": ("all", 500)}
AFR_MODES = [("afr-k1", 1), ("afr-k2", 2), ("afr-k4", 4), ("afr-fix", 10_000)]


def prepare(name):
    """data/parquet/<name>.parquet with a, b (int32), ts (int64; 0 for static graphs), w (int32 in [0, 1000))."""
    out = PARQ / f"{name}.parquet"
    if out.exists():
        return out
    PARQ.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    if name in TEMPORAL:  # events: keep repeated (a, b) pairs at different times
        src = f"read_csv('{ROOT}/data/temporal/{name}.txt.gz', delim=' ', header=false, " \
              "columns={'a': 'INTEGER', 'b': 'INTEGER', 'ts': 'BIGINT'})"
        sel = f"SELECT a, b, ts FROM {src} WHERE a <> b"
    else:  # CRLF line ends and '#' headers defeat DuckDB's CSV sniffer; pandas reads them (as in afr.py)
        raw = pd.read_csv(ROOT / "data" / "raw" / f"{name}.txt", sep="\t", comment="#", header=None,
                          names=["a", "b"], dtype="int32")
        con.register("raw", raw)
        sel = "SELECT DISTINCT a, b, 0::BIGINT AS ts FROM raw WHERE a <> b"
    con.execute(f"COPY (SELECT a, b, ts, (hash(a, b, ts) % 1000)::INTEGER AS w FROM ({sel})) TO '{out}' "
                "(FORMAT parquet)")
    return out


def sql(kind, k, names, param):
    """Explicit JOIN ... ON form (RPT+ only sees joins written this way)."""
    f = [f"{names[i]} f{i + 1}" for i in range(k)]
    if kind == "temporal":
        q = f"SELECT count(*) FROM {f[0]}"
        for i in range(1, k):
            on = f"f{i}.b = f{i + 1}.a AND f{i}.ts < f{i + 1}.ts"
            if i == k - 1:
                on += f" AND f{k}.b = f1.a AND f{k}.ts <= f1.ts + {param}"
            q += f" JOIN {f[i]} ON {on}"
        return q
    which, thr = param
    q = f"SELECT count(*) FROM {f[0]} JOIN {f[1]} ON f1.b = f2.a JOIN {f[2]} ON f2.b = f3.a AND f3.b = f1.a"
    copies = [1] if which == "one" else [1, 2, 3]
    return q + " WHERE " + " AND ".join(f"f{i}.w < {thr}" for i in copies)


def duck13(setup, query, reps, timeout, threads, rpt):
    """Median wall time (s) and count of query in the DuckDB 1.3 CLI; (nan, None) on timeout or error."""
    script = f"SET threads = {threads};\nSET memory_limit = '96GB';\n{setup}\n.mode csv\n.headers off\n.timer on\n"
    script += (query + ";\n") * reps
    env = dict(os.environ, RPT_DISABLE="0" if rpt else "1")
    try:
        p = subprocess.run([str(DUCK13)], input=script, capture_output=True, text=True, timeout=timeout * reps + 120,
                           env=env)
    except subprocess.TimeoutExpired:
        return float("nan"), None
    times = [float(x) for x in re.findall(r"Run Time \(s\): real ([0-9.]+)", p.stdout)][-reps:]
    counts = [ln for ln in p.stdout.splitlines() if re.fullmatch(r"\d+", ln.strip())]
    if len(times) < reps or not counts:
        print("    duckdb 1.3 error:", (p.stderr or p.stdout)[-300:].strip(), flush=True)
        return float("nan"), None
    return statistics.median(times), int(counts[-1])


def timed13(setup, query, args, rpt):
    """One run under the timeout first; repeat for the median only if it is quick."""
    s, c = duck13(setup, query, 1, args.timeout, args.threads, rpt)
    if s == s and s < 20 and args.reps > 1:
        s, c = duck13(setup, query, args.reps, args.timeout, args.threads, rpt)
    return s, c


def duck20(path, query, args):
    con = duckdb.connect()
    con.execute(f"SET threads = {args.threads}")
    con.execute("SET memory_limit = '96GB'")
    con.execute(f"CREATE TABLE e AS FROM '{path}'")
    import threading
    out = []
    for _ in range(args.reps):
        timer = threading.Timer(args.timeout, con.interrupt)
        timer.start()
        t = time.perf_counter()
        try:
            c = con.execute(query).fetchall()[0][0]
            out.append(time.perf_counter() - t)
        except duckdb.InterruptException:
            return float("nan"), None
        finally:
            timer.cancel()
        if out[-1] > 20:
            break
    return statistics.median(out), c


def load_lib():
    if not LIB.exists() or LIB.stat().st_mtime < (HERE / "afr_cycle.cpp").stat().st_mtime:
        subprocess.run(["g++", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", str(HERE / "afr_cycle.cpp"),
                        "-o", str(LIB)], check=True)
    lib = ctypes.CDLL(str(LIB))
    P = ctypes.c_void_p
    lib.afr_cycle.argtypes = [ctypes.c_int, ctypes.c_int, P, P, P, P, ctypes.c_int64, P, ctypes.c_int, ctypes.c_int,
                              P, P, P, P, P]
    lib.afr_cycle.restype = ctypes.c_int
    return lib


def afr_reduce(lib, copies, k, gaps, temporal, max_rounds, threads, domain):
    """copies: k dicts of numpy columns. Returns reduced copies, per-round stats, compact seconds."""
    arr = lambda xs: (ctypes.c_void_p * k)(*[x.ctypes.data for x in xs])  # noqa: E731
    ids = [np.empty(len(c["a"]), "int64") for c in copies]
    rows = np.zeros(k, "int64")
    st_rows, st_secs = np.zeros(max(max_rounds, 1), "int64"), np.zeros(max(max_rounds, 1))
    csecs = ctypes.c_double()
    g = np.asarray(gaps, "int64")
    n = np.asarray([len(c["a"]) for c in copies], "int64")
    t = lib.afr_cycle(k, 1, arr([c["a"] for c in copies]), arr([c["b"] for c in copies]),
                      arr([c["ts"] for c in copies]) if temporal else None, n.ctypes.data, domain, g.ctypes.data,
                      max_rounds, threads, arr(ids), rows.ctypes.data, st_rows.ctypes.data, st_secs.ctypes.data,
                      ctypes.byref(csecs))
    t0 = time.perf_counter()
    out = [{col: v[ids[i][:rows[i]]] for col, v in copies[i].items()} for i in range(k)]
    gather = time.perf_counter() - t0
    rounds = [{"round": r + 1, "rows_after": int(st_rows[r]), "secs": float(st_secs[r])} for r in range(t)]
    return out, rounds, csecs.value + gather


def run_case(lib, args, name, path, base, kind, k, qname, param, results, round_rows):
    tag = dict(dataset=name, kind=kind, query=qname, k=k)
    base_names = ["e"] * k
    q = sql(kind, k, base_names, param)
    setup = f"CREATE TABLE e AS FROM '{path}';"
    ref = {}
    for system, fn in [("duckdb1.3", lambda: timed13(setup, q, args, False)),
                       ("rpt+", lambda: timed13(setup, q, args, True)),
                       ("duckdb2.0", lambda: duck20(path, q, args))]:
        s, c = fn()
        ref[system] = (s, c)
        results.append({**tag, "system": system, "rounds": 0, "rows": k * len(base["a"]), "reduce_s": 0,
                        "compact_s": 0, "join_s": s, "total_s": s, "count": c})
        print(f"  {qname:12s} {system:10s} {s:8.3f}s count={c}", flush=True)
    plain = ref["duckdb1.3"][0]
    counts = {c for _, c in ref.values() if c is not None}

    # per-copy inputs after the query's local filters (filter time counts toward AFR)
    t0 = time.perf_counter()
    if kind == "filtered":
        which, thr = param
        keep = base["w"] < thr
        fb = {c: v[keep] for c, v in base.items()}
        copies = [fb if (which == "all" or i == 0) else base for i in range(k)]
        gaps = [0] * k
    else:
        copies = [base] * k
        gaps = [1] * (k - 1) + [-param]
    filt_s = time.perf_counter() - t0
    domain = int(max(base["a"].max(), base["b"].max())) + 1
    modes = list(AFR_MODES) + ([("afr-eq-fix", 10_000)] if kind == "temporal" else [])
    if plain != plain:  # plain join timed out: try the fixpoint first, skip the rest if it times out too
        modes = [m for m in modes if m[0] == "afr-fix"] + [m for m in modes if m[0] != "afr-fix"]
    tmp = PARQ / f"tmp{args.tag}"
    tmp.mkdir(exist_ok=True)
    for mode, mr in modes:
        temporal = kind == "temporal" and mode != "afr-eq-fix"
        red, rounds, csecs = afr_reduce(lib, copies, k, gaps, temporal, mr, args.threads, domain)
        red_s = filt_s + sum(r["secs"] for r in rounds)
        names = []
        for i, c in enumerate(red):
            p = tmp / f"f{i + 1}.parquet"
            duckdb.from_df(pd.DataFrame(c)).write_parquet(str(p))
            names.append(f"f{i + 1}x")
        setup_r = "\n".join(f"CREATE TABLE f{i + 1}x AS FROM '{tmp}/f{i + 1}.parquet';" for i in range(k))
        js, c = timed13(setup_r, sql(kind, k, names, param), args, False)
        nrows = sum(len(x["a"]) for x in red)
        total = red_s + csecs + js
        results.append({**tag, "system": mode, "rounds": len(rounds), "rows": nrows, "reduce_s": red_s,
                        "compact_s": csecs, "join_s": js, "total_s": total, "count": c})
        for r in rounds:
            round_rows.append({**tag, "mode": mode, **r})
        if plain != plain and mode == "afr-fix" and js != js:
            print(f"  {qname:12s} afr-fix timed out too; skipping the other modes", flush=True)
            break
        bad = "" if c is None or not counts or c in counts else f"  COUNT MISMATCH {counts}"
        print(f"  {qname:12s} {mode:10s} {total:8.3f}s rounds={len(rounds):4d} rows={nrows:10d} "
              f"reduce={red_s:.3f}s join={js:.3f}s  vs1.3 {plain / total:6.2f}x  vsRPT+ {ref['rpt+'][0] / total:6.2f}x"
              f"{bad}", flush=True)


def save(rows, f, name):
    path = RESULTS / f
    df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        redone = old.dataset.eq(name) & old["query"].isin(set(df["query"]))  # keep this dataset's other queries
        df = pd.concat([old[~redone], df])
    df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=TEMPORAL + STATIC)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--queries", default="tri,c4")
    ap.add_argument("--tag", default="", help="suffix for result files and the temp dir (parallel jobs)")
    args = ap.parse_args()
    lib = load_lib()
    RESULTS.mkdir(exist_ok=True)
    for name in args.datasets:
        path = prepare(name)
        e = duckdb.connect().execute(f"SELECT a, b, ts, w FROM '{path}'").fetchnumpy()
        base = {c: np.ascontiguousarray(e[c]) for c in ["a", "b", "ts", "w"]}
        print(f"{name}: {len(base['a'])} edges", flush=True)
        results, round_rows = [], []
        for qn in args.queries.split(","):
            k = {"tri": 3, "c4": 4}[qn]
            if name in TEMPORAL:
                for wn, d in WINDOWS.items():
                    run_case(lib, args, name, path, base, "temporal", k, f"{qn}-{wn}", d, results, round_rows)
            elif k == 3:
                for fn, p in FILTERS.items():
                    run_case(lib, args, name, path, base, "filtered", k, f"{qn}-{fn}", p, results, round_rows)
            save(results, f"afr_vs_rpt{args.tag}.csv", name)
            save(round_rows, f"afr_vs_rpt_rounds{args.tag}.csv", name)


if __name__ == "__main__":
    main()
