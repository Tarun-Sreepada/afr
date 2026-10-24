# AFR: Adaptive Fixpoint Reduction

**Verdict: AFR did not work out.** Repeating semi-join rounds past the first one does not pay on real cyclic
workloads. It pays only on synthetic deep dead-end traps. The one positive result is a single round of
*time-aware* semi-join filtering on temporal cycles (1.2–2× over RPT+ on the cheaper queries). That is a
one-pass extension of predicate transfer, not a fixpoint. This repo holds the code, data pipeline and results
of that investigation.

## What AFR is

Before a multi-way join, drop every row that cannot take part in any result (a *semi-join reduction*). For an
**acyclic** join (tables linked like a tree, e.g. a star schema) one pass inward and one outward removes every
such row (Yannakakis). Predicate transfer systems (RPT, RPT+, Parachute) stop after those passes.

For a **cyclic** join the conditions form a loop, e.g. directed triangles in a graph:

```sql
SELECT count(*)
FROM edges f1 JOIN edges f2 ON f1.b = f2.a
              JOIN edges f3 ON f2.b = f3.a AND f3.b = f1.a;   -- x → y → z → x
```

Removing dead rows from one copy can make rows of another copy dead, which can make more rows of the first dead,
around the loop. Getting them all needs repeated rounds, up to a fixpoint. AFR repeats rounds while a stopping
rule says the next one is worth it, then compacts the survivors and runs the normal join.

### Example: a dead-end trap

A hub vertex H with 10,000 incoming chains of depth d and 10,000 outgoing chains of depth d that all dead-end,
plus 100 real triangles:

```
c1 → c2 → … → cd → H → o1 → o2 → … → od   (×10,000 on each side; no chain closes a cycle)
```

The join builds 10,000 × 10,000 = 10^8 two-hop paths through H, none of which closes a triangle. Each semi-join
round peels a few hops off the dead ends. Once the peeling reaches H, the hub's edges die and the blow-up
disappears. With d = 8, two fixed rounds (RPT-style) leave H alive (0.36×). Running to the fixpoint costs
milliseconds and gives 237× over plain DuckDB.

### Example: a temporal cycle

```sql
-- message cascades x → y → z → x with increasing timestamps, closing within one hour
SELECT count(*)
FROM msgs f1 JOIN msgs f2 ON f1.b = f2.a AND f1.ts < f2.ts
             JOIN msgs f3 ON f2.b = f3.a AND f2.ts < f3.ts AND f3.b = f1.a AND f3.ts <= f1.ts + 3600;
```

RPT+ only transfers the equality part (`f1.b = f2.a`). AFR's temporal semi-join keeps, per key, the earliest
(forward) or latest (backward) surviving timestamp instead of one bit. A row `x → y` at time t survives only if
`y` has an outgoing edge later than t. It survives the backward pass only if `x` has an incoming edge earlier
than t.

## Conclusion

| Question | Answer |
|---|---|
| Do rounds past the first pay on real graphs? | **No.** On every SNAP graph, round 1 gets essentially all of the gain. Cases where 1 round *beat* the fixpoint turned out to be DuckDB picking a different hash-join build side (see *The one-round anomaly*); with the plan fixed, all round counts run equally fast. |
| Does the fixpoint ever pay? | Only on synthetic dead-end chains of depth ≥ 8 leading into a hub (up to ~240× on triangles). |
| Does AFR beat RPT+? | On temporal cycles, by 1.2–2× on cheap queries, from **one** round of per-key min/max timestamp bounds. That's not a fixpoint effect, and the gain vanishes on expensive queries. |
| Does it rescue queries that time out? | No (email-Eu-core temporal 4-cycles: plain DuckDB and the AFR fixpoint both exceed 300 s). |
| Do stopping rules matter? | A row-fraction rule (stop when a round removes < 5%) misses traps embedded in large tables. A time-budget rule catches them, at up to ~10% overhead when there is nothing to gain. |
| Does REFIM-style threshold pruning (`HAVING SUM >= M`) cascade instead? | Not in non-recursive SQL. See *Threshold pruning* below. |

**Why it doesn't pay.**
- **Hubs don't die.** A semi-join is an existence test, and real high-degree vertices sit in the dense core of
  the graph. One surviving in-edge and one surviving out-edge keep a hub, and all its other edges, alive.
- **What does die is cheap.** The rows a fixpoint removes are low-degree periphery. Deleting them barely changes
  the join's intermediate result (web-Google: 5.0e7 → 4.98e7 two-hop paths).
- **A cascade over several rounds needs a deep dead chain feeding a hub.** That shape appears in the synthetic
  traps and nowhere in the SNAP data.

## Experiments and results

### 1. Reduction in plain SQL (`experiments/afr.py`)

Each round is a SQL statement that keeps an edge only if its endpoints still have the edges the pattern needs. On
soc-Epinions1, soc-Slashdot0902, email-EuAll and web-Google (triangles, 4-cycles, 2-hop paths) the best speedup
over plain DuckDB 2.0 was 1.27× (email-EuAll triangles, 1 round). Fixpoint reduction was 0.14–1.07×. Results:
`results/summary.csv`, `results/rounds.csv`, `results/joins.csv`, plots in `results/*.png`.

### 2. Native reduction operator (`experiments/afr_kernel.cpp`, `experiments/afr_native.py`)

A C++ version of the operator design:
- 2,048-row chunks, with one selection vector per chunk and per join copy
- bitset semi-joins on the integer keys
- a semi-join is skipped when its build copy hasn't changed
- survivors are compacted at the end

DuckDB 2.0 then runs the join on the survivors. Loading the survivors into DuckDB is not counted, since a
built-in operator wouldn't pay it.

Two schemes:
- **per-copy:** RPT-style forward + backward sweep over f1..fk, where one round peels several hops.
- **shared:** self-join cycles only; one table, one build and one probe per round.

Stopping rules:
- `rho`: stop when a round removes < 5% of rows.
- `cost`: the cost formula τ* = 2c_s / (c_j·α), with c_s measured live.
- `work`: stop when the last round's drop in two-hop count W isn't worth the round.
- `budget`: stop once reduction time > 10% of the estimated join cost.

Cost constants are calibrated on a random graph, not on the test data. 16 threads.

Triangles, speedup over plain DuckDB including reduction time:

| data | plain | k1 | k2 (RPT-like) | rho 5% | budget 10% | fixpoint |
|---|---|---|---|---|---|---|
| trap3 | 2.0 s | 812× | 922× | 988× | 878× | 985× |
| trap4 | 0.95 s | 0.52× | 306× | 346× | 301× | 360× |
| trap8 | 0.76 s | 0.44× | **0.36×** | 186× | 126× | 237× |
| trap16 | 0.42 s | 0.38× | **0.57×** | 50× | 27× | 61× |
| soc-Epinions1 | 0.31 s | 1.14× | 1.05× | 1.07× | 0.99× | 1.02× |
| soc-Slashdot0902 | 0.61 s | 0.90× | 0.96× | 0.87× | 0.85× | 0.87× |
| email-EuAll | 0.28 s | 1.35× | 1.41× | 1.37× | 1.35× | 1.30× |
| web-Google | 0.30 s | 0.79× | 0.74× | 0.73× | 0.77× | 0.54× |
| web-Google + trap4 (shared scheme) | 2.19 s | 0.71× | 1.03× | **1.04×** | **5.65×** | 4.08× |

4-cycles look the same (traps up to ~3000×, real graphs 0.75–1.14×). 2-hop paths (acyclic) give 0.8–1.4×.
Results: `results/native.csv`, `results/native_rounds.csv`, `results/native.log`.

### 3. AFR vs RPT+ (`experiments/afr_cycle.cpp`, `experiments/afr_vs_rpt.py`)

Systems:
- **plain DuckDB 1.3:** the RPT+ binary with predicate transfer switched off (`RPT_DISABLE=1`, our patch)
- **RPT+:** [embryo-labs/dynamic-predicate-transfer](https://github.com/embryo-labs/dynamic-predicate-transfer),
  DuckDB 1.3, commit `3ce6a1f`
- **DuckDB 2.0:** the Python package
- **AFR:** reduce in C++ for 1, 2 or 4 rounds or to the fixpoint, then run the same SQL in plain DuckDB 1.3 on the
  reduced copies. AFR time = filter + reduce + compact + join.
- **`afr-eq-fix`:** the same fixpoint with equality semi-joins only, ignoring timestamps

Queries:
- **temporal:** triangles and 4-cycles with increasing timestamps, closing within 1 hour, 1 day or 1 week, on
  SNAP temporal graphs
- **filtered:** equality triangles on static SNAP graphs, with a filter on a random edge attribute `w`, either on
  f1 only (`one1%`, `one10%`) or on every copy (`all5%`, `all20%`, `all50%`)

Setup: 16 threads per job, two jobs pinned one per socket with `numactl`. Median of 3 runs for queries under
20 s, otherwise a single run. 300 s timeout. All counts match across systems.

Speedup over plain DuckDB 1.3 (selected rows; full data in `results/afr_vs_rpt_s0.csv`, `results/afr_vs_rpt_s1.csv`):

| data | query | plain 1.3 | RPT+ | AFR k1 | AFR k2 | AFR fixpoint | AFR equality-only |
|---|---|---|---|---|---|---|---|
| CollegeMsg (60K events) | tri, 1h | 0.08 s | 0.99× | 1.18× | 1.27× | 1.24× | 1.02× |
| CollegeMsg | c4, 1h | 3.4 s | 1.01× | 1.23× | 1.20× | 1.23× | 0.97× |
| email-Eu-core-temporal (332K) | tri, 1h | 20.5 s | 1.02× | 1.40× | 1.43× | 1.22× | 1.01× |
| email-Eu-core-temporal | c4, 1h / 1d / 1w | > 300 s | timeout | — | — | timeout | — |
| sx-mathoverflow (390K) | tri, 1h | 1.4 s | 0.98× | 1.36× | 1.21× | 1.19× | 1.11× |
| sx-mathoverflow | c4, 1h | 127 s | 0.98× | 0.99× | 1.08× | 1.07× | 1.11× |
| sx-askubuntu (727K) | tri, 1h | 0.92 s | 1.00× | 1.54× | 1.21× | 1.13× | 1.15× |
| sx-askubuntu | c4, 1w | 18.4 s | 0.98× | 0.94× | 0.76× | 0.77× | 1.00× |
| sx-superuser (1.1M) | tri, 1h | 1.6 s | 1.00× | 2.01× | 1.17× | 1.08× | 0.98× |
| sx-superuser | c4, 1h | 28.3 s | 1.00× | 1.25× | 1.38× | 1.34× | 1.01× |
| wiki-talk-temporal (6.1M) | tri, 1h | 4.7 s | 1.00× | 1.70× | 2.04× | 1.47× | 1.46× |
| soc-Epinions1 | tri, one1% | 0.02 s | 1.11× | 0.57× | 0.64× | 0.62× | — |
| soc-Slashdot0902 | tri, all20% | 0.02 s | 1.05× | 0.28× | 0.27× | 0.28× | — |
| email-EuAll | tri, all50% | 0.16 s | 1.02× | 1.94× | 2.26× | 2.31× | — |
| web-Google | tri, one1% | 0.03 s | 1.29× | 0.51× | 0.50× | 0.46× | — |

Findings:
- **RPT+ is plain DuckDB on these queries (0.98–1.04×).** It builds Bloom filters only from filtered (or
  intermediate) tables, and only along a spanning tree of the join graph, so one edge of each cycle is never
  used. It ignores inequality predicates. And it only sees joins written as `JOIN … ON`, because it inspects them
  before join ordering.
- **The temporal gain comes from round 1 of the timestamp bounds.** The equality-only fixpoint gives 1.0–1.15×
  on most graphs (1.46× on wiki-talk). Going from 1 round to the fixpoint adds nothing.
- **Expensive queries don't benefit.** The cost is in two-hop paths through hubs, and hubs have early and late
  edges in every direction, so they pass the timestamp test.
- **The filtered static triangles take milliseconds.** Reduction overhead outweighs the join, so AFR mostly
  loses (0.3–0.6× on soc-Epinions1, soc-Slashdot0902 and web-Google). email-EuAll is the exception (up to 2.3×),
  but its queries take 0.01–0.16 s, so the absolute differences are tens of milliseconds.
- **1 round sometimes beats the fixpoint, but that's a plan effect.** See *The one-round anomaly* below.
- **Coverage:** when this README was last updated the sweep was still running wiki-talk 4-cycles and the
  sx-stackoverflow triangles. sx-stackoverflow 4-cycles were skipped (expected to time out everywhere). The CSVs
  hold whatever had finished.
- `results/afr_vs_rpt_unpinned.*` is an earlier run without `numactl` pinning, kept for reference.
- The largest joins (4-cycles on the bigger temporal graphs) spill to disk (DuckDB's `.tmp/`) even with a 96 GB
  memory limit, so their timings include spill I/O for every system.

### The one-round anomaly (`experiments/one_round_anomaly.py`)

In several temporal cases AFR after 1 round was much faster than after 2 rounds or at the fixpoint, although
the fixpoint keeps fewer rows (sx-superuser triangles, 1h: 2.01× vs 1.08×; sx-askubuntu: 1.54× vs 1.13×). The
script reduces one case with 1 round, 2 rounds and to the fixpoint, and profiles DuckDB 1.3's join on each. It
also runs each with join ordering and build-side selection turned off
(`disabled_optimizers = 'join_order,build_side_probe_side'`), building the lower join's hash table on f2 (as the
SQL is written) and on f1 (FROM order swapped).

Triangles, 1-hour window, join time (median of 5, 8 threads, run while the sweep occupied the machine, so only
relative numbers matter):

| case | mode | 2-hop rows | DuckDB's own plan | fixed plan, build f2 | fixed plan, build f1 |
|---|---|---|---|---|---|
| sx-askubuntu | no reduction | 139.7M | 0.98 s | 1.00 s | 1.00 s |
| sx-askubuntu | 1 round | 78.6M | 0.57 s (builds f1) | 0.75 s | 0.58 s |
| sx-askubuntu | 2 rounds | 76.3M | 0.74 s (builds f2) | 0.73 s | 0.55 s |
| sx-askubuntu | fixpoint (11) | 75.5M | 0.72 s (builds f2) | 0.75 s | 0.55 s |
| sx-superuser | 1 round | 150.6M | 0.77 s (builds f1) | 1.35 s | 0.77 s |
| sx-superuser | 2 rounds | 146.6M | 1.32 s (builds f2) | 1.32 s | 0.76 s |
| sx-superuser | fixpoint (17) | 145.0M | 1.32 s (builds f2) | 1.35 s | 0.76 s |

- **The anomaly is the build side.** DuckDB builds the hash table on the smaller copy. After 1 round f1 is
  smallest; after more rounds f2 is, and building on f2 is 1.3–1.8× slower on these graphs. With the plan held
  fixed, 1 round, 2 rounds and the fixpoint run equally fast. The extra rounds remove only 2–4% more two-hop rows.
- **So extra rounds don't hurt, and they don't help either.** The conclusion that round 1 gets the gain stands.
- **Part of AFR's measured gain depends on this plan choice.** On unreduced input both build sides cost the same.
  On sx-askubuntu, AFR's speedup after reduction is about 1.3× with the slow side and 1.8× with the fast one. Which side
  wins depends on the data, and a cost-based engine would have to model it. The per-operator profiles show
  similar summed operator time for both sides while wall time differs, which suggests a difference in parallel
  efficiency rather than in total work; this was not investigated further.
- Profiles and full output: `results/anomaly/`.

### Threshold pruning (desk check, no code)

REFIM's multi-round gains come from pruning by an upper bound ("total utility ≥ min_util") that cascades. The
SQL analogue would be `GROUP BY g HAVING SUM(x) >= M` with pruning of rows whose groups can't reach M.

For monotone aggregates (`COUNT`, `SUM` of non-negative values, `MAX`) in non-recursive SQL, exact pruning is
complete after one round. A row is dead only if all its join results land in failing groups. Removing dead rows
leaves passing groups unchanged and only lowers failing groups. A row that had a join result in a passing group
still has it, so no new row becomes dead.

Cascades over several rounds come only from slack in *estimated* group totals. For acyclic queries, exact totals
take one pass (eager aggregation, factorized execution). For cyclic ones, the slack is the same mechanism AFR
already exploits. Or cascades come from recursive definitions (k-core, k-truss), which is the recursive
aggregation literature, not a join optimization. So this idea was not pursued.

## Layout

```
experiments/
  afr.py                         experiment 1: reduction rounds in plain SQL (DuckDB 2.0)
  plot.py                        plots for experiment 1
  afr_kernel.cpp, afr_native.py  experiment 2: native operator with stopping rules, traps
  afr_cycle.cpp, afr_vs_rpt.py   experiment 3: per-copy equality/temporal reduction vs RPT+
  one_round_anomaly.py           profiles why 1 round can beat the fixpoint (hash-join build side)
  rptplus_disable_toggle.patch   adds RPT_DISABLE=1 to the RPT+ fork (plain DuckDB 1.3 baseline)
results/                         CSVs, logs and plots of all runs
data/                            downloaded graphs and Parquet files (not committed)
systems/rptplus/                 the RPT+ fork, cloned and built locally (not committed)
```

## Setup

Requirements: Linux, `g++` with OpenMP (tested with GCC 13), [uv](https://docs.astral.sh/uv/), and for
experiment 3, CMake + Ninja and `numactl`.

```bash
uv sync    # Python 3.12, duckdb 2.0, pandas, numpy, matplotlib
```

### Download the data

```bash
# static SNAP graphs (experiments 1–3)
mkdir -p data/raw && cd data/raw
for g in soc-Epinions1 soc-Slashdot0902 email-EuAll web-Google; do
  curl -fL https://snap.stanford.edu/data/$g.txt.gz | gunzip > $g.txt
done
cd ../..

# SNAP temporal graphs (experiment 3; sx-stackoverflow is ~530 MB compressed)
mkdir -p data/temporal && cd data/temporal
for g in CollegeMsg email-Eu-core-temporal sx-mathoverflow sx-askubuntu sx-superuser wiki-talk-temporal sx-stackoverflow; do
  curl -fLO https://snap.stanford.edu/data/$g.txt.gz
done
cd ../..
```

`afr_vs_rpt.py` converts each graph to `data/parquet/<name>.parquet` on first use (temporal graphs keep repeated
`(a, b)` events; static graphs are de-duplicated; self-loops are dropped).

### Compile the kernels

The Python scripts compile them automatically when the `.so` is missing or older than the source. By hand:

```bash
g++ -O3 -march=native -fopenmp -shared -fPIC experiments/afr_kernel.cpp -o experiments/libafr.so
g++ -O3 -march=native -fopenmp -shared -fPIC experiments/afr_cycle.cpp  -o experiments/libafr_cycle.so
```

### Build RPT+ (experiment 3)

```bash
mkdir -p systems && git clone https://github.com/embryo-labs/dynamic-predicate-transfer.git systems/rptplus
cd systems/rptplus
git checkout 3ce6a1f3df36661267375b6d76448d72be903029
git apply ../../experiments/rptplus_disable_toggle.patch
GEN=ninja make release -j"$(nproc)"     # → build/release/duckdb
cd ../..
```

`RPT_DISABLE=1 systems/rptplus/build/release/duckdb` then runs plain DuckDB 1.3, and without the variable it runs
RPT+.

## Running the experiments

```bash
# 1. reduction rounds in SQL
uv run python experiments/afr.py [graph ...] [--threads 16] [--reps 3] [--timeout 300]
uv run python experiments/plot.py

# 2. native operator, stopping rules, traps
uv run python experiments/afr_native.py [dataset ...] [--threads 16] [--fan 10000]
#    datasets: trap1 … trap16, soc-Epinions1, soc-Slashdot0902, email-EuAll, web-Google, web-Google+trap4

# 3. AFR vs RPT+ (temporal and filtered cycles); one job per socket
numactl --cpunodebind=0 --membind=0 uv run python experiments/afr_vs_rpt.py \
    email-Eu-core-temporal sx-superuser email-EuAll web-Google sx-stackoverflow \
    --tag _s0 > results/afr_vs_rpt_s0.log 2>&1 &
numactl --cpunodebind=1 --membind=1 uv run python experiments/afr_vs_rpt.py \
    CollegeMsg sx-mathoverflow sx-askubuntu soc-Epinions1 soc-Slashdot0902 wiki-talk-temporal \
    --tag _s1 > results/afr_vs_rpt_s1.log 2>&1 &
#    options: --queries tri,c4  --timeout 300  --reps 3  --threads 16
#    (the published run stopped sx-stackoverflow after its triangles; use --queries tri for it)
```

```bash
# the one-round anomaly: profiles + fixed-plan runs for one temporal triangle case
numactl --cpunodebind=1 --membind=1 uv run python experiments/one_round_anomaly.py sx-askubuntu --window 3600 --fixed-plan
```

Each script writes CSVs to `results/` and replaces earlier rows for the datasets (experiment 3: dataset and
query) it reruns.

Notes:
- DuckDB counts a directed triangle once per rotation (3 times) with the queries above, and all systems agree on
  the counts.
- The machine used: 2 × AMD EPYC 9224 (24 cores each, 2 NUMA nodes), 377 GB RAM, shared with other users. Pin
  thread counts and sockets so runs are comparable.

## Where it came from

REFIM (a high-utility itemset miner) repeats bound refinement on each projected database until nothing drops. A
sweep of the number of rounds there gave 2.2–2.5× on BMS and more than 2.4× on kosarak (sparse, long-tailed
inputs; kosarak kept paying through round 4), and a slight cost on dense ones. AFR asked whether cyclic joins on real graphs behave like kosarak. They don't, because semi-joins
test existence and REFIM's bounds are sums.

## Related work

- Predicate transfer: Yang et al., CIDR 2024; Robust Predicate Transfer, SIGMOD 2025 (full reduction for acyclic
  queries, in DuckDB); RPT+ (Qiao, Boncz, Zhang, *Robust Predicate Transfer with Dynamic Execution*, PVLDB 19(6),
  2026), the baseline in experiment 3; Parachute, arXiv 2025 (single pass).
- Yannakakis's full reducer for acyclic joins.
- Data chunk compaction in vectorized execution (Qiao & Zhang, SIGMOD 2025).
- Factorized execution in DuckDB (Groß, ten Wolde, Boncz, CIDR 2025). Umbra: Lehner & Neumann, VLDB 2026.
- Worst-case optimal joins for cyclic queries: Ngo et al.; Leapfrog Triejoin; on GPUs, SRDatalog 2026.
- Graph queries in DuckDB: DuckPGQ.
