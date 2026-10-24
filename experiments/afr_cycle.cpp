// AFR over a chain/cycle query f_1 .. f_k with join edges f_i.b = f_{i+1}.a (plus f_k.b = f_1.a if cyclic),
// where each copy f_i has its own (already filtered) rows and, optionally, a timestamp with one difference
// constraint per query edge:  ts_{i+1} - ts_i >= gap_i  (gap 1 = strictly later; the closing edge of a temporal
// cycle within a window D is ts_1 - ts_k >= -D).
//
// A semi-join from build copy i to probe copy j = i+1 (forward) keeps a row r of f_j iff some surviving row of
// f_i has b = r.a and satisfies the edge's constraint, i.e.
//   equality only:  bit[r.a]                     (bitset over the key domain, as in afr_kernel.cpp)
//   temporal:       r.ts - min_ts_i[r.a] >= gap  (min over surviving f_i rows per key; +inf if none)
// and backward (probe copy i against build copy j) keeps r of f_i iff  max_ts_j[r.b] - r.ts >= gap.
// One round = forward sweep around the query edges, then backward sweep; a semi-join is skipped if its build
// copy has not lost rows since it last ran. Rounds repeat up to max_rounds or until nothing is removed.
// Each copy keeps per-chunk (2048-row) selection vectors that shrink in place; survivors are returned as row ids.
//
// Build: g++ -O3 -march=native -fopenmp -shared -fPIC afr_cycle.cpp -o libafr_cycle.so

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int64_t VECTOR_SIZE = 2048;
constexpr int64_t INF = std::numeric_limits<int64_t>::max() / 4;

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Copy {
    const int32_t *col[2];  // a, b
    const int64_t *ts;      // nullptr for equality-only queries
    int64_t n, nchunks;
    std::vector<uint16_t> sel;
    std::vector<int32_t> count;
    int64_t rows, version = 0;

    Copy(const int32_t *a, const int32_t *b, const int64_t *t, int64_t n_)
        : col{a, b}, ts(t), n(n_), nchunks((n_ + VECTOR_SIZE - 1) / VECTOR_SIZE), sel(nchunks * VECTOR_SIZE),
          count(nchunks), rows(n_) {
        for (int64_t c = 0; c < nchunks; c++) {
            int64_t m = std::min(VECTOR_SIZE, n - c * VECTOR_SIZE);
            count[c] = (int32_t)m;
            for (int64_t j = 0; j < m; j++) sel[c * VECTOR_SIZE + j] = (uint16_t)j;
        }
    }
};

void build_bits(std::vector<uint64_t> &bs, const Copy &s, int k) {
    std::fill(bs.begin(), bs.end(), 0);
#pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < s.nchunks; c++) {
        const int32_t *kc = s.col[k] + c * VECTOR_SIZE;
        const uint16_t *sc = s.sel.data() + c * VECTOR_SIZE;
        for (int32_t j = 0; j < s.count[c]; j++) {
            uint32_t v = (uint32_t)kc[sc[j]];
            uint64_t m = 1ULL << (v & 63);
            if (!(__atomic_load_n(&bs[v >> 6], __ATOMIC_RELAXED) & m)) __atomic_fetch_or(&bs[v >> 6], m, __ATOMIC_RELAXED);
        }
    }
}

// ext[v] = min (want_min) or max of ts over surviving rows with column k = v
void build_ext(std::vector<int64_t> &ext, const Copy &s, int k, bool want_min) {
    std::fill(ext.begin(), ext.end(), want_min ? INF : -INF);
#pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < s.nchunks; c++) {
        const int32_t *kc = s.col[k] + c * VECTOR_SIZE;
        const int64_t *tc = s.ts + c * VECTOR_SIZE;
        const uint16_t *sc = s.sel.data() + c * VECTOR_SIZE;
        for (int32_t j = 0; j < s.count[c]; j++) {
            int64_t *p = &ext[(uint32_t)kc[sc[j]]];
            int64_t t = tc[sc[j]], cur = __atomic_load_n(p, __ATOMIC_RELAXED);
            while (want_min ? t < cur : t > cur)
                if (__atomic_compare_exchange_n(p, &cur, t, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) break;
        }
    }
}

// Shrink s's selection: keep rows whose column k passes. fwd: probe is the later copy (ts - ext >= gap),
// otherwise the earlier one (ext - ts >= gap). Returns rows removed.
int64_t probe(Copy &s, int k, const std::vector<uint64_t> *bs, const std::vector<int64_t> *ext, bool fwd,
              int64_t gap) {
    int64_t removed = 0;
#pragma omp parallel for schedule(static) reduction(+ : removed)
    for (int64_t c = 0; c < s.nchunks; c++) {
        int32_t cnt = s.count[c];
        if (cnt == 0) continue;
        const int32_t *x = s.col[k] + c * VECTOR_SIZE;
        uint16_t *sc = s.sel.data() + c * VECTOR_SIZE;
        int32_t m = 0;
        if (bs) {
            for (int32_t j = 0; j < cnt; j++) {
                uint16_t r = sc[j];
                uint32_t v = (uint32_t)x[r];
                sc[m] = r;
                m += (*bs)[v >> 6] >> (v & 63) & 1;
            }
        } else {
            const int64_t *tc = s.ts + c * VECTOR_SIZE;
            for (int32_t j = 0; j < cnt; j++) {
                uint16_t r = sc[j];
                int64_t e = (*ext)[(uint32_t)x[r]];
                sc[m] = r;
                m += fwd ? tc[r] - e >= gap : e - tc[r] >= gap;
            }
        }
        s.count[c] = m;
        removed += cnt - m;
    }
    s.rows -= removed;
    if (removed) s.version++;
    return removed;
}

}  // namespace

extern "C" {

// a[i], b[i], ts[i] (ts may be null for all copies), n[i]: copy i. gap[e]: constraint of query edge e
// (f_e.b = f_{e+1 mod k}.a). out_ids[i]: room for n[i] row ids; out_rows[i] gets the survivor count.
// st_rows[t] (rows of all copies after round t) and st_secs[t] for up to max_rounds rounds. Returns rounds run.
int afr_cycle(int k, int cyclic, const int32_t **a, const int32_t **b, const int64_t **ts, const int64_t *n,
              int64_t domain, const int64_t *gap, int max_rounds, int threads, int64_t **out_ids, int64_t *out_rows,
              int64_t *st_rows, double *st_secs, double *compact_secs) {
    omp_set_num_threads(threads);
    std::vector<Copy> f;
    for (int i = 0; i < k; i++) f.emplace_back(a[i], b[i], ts ? ts[i] : nullptr, n[i]);
    bool temporal = ts != nullptr;
    std::vector<uint64_t> bits(temporal ? 0 : (domain >> 6) + 1);
    std::vector<int64_t> ext(temporal ? domain : 0);

    struct SJ { int pc, pk, bc, bk; bool fwd; int64_t gap, seen; };
    std::vector<SJ> sweep;
    int ne = cyclic ? k : k - 1;
    for (int e = 0; e < ne; e++) sweep.push_back({(e + 1) % k, 0, e, 1, true, gap[e], -1});
    for (int e = ne - 1; e >= 0; e--) sweep.push_back({e, 1, (e + 1) % k, 0, false, gap[e], -1});

    int t = 0;
    while (t < max_rounds) {
        double t0 = now();
        int64_t removed = 0;
        for (auto &s : sweep) {
            if (s.seen == f[s.bc].version) continue;
            s.seen = f[s.bc].version;
            if (temporal) {
                build_ext(ext, f[s.bc], s.bk, s.fwd);  // forward needs the earliest build ts, backward the latest
                removed += probe(f[s.pc], s.pk, nullptr, &ext, s.fwd, s.gap);
            } else {
                build_bits(bits, f[s.bc], s.bk);
                removed += probe(f[s.pc], s.pk, &bits, nullptr, s.fwd, 0);
            }
        }
        int64_t total = 0;
        for (auto &c : f) total += c.rows;
        st_rows[t] = total;
        st_secs[t] = now() - t0;
        t++;
        if (removed == 0) break;
    }

    double t0 = now();
    for (int i = 0; i < k; i++) {
        auto &s = f[i];
        std::vector<int64_t> off(s.nchunks + 1, 0);
        for (int64_t c = 0; c < s.nchunks; c++) off[c + 1] = off[c] + s.count[c];
#pragma omp parallel for schedule(static)
        for (int64_t c = 0; c < s.nchunks; c++)
            for (int32_t j = 0; j < s.count[c]; j++) out_ids[i][off[c] + j] = c * VECTOR_SIZE + s.sel[c * VECTOR_SIZE + j];
        out_rows[i] = off[s.nchunks];
    }
    *compact_secs = now() - t0;
    return t;
}
}
