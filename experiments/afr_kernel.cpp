// AFR reduction operator, modelled on the PhysicalAdaptiveFixpoint design: the edge table is split into
// 2048-row chunks, every copy f_i of it in the join keeps one in-place selection vector per chunk, a semi-join
// builds a direct bitset over the integer key domain from the surviving keys of one copy and shrinks the
// selection of another, and after the last round the survivors are compacted into dense arrays.
//
// Two reduction schemes:
//   per-copy  one selection per copy f_1..f_k; a round is a forward sweep f_{i+1}.a IN f_i.b around the query
//             edges and a backward sweep f_i.b IN f_{i+1}.a (RPT-style; works for any chain or cycle)
//   shared    self-join cycles only: one selection; an edge survives if its a is some surviving b and its b is
//             some surviving a (one build and one probe per round)
//
// Stopping rules (after every round, besides "nothing removed" and the round cap):
//   FIXED  run max_rounds rounds
//   RHO    stop if rho_t = removed / rows_before < tau                                  (blueprint rule)
//   COST   stop if rho_t < tau* = 2 c_s / (c_j * alpha_t), c_s measured from the round just run,
//          alpha_t = W_t / N_t, c_j given (seconds per join input row)                  (blueprint cost model)
//   WORK   stop if c_w * (W_{t-1} - W_t) < round seconds, W = size of the first binary join f_1 JOIN f_2,
//          c_w given (seconds per intermediate row)                                     (work-aware rule)
//   BUDGET stop once the reduction time spent so far exceeds tau * c_w * W_t, the estimated join cost left:
//          later rounds may pay off only when a cascade reaches a hub, so instead of predicting the next
//          round's gain, cap the overhead at a (1 + tau) factor of the join it might save   (budget rule)
//
// Build: g++ -O3 -march=native -fopenmp -shared -fPIC afr_kernel.cpp -o libafr.so

#include <omp.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr int64_t VECTOR_SIZE = 2048;
enum Rule { FIXED = 0, RHO = 1, COST = 2, WORK = 3, BUDGET = 4 };

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Selection {  // one copy f_i: per-chunk selection vectors over the shared base columns
    std::vector<uint16_t> sel;    // sel[c * VECTOR_SIZE + j], j < count[c]
    std::vector<int32_t> count;
    int64_t rows = 0;
    int64_t version = 0;          // bumped whenever rows are removed
};

struct Ctx {
    const int32_t *col[2];        // col[0] = a, col[1] = b
    int64_t n, nchunks, domain;
    std::vector<uint64_t> bits;

    Selection fresh() const {
        Selection s;
        s.sel.resize(nchunks * VECTOR_SIZE);
        s.count.resize(nchunks);
        for (int64_t c = 0; c < nchunks; c++) {
            int64_t m = std::min(VECTOR_SIZE, n - c * VECTOR_SIZE);
            s.count[c] = (int32_t)m;
            for (int64_t j = 0; j < m; j++) s.sel[c * VECTOR_SIZE + j] = (uint16_t)j;
        }
        s.rows = n;
        return s;
    }

    void build(std::vector<uint64_t> &bs, const Selection &s, int k) const {
        std::fill(bs.begin(), bs.end(), 0);
        const int32_t *key = col[k];
#pragma omp parallel for schedule(static)
        for (int64_t c = 0; c < nchunks; c++) {
            const int32_t *kc = key + c * VECTOR_SIZE;
            const uint16_t *sc = s.sel.data() + c * VECTOR_SIZE;
            for (int32_t j = 0; j < s.count[c]; j++) {
                uint32_t v = (uint32_t)kc[sc[j]];
                uint64_t m = 1ULL << (v & 63);
                if (!(__atomic_load_n(&bs[v >> 6], __ATOMIC_RELAXED) & m))
                    __atomic_fetch_or(&bs[v >> 6], m, __ATOMIC_RELAXED);
            }
        }
    }

    // Keep rows of s whose column k0 is in bs0 (and column k1 in bs1, if given); returns rows removed.
    int64_t probe(Selection &s, int k0, const std::vector<uint64_t> &bs0, int k1 = -1,
                  const std::vector<uint64_t> *bs1 = nullptr) const {
        int64_t removed = 0;
#pragma omp parallel for schedule(static) reduction(+ : removed)
        for (int64_t c = 0; c < nchunks; c++) {
            int32_t cnt = s.count[c];
            if (cnt == 0) continue;
            const int32_t *x = col[k0] + c * VECTOR_SIZE;
            const int32_t *y = k1 >= 0 ? col[k1] + c * VECTOR_SIZE : nullptr;
            uint16_t *sc = s.sel.data() + c * VECTOR_SIZE;
            int32_t m = 0;
            for (int32_t j = 0; j < cnt; j++) {  // branchless in-place shrink
                uint16_t r = sc[j];
                uint32_t v = (uint32_t)x[r];
                bool keep = bs0[v >> 6] >> (v & 63) & 1;
                if (y) {
                    uint32_t w = (uint32_t)y[r];
                    keep &= (*bs1)[w >> 6] >> (w & 63) & 1;
                }
                sc[m] = r;
                m += keep;
            }
            s.count[c] = m;
            removed += cnt - m;
        }
        s.rows -= removed;
        if (removed) s.version++;
        return removed;
    }

    // |f_l JOIN f_r ON f_l.b = f_r.a| = sum_v cnt_l.b(v) * cnt_r.a(v)
    double two_hop(const Selection &l, const Selection &r) const {
        std::vector<uint32_t> cl(domain, 0), cr(domain, 0);
        auto count = [&](std::vector<uint32_t> &cnt, const Selection &s, int k) {
#pragma omp parallel for schedule(static)
            for (int64_t c = 0; c < nchunks; c++)
                for (int32_t j = 0; j < s.count[c]; j++)
                    __atomic_fetch_add(&cnt[col[k][c * VECTOR_SIZE + s.sel[c * VECTOR_SIZE + j]]], 1u,
                                       __ATOMIC_RELAXED);
        };
        count(cl, l, 1);
        count(cr, r, 0);
        double w = 0;
#pragma omp parallel for schedule(static) reduction(+ : w)
        for (int64_t v = 0; v < domain; v++) w += (double)cl[v] * cr[v];
        return w;
    }

    int64_t compact(const Selection &s, int32_t *oa, int32_t *ob) const {
        std::vector<int64_t> off(nchunks + 1, 0);
        for (int64_t c = 0; c < nchunks; c++) off[c + 1] = off[c] + s.count[c];
#pragma omp parallel for schedule(static)
        for (int64_t c = 0; c < nchunks; c++) {
            const uint16_t *sc = s.sel.data() + c * VECTOR_SIZE;
            for (int32_t j = 0; j < s.count[c]; j++) {
                oa[off[c] + j] = col[0][c * VECTOR_SIZE + sc[j]];
                ob[off[c] + j] = col[1][c * VECTOR_SIZE + sc[j]];
            }
        }
        return off[nchunks];
    }
};

}  // namespace

extern "C" {

// a, b: n edges with keys in [0, domain). k: number of copies in the query chain f_1 .. f_k (joined
// f_i.b = f_{i+1}.a, plus f_k.b = f_1.a if cyclic). shared != 0 uses one selection for all copies (cyclic
// self-joins only). out_a/out_b: (shared ? 1 : k) arrays of n, out_rows gets their lengths. Per-round stats go
// to st_rows (rows after), st_secs (round time, incl. computing W only when the rule needs it), st_w (W after)
// for up to max_rounds rounds; w0 gets W before round 1. Returns the number of rounds run.
int afr_reduce(const int32_t *a, const int32_t *b, int64_t n, int64_t domain, int k, int cyclic, int shared,
               int rule, int max_rounds, double tau, double c_j, double c_w, int threads, int32_t **out_a,
               int32_t **out_b, int64_t *out_rows, int64_t *st_rows, double *st_secs, double *st_w, double *w0,
               double *compact_secs) {
    omp_set_num_threads(threads);
    Ctx ctx{{a, b}, n, (n + VECTOR_SIZE - 1) / VECTOR_SIZE, domain, std::vector<uint64_t>((domain >> 6) + 1)};
    std::vector<uint64_t> bits2((domain >> 6) + 1);
    int ncopies = shared ? 1 : k;
    std::vector<Selection> f(ncopies, ctx.fresh());
    auto total = [&] { int64_t t = 0; for (auto &s : f) t += s.rows; return t; };
    auto W = [&] { return shared ? ctx.two_hop(f[0], f[0]) : ctx.two_hop(f[0], f[1]); };
    bool need_w = rule == COST || rule == WORK || rule == BUDGET;
    double spent = 0;

    // semi-join edges: probe (copy, col) against build (copy, col), with the build version last seen
    struct SJ { int pc, pk, bc, bk; int64_t seen; };
    std::vector<SJ> sweep;
    if (!shared) {
        int ne = cyclic ? k : k - 1;
        for (int i = 0; i < ne; i++) sweep.push_back({(i + 1) % k, 0, i, 1, -1});  // forward
        for (int i = ne - 1; i >= 0; i--) sweep.push_back({i, 1, (i + 1) % k, 0, -1});  // backward
    }

    double w_prev = W();
    *w0 = w_prev;
    int64_t n_prev = total();
    int t = 0;
    while (t < max_rounds) {
        double t0 = now();
        int64_t removed = 0;
        if (shared) {
            ctx.build(ctx.bits, f[0], 1);  // surviving b's: a must be one (has an incoming edge)
            ctx.build(bits2, f[0], 0);     // surviving a's: b must be one (has an outgoing edge)
            removed = ctx.probe(f[0], 0, ctx.bits, 1, &bits2);
        } else {
            for (auto &e : sweep) {
                if (e.seen == f[e.bc].version) continue;  // build copy unchanged since this semi-join last ran
                e.seen = f[e.bc].version;
                ctx.build(ctx.bits, f[e.bc], e.bk);
                removed += ctx.probe(f[e.pc], e.pk, ctx.bits);
            }
        }
        double secs_reduce = now() - t0;
        double w = W();
        double secs = need_w ? now() - t0 : secs_reduce;
        int64_t n_now = total();
        st_rows[t] = n_now;
        st_secs[t] = secs;
        st_w[t] = w;
        t++;
        if (removed == 0) break;
        double rho = (double)removed / n_prev;
        bool stop = false;
        if (rule == RHO) stop = rho < tau;
        if (rule == COST) {
            double c_s = secs_reduce / (2.0 * n_prev);
            double alpha = n_now ? w / n_now : 0;
            stop = rho < 2 * c_s / (c_j * alpha);
        }
        if (rule == WORK) stop = c_w * (w_prev - w) < secs;
        spent += secs;
        if (rule == BUDGET) stop = spent > tau * c_w * w;
        w_prev = w;
        n_prev = n_now;
        if (stop) break;
    }

    double t0 = now();
    for (int i = 0; i < ncopies; i++) out_rows[i] = ctx.compact(f[i], out_a[i], out_b[i]);
    *compact_secs = now() - t0;
    return t;
}
}
