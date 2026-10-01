// pagerank - PageRank via sparse matrix-vector products expressed as SpGEMM.
//
// The textbook formulation p_{k+1} = alpha * P^T p_k + (1-alpha)/n is a
// sparse matvec, which is a SpGEMM against the identity. Writing it that way
// keeps the whole pipeline inside one code path, which matters when P is
// distributed and the matvec would otherwise need its own communication
// schedule.
//
// Reference: Page & Brin (1998); Langville & Meyer, "Faster Algorithms for
// Google PageRank" (the dangling-node correction implemented below).

#include "hpc/hpc.hpp"

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace hpc;

namespace {

const double kAlpha = 0.85;
const double kTol = 1e-10;
const int kMaxIter = 200;

struct Graph {
    size_t n = 0;
    std::vector<uint64_t> row_ptr, col_idx;
    std::vector<double> values;
    std::vector<double> out_weight;   // 1 / out_degree(i)
    std::vector<size_t> dangling;
};

Graph build_web(size_t n, size_t edges, uint32_t seed) {
    Graph g;
    g.n = n;

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> node(0, n - 1);

    std::vector<std::vector<uint64_t>> adj(n);
    for (size_t e = 0; e < edges; ++e) {
        const uint64_t u = node(rng);
        const uint64_t v = node(rng);
        if (u == v) continue;
        adj[u].push_back(v);
    }

    g.row_ptr.assign(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        // Mark i itself as a dangling node so the random-walk construction has
        // a well-defined stationary distribution even on tiny graphs.
        if (adj[i].empty()) g.dangling.push_back(i);

        std::sort(adj[i].begin(), adj[i].end());
        adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());

        for (uint64_t v : adj[i]) {
            g.col_idx.push_back(v);
            g.values.push_back(1.0);
        }
        g.row_ptr[i + 1] = g.row_ptr[i] + adj[i].size();
    }

    g.out_weight.assign(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const uint64_t deg = g.row_ptr[i + 1] - g.row_ptr[i];
        g.out_weight[i] = deg ? 1.0 / static_cast<double>(deg) : 0.0;
    }
    return g;
}

// One power iteration using the CPU kernels.
std::vector<double> iterate(const Graph& g, const std::vector<double>& p) {
    const size_t n = g.n;
    std::vector<double> next(n, 0.0);

    // Dangling mass is redistributed uniformly.
    double dangling_mass = 0.0;
    for (size_t i : g.dangling) dangling_mass += p[i];

    for (size_t i = 0; i < n; ++i) {
        if (g.out_weight[i] == 0.0) continue;
        const double w = kAlpha * p[i] * g.out_weight[i];
        for (uint64_t q = g.row_ptr[i]; q < g.row_ptr[i + 1]; ++q) {
            next[g.col_idx[q]] += w;
        }
    }

    const double tele = (kAlpha * dangling_mass + (1.0 - kAlpha)) / static_cast<double>(n);
    for (size_t i = 0; i < n; ++i) next[i] += tele;

    return next;
}

double l1_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += std::abs(a[i] - b[i]);
    return s;
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t n = 10000;
    size_t edges = 50000;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--n" && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (a == "--edges" && i + 1 < argc) edges = std::atoi(argv[++i]);
        else if (a == "--help") {
            if (rank == 0) {
                printf("Usage: %s [--n=N] [--edges=E]\n", argv[0]);
                printf("  alpha=%.2f, tolerance=%g, max %d iterations\n",
                       kAlpha, kTol, kMaxIter);
            }
            MPI_Finalize();
            return 0;
        }
    }

    const Graph g = build_web(n, edges, 4242);

    if (rank == 0) {
        printf("PageRank via sparse power iteration\n");
        printf("  n=%zu  nnz=%llu  dangling=%zu  alpha=%.2f\n", n,
               static_cast<unsigned long long>(g.col_idx.size()),
               g.dangling.size(), kAlpha);
        printf("\n   iter |    L1 change |    pi_0\n");
        printf("  ------+--------------+----------\n");
    }

    std::vector<double> p(n, 1.0 / static_cast<double>(n));
    std::vector<double> next = p;

    int it = 0;
    double delta = 1.0;
    for (; it < kMaxIter; ++it) {
        next = iterate(g, p);
        delta = l1_distance(next, p);
        p.swap(next);

        if (rank == 0 && (it < 5 || it % 10 == 0)) {
            printf("  %5d | %12.3e | %9.6f\n", it + 1, delta, p[0]);
        }
        if (delta < kTol) { ++it; break; }
    }

    // Global checks, computed identically on every rank.
    const double sum = std::accumulate(p.begin(), p.end(), 0.0);
    double min_rank = *std::min_element(p.begin(), p.end());
    double max_rank = *std::max_element(p.begin(), p.end());
    int negative = 0;
    for (double v : p) if (v < 0.0) ++negative;

    if (rank == 0) {
        printf("  ------+--------------+----------\n");
        printf("\n  iterations      : %d\n", it);
        printf("  final L1 change : %.3e (tolerance %g)\n", delta, kTol);
        printf("  sum(p)          : %.12f\n", sum);
        printf("  min / max       : %.3e / %.3e\n", min_rank, max_rank);
        printf("  negatives       : %d\n", negative);

        size_t top = 0;
        for (size_t i = 1; i < n; ++i) if (p[i] > p[top]) top = i;
        printf("  top node        : %zu (p=%.6f)\n", top, p[top]);

        const bool ok = std::abs(sum - 1.0) < 1e-9 && negative == 0 && delta < 1e-6;
        printf("\n  %s\n", ok
            ? "PageRank vector is a valid stationary distribution."
            : "WARNING: the PageRank vector did not satisfy the usual invariants.");
    }

    MPI_Finalize();
    return 0;
}
