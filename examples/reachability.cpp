// reachability - transitive closure of a directed graph, computed as a power
// series of the adjacency matrix over the Boolean semiring (OR for add, AND
// for multiply).
//
//   W^k[i][j] != 0  <=>  there is a walk of length exactly k from i to j
//   (I + W)^(n-1)   is the transitive closure
//
// Reference: Boolean matrix powers as used in classic graph analysis; the
// semiring formulation follows Golumbic & Stockmeyer, "The Algebraic
// Complexity of Graph Matching".

#include "hpc/hpc.hpp"

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <queue>

using namespace hpc;

namespace {

struct Graph {
    std::unique_ptr<Matrix<double, MatrixFormat::COO>> m;
    std::vector<uint64_t> row_ptr;
    size_t n = 0;
};

Graph make_random_digraph(size_t n, double out_degree, uint32_t seed) {
    Graph g;
    g.n = n;
    g.m = std::make_unique<Matrix<double, MatrixFormat::COO>>(n, n);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> col(0, n - 1);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::vector<std::vector<uint64_t>> adj(n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t e = 0; e < static_cast<size_t>(out_degree); ++e) {
            const uint64_t j = col(rng);
            adj[i].push_back(j);
            g.m->coo_rows_.push_back(i);
            g.m->coo_cols_.push_back(j);
            g.m->coo_values_.push_back(1.0);
        }
    }

    // Sort and unique each row, then rebuild the COO arrays.
    g.m->coo_rows_.clear();
    g.m->coo_cols_.clear();
    g.m->coo_values_.clear();
    for (size_t i = 0; i < n; ++i) {
        std::sort(adj[i].begin(), adj[i].end());
        adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
        for (uint64_t j : adj[i]) {
            g.m->coo_rows_.push_back(i);
            g.m->coo_cols_.push_back(j);
            g.m->coo_values_.push_back(1.0);
        }
    }
    g.m->nnz_ = g.m->coo_rows_.size();

    g.row_ptr.assign(n + 1, 0);
    for (size_t i = 0; i < n; ++i) g.row_ptr[i + 1] = g.row_ptr[i] + adj[i].size();
    return g;
}

struct Bools {
    std::vector<uint64_t> row_ptr;
    std::vector<uint64_t> col_idx;
    std::vector<int32_t> values;
    uint64_t nnz() const { return row_ptr.empty() ? 0 : row_ptr.back(); }
    size_t rows() const { return row_ptr.size() - 1; }
};

Bools bool_product(const Bools& A, const Bools& B, LorLandInt32 sem) {
    Bools C;
    C.row_ptr.assign(A.rows() + 1, 0);

    uint64_t max_row_b = 0;
    for (size_t i = 0; i < B.rows(); ++i) {
        max_row_b = std::max<uint64_t>(max_row_b, B.row_ptr[i + 1] - B.row_ptr[i]);
    }
    const uint64_t cap = (A.nnz() + 1) * (max_row_b + 1) + A.rows() + 16;

    C.col_idx.assign(cap, 0);
    C.values.assign(cap, 0);

    cpu::gustavson_spgemm<int64_t, int32_t, LorLandInt32>(
        A.row_ptr.data(), A.col_idx.data(), A.values.data(),
        B.row_ptr.data(), B.col_idx.data(), B.values.data(),
        C.row_ptr.data(), C.col_idx.data(), C.values.data(),
        static_cast<int64_t>(A.rows()), static_cast<int64_t>(B.rows()),
        static_cast<int64_t>(A.rows()), sem);

    C.col_idx.resize(C.nnz());
    C.values.resize(C.nnz());
    return C;
}

// Reference transitive closure by BFS from every vertex.
std::vector<std::vector<char>> bfs_closure(const std::vector<std::vector<uint64_t>>& adj,
                                           size_t n) {
    std::vector<std::vector<char>> reach(n, std::vector<char>(n, 0));
    for (size_t s = 0; s < n; ++s) {
        std::vector<char> seen(n, 0);
        std::queue<size_t> q;
        q.push(s);
        seen[s] = 1;
        while (!q.empty()) {
            const size_t u = q.front();
            q.pop();
            for (uint64_t v : adj[u]) {
                if (!seen[v]) { seen[v] = 1; q.push(static_cast<size_t>(v)); }
            }
        }
        reach[s] = seen;
    }
    return reach;
}

std::vector<std::vector<uint64_t>> to_adj(const Graph& g) {
    std::vector<std::vector<uint64_t>> adj(g.n);
    for (size_t i = 0; i < g.n; ++i) {
        for (uint64_t p = g.row_ptr[i]; p < g.row_ptr[i + 1]; ++p) {
            adj[i].push_back(g.m->coo_cols_[p]);
        }
    }
    return adj;
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t n = 200;
    double out_degree = 5.0;
    int max_pow = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--n" && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (a == "--degree" && i + 1 < argc) out_degree = std::atof(argv[++i]);
        else if (a == "--max-pow" && i + 1 < argc) max_pow = std::atoi(argv[++i]);
        else if (a == "--help") {
            if (rank == 0) {
                printf("Usage: %s [--n=N] [--degree=D] [--max-pow=K]\n", argv[0]);
                printf("  Computes the transitive closure by Boolean SpGEMM powers\n"
                       "  and cross-checks it against BFS.\n");
            }
            MPI_Finalize();
            return 0;
        }
    }

    const Graph g = make_random_digraph(n, out_degree, 2026);

    if (rank == 0) {
        printf("Transitive closure via Boolean SpGEMM\n");
        printf("  n=%zu  average out-degree=%.1f  nnz(W)=%llu\n", n, out_degree,
               static_cast<unsigned long long>(g.m->nnz_));
        printf("  (OR is the additive operator, AND the multiplicative one)\n\n");
    }

    LorLandInt32 sem;

    Bools cur;
    cur.row_ptr = g.row_ptr;
    cur.col_idx = g.m->coo_cols_;
    cur.values.assign(g.m->coo_values_.size(), 1);
    cur.values.assign(cur.col_idx.size(), 1);

    const int limit = max_pow > 0 ? max_pow : static_cast<int>(n) - 1;
    bool converged = false;

    for (int k = 1; k <= limit; ++k) {
        const Bools next = bool_product(cur, cur, sem);

        // Stop once the support stops growing.
        if (next.nnz() == cur.nnz() && k > 1) {
            converged = true;
            if (rank == 0) {
                printf("  W^%-4d nnz=%-9llu (stable, stopping)\n", k,
                       static_cast<unsigned long long>(next.nnz()));
            }
            cur = next;
            break;
        }

        if (rank == 0) {
            printf("  W^%-4d nnz=%llu\n", k, static_cast<unsigned long long>(next.nnz()));
        }
        cur = next;
    }

    // Cross-check against BFS.
    const auto adj = to_adj(g);
    const auto ref = bfs_closure(adj, n);

    if (rank == 0) {
        printf("\n  reached by W^k but not by BFS: ");
        int fp = 0;
        for (size_t i = 0; i < n; ++i) {
            for (uint64_t p = cur.row_ptr[i]; p < cur.row_ptr[i + 1]; ++p) {
                if (!ref[i][cur.col_idx[p]]) ++fp;
            }
        }
        printf("%d\n", fp);

        printf("  reached by BFS but not by W^k: ");
        int fn = 0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (!ref[i][j]) continue;
                bool found = false;
                for (uint64_t p = cur.row_ptr[i]; p < cur.row_ptr[i + 1]; ++p) {
                    if (cur.col_idx[p] == j) { found = true; break; }
                }
                if (!found) ++fn;
            }
        }
        printf("%d\n", fn);

        // Summarise strongly-connected structure via reachability of node 0.
        size_t reach0 = 0;
        for (size_t j = 0; j < n; ++j) reach0 += ref[0][j] ? 1 : 0;
        printf("  nodes reachable from 0: %zu / %zu\n", reach0, n);
        printf("  %s\n", (fp == 0 && fn == 0)
            ? "Boolean closure matches BFS exactly."
            : "MISMATCH between Boolean closure and BFS.");

        if (!converged) {
            printf("  (did not converge within %d powers; raise --max-pow)\n", limit);
        }
    }

    MPI_Finalize();
    return 0;
}
