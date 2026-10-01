// shortest_path - all-pairs shortest paths as min-plus powers of the adjacency
// matrix. This is the classic demonstration of why the semiring abstraction
// matters: no kernel change is needed, only a different (add, multiply) pair.
//
//   W^k[i][j] = shortest distance from i to j using exactly k edges
//   W^n      = the distance matrix
//
// Reference: Zwick (2002), "All pairs shortest paths via multiple matrix
// multiplication"; Cormen et al. (2009), "Faster algorithms for shortest paths".

#include "hpc/hpc.hpp"

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <limits>

using namespace hpc;

namespace {

const double INF = std::numeric_limits<double>::infinity();

struct Graph {
    std::unique_ptr<Matrix<double, MatrixFormat::COO>> m;
    std::vector<uint64_t> row_ptr;
};

Graph make_ring(size_t n) {
    Graph g;
    g.m = std::make_unique<Matrix<double, MatrixFormat::COO>>(n, n);

    // Undirected ring: each vertex has edges to its two neighbours.
    for (size_t i = 0; i < n; ++i) {
        g.m->coo_rows_.push_back(i);
        g.m->coo_cols_.push_back((i + 1) % n);
        g.m->coo_values_.push_back(1.0);

        g.m->coo_rows_.push_back(i);
        g.m->coo_cols_.push_back((i + n - 1) % n);
        g.m->coo_values_.push_back(1.0);
    }
    g.m->nnz_ = g.m->coo_rows_.size();

    g.row_ptr.assign(n + 1, 0);
    for (size_t i = 0; i < n; ++i) g.row_ptr[i + 1] = g.row_ptr[i] + 2;
    return g;
}

Graph make_grid(size_t r, size_t c) {
    Graph g;
    const size_t n = r * c;
    g.m = std::make_unique<Matrix<double, MatrixFormat::COO>>(n, n);

    auto add = [&](size_t u, size_t v, double w) {
        g.m->coo_rows_.push_back(u);
        g.m->coo_cols_.push_back(v);
        g.m->coo_values_.push_back(w);
    };

    for (size_t i = 0; i < r; ++i) {
        for (size_t j = 0; j < c; ++j) {
            const size_t u = i * c + j;
            if (i + 1 < r) { add(u, u + c, 1.0); add(u + c, u, 1.0); }
            if (j + 1 < c) { add(u, u + 1, 1.0); add(u + 1, u, 1.0); }
        }
    }
    g.m->nnz_ = g.m->coo_rows_.size();

    g.row_ptr.assign(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        g.row_ptr[i + 1] = g.row_ptr[i] + 2;
    }
    // Fix the boundary rows whose degree differs.
    for (size_t i = 0; i < n; ++i) {
        uint64_t deg = 0;
        for (size_t p = g.m->coo_rows_.size(); p-- > 0;) {
            if (g.m->coo_rows_[p] != i) break;
            ++deg;
        }
        g.row_ptr[i + 1] = g.row_ptr[i] + deg;
    }
    return g;
}

// Read a CSR triple from disk.
Graph load_mtx(const char* path) {
    Graph g;
    MatrixBase* raw = nullptr;
    if (!read_matrix_market(path, &raw)) {
        fprintf(stderr, "could not read %s\n", path);
        std::exit(1);
    }
    const auto* m = dynamic_cast<const Matrix<double, MatrixFormat::COO>*>(raw);
    g.m = std::make_unique<Matrix<double, MatrixFormat::COO>>(*m);
    delete raw;

    const size_t n = g.m->num_rows();
    g.row_ptr.assign(n + 1, 0);
    for (size_t i = 0; i < g.m->nnz_; ++i) ++g.row_ptr[g.m->coo_rows_[i] + 1];
    for (size_t i = 0; i < n; ++i) g.row_ptr[i + 1] += g.row_ptr[i];
    return g;
}

struct Result {
    std::vector<uint64_t> row_ptr;
    std::vector<uint64_t> col_idx;
    std::vector<double> values;
    uint64_t nnz() const { return row_ptr.empty() ? 0 : row_ptr.back(); }
};

// One min-plus SpGEMM: C = A (*) B where (*) is (min, +).
Result minplus_product(const Result& A, const Result& B, MinPlusDouble sem) {
    Result C;
    const size_t rows = A.row_ptr.size() - 1;
    C.row_ptr.assign(rows + 1, 0);

    // Upper bound on the nnz of the product.
    uint64_t max_row_b = 0;
    for (size_t i = 0; i + 1 < B.row_ptr.size(); ++i) {
        max_row_b = std::max<uint64_t>(max_row_b, B.row_ptr[i + 1] - B.row_ptr[i]);
    }
    const uint64_t cap = (A.nnz() + 1) * (max_row_b + 1) + rows + 16;

    C.col_idx.assign(cap, 0);
    C.values.assign(cap, 0.0);

    cpu::merge_path_spgemm<int64_t, double, MinPlusDouble>(
        A.row_ptr.data(), A.col_idx.data(), A.values.data(),
        B.row_ptr.data(), B.col_idx.data(), B.values.data(),
        C.row_ptr.data(), C.col_idx.data(), C.values.data(),
        static_cast<int64_t>(rows), static_cast<int64_t>(B.row_ptr.size() - 1),
        static_cast<int64_t>(A.row_ptr.size() - 1), sem);

    C.col_idx.resize(C.nnz());
    C.values.resize(C.nnz());
    return C;
}

double lookup(const Result& r, size_t i, size_t j) {
    for (uint64_t p = r.row_ptr[i]; p < r.row_ptr[i + 1]; ++p) {
        if (r.col_idx[p] == j) return r.values[p];
    }
    return INF;
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string kind = "ring";
    size_t n = 12;
    const char* file = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--graph" && i + 1 < argc) kind = "file";
        else if (std::string(argv[i]) == "--file" && i + 1 < argc) file = argv[++i];
        else if (std::string(argv[i]) == "--n" && i + 1 < argc) n = std::atoi(argv[++i]);
    }

    Graph g;
    if (kind == "file" && file) {
        g = load_mtx(file);
        n = g.m->num_rows();
    } else if (kind == "grid") {
        size_t r = static_cast<size_t>(std::sqrt(static_cast<double>(n)));
        if (r * r < n) ++r;
        g = make_grid(r, n / r);
        n = g.m->num_rows();
    } else {
        g = make_ring(n);
    }

    if (rank == 0) {
        printf("All-pairs shortest paths via min-plus SpGEMM\n");
        printf("  graph: %s, n=%zu, nnz=%llu\n", kind.c_str(), n,
               static_cast<unsigned long long>(g.m->nnz_));
        printf("  (min is the additive operator, + the multiplicative one)\n\n");
    }

    MinPlusDouble sem;

    Result cur;
    cur.row_ptr = g.row_ptr;
    cur.col_idx = g.m->coo_cols_;
    cur.values = g.m->coo_values_;

    // Repeated squaring needs only log2(diameter) products; we instead iterate
    // with W <- W (min) W until the matrix stops changing, which is at most
    // `n` iterations and terminates as soon as the distances stabilise.
    for (size_t iter = 1; iter <= n; ++iter) {
        const Result next = minplus_product(cur, cur, sem);

        const bool changed = (next.nnz() != cur.nnz());
        if (rank == 0) {
            printf("  W^%2zu  nnz=%-8llu  d(0,min(n-1,9))=%s\n",
                   iter + 1, static_cast<unsigned long long>(next.nnz()),
                   std::isfinite(lookup(next, 0, std::min<size_t>(n - 1, 9)))
                       ? std::to_string(lookup(next, 0, std::min<size_t>(n - 1, 9))).c_str()
                       : "inf");
        }

        cur = next;
        if (!changed && iter > 2) break;
    }

    if (rank == 0) {
        printf("\nDistance matrix (first 8 rows, first 8 columns):\n");
        printf("      ");
        for (size_t j = 0; j < 8 && j < n; ++j) printf("%6zu", j);
        printf("\n");
        for (size_t i = 0; i < 8 && i < n; ++i) {
            printf("  %3zu ", i);
            for (size_t j = 0; j < 8 && j < n; ++j) {
                const double d = lookup(cur, i, j);
                if (std::isfinite(d)) printf("%6.0f", d);
                else printf("     .");
            }
            printf("\n");
        }

        // Triangle inequality spot-check: d(i,k) <= d(i,j) + d(j,k)
        int violations = 0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                for (size_t k = 0; k < n; ++k) {
                    const double dik = lookup(cur, i, k);
                    const double dij = lookup(cur, i, j);
                    const double djk = lookup(cur, j, k);
                    if (!std::isfinite(dik) || !std::isfinite(dij) || !std::isfinite(djk)) continue;
                    if (dik > dij + djk + 1e-9) ++violations;
                }
            }
        }
        printf("\n  triangle-inequality violations: %d %s\n", violations,
               violations == 0 ? "(valid distance matrix)" : "(INVALID)");
    }

    MPI_Finalize();
    return 0;
}
