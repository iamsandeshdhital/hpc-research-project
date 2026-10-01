// Property-based (randomised) tests for SpGEMM.
//
// Rather than checking hand-picked inputs, these tests generate hundreds of
// random matrices and assert invariants that must hold for *every* valid
// input. This is where a SpGEMM implementation most often breaks: empty rows,
// single-element matrices, duplicate products, dense extremes, and semirings
// whose "add" is not ordinary addition.

#include "test_framework.hpp"
#include "hpc/hpc.hpp"

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>
#include <tuple>
#include <string>

using namespace hpc;

namespace {

struct Case {
    uint64_t rows, cols, nnz;
};

// Generate a random COO matrix. n may be tiny (including 1) and nnz may be 0.
std::unique_ptr<Matrix<double, MatrixFormat::COO>>
random_matrix(uint64_t n, double density, uint32_t seed) {
    auto m = std::make_unique<Matrix<double, MatrixFormat::COO>>(n, n);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::vector<std::tuple<uint64_t, uint64_t, double>> e;
    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            if (uni(rng) < density) {
                // Nonzero values: keep them away from 0 so that products do not
                // accidentally vanish and get dropped by remove_zeros.
                double v = uni(rng) * 2.0 - 1.0;
                if (std::abs(v) < 0.05) v += 0.5;
                e.emplace_back(i, j, v);
            }
        }
    }

    m->nnz_ = e.size();
    m->coo_rows_.reserve(e.size());
    m->coo_cols_.reserve(e.size());
    m->coo_values_.reserve(e.size());
    for (const auto& t : e) {
        m->coo_rows_.push_back(std::get<0>(t));
        m->coo_cols_.push_back(std::get<1>(t));
        m->coo_values_.push_back(std::get<2>(t));
    }
    return m;
}

// Build CSR from a dense vector-of-vectors.
struct Csr {
    uint64_t rows = 0, cols = 0;
    std::vector<uint64_t> row_ptr;
    std::vector<uint64_t> col_idx;
    std::vector<double> values;
    uint64_t nnz() const { return row_ptr.empty() ? 0 : row_ptr[rows]; }
};

Csr to_csr(const std::vector<std::vector<double>>& M) {
    Csr c;
    c.rows = M.size();
    c.cols = M.empty() ? 0 : M[0].size();
    c.row_ptr.assign(c.rows + 1, 0);
    for (size_t i = 0; i < c.rows; ++i) {
        for (size_t j = 0; j < c.cols; ++j) {
            if (M[i][j] != 0.0) {
                c.col_idx.push_back(j);
                c.values.push_back(M[i][j]);
                ++c.row_ptr[i + 1];
            }
        }
    }
    for (size_t i = 0; i < c.rows; ++i) c.row_ptr[i + 1] += c.row_ptr[i];
    return c;
}

std::vector<std::vector<double>> to_dense(const Csr& c) {
    std::vector<std::vector<double>> M(c.rows, std::vector<double>(c.cols, 0.0));
    for (size_t i = 0; i < c.rows; ++i) {
        for (uint64_t p = c.row_ptr[i]; p < c.row_ptr[i + 1]; ++p) {
            M[i][c.col_idx[p]] = c.values[p];
        }
    }
    return M;
}

// Reference dense product.
std::vector<std::vector<double>> dense_product(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) {
    const size_t m = A.size();
    const size_t k = A.empty() ? 0 : A[0].size();
    const size_t n = B.empty() ? 0 : B[0].size();
    std::vector<std::vector<double>> C(m, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < m; ++i)
        for (size_t p = 0; p < k; ++p) {
            if (A[i][p] == 0.0) continue;
            for (size_t j = 0; j < n; ++j) {
                if (B[p][j] == 0.0) continue;
                C[i][j] += A[i][p] * B[p][j];
            }
        }
    return C;
}

bool same(const std::vector<std::vector<double>>& X,
          const std::vector<std::vector<double>>& Y, double tol) {
    if (X.size() != Y.size()) return false;
    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i].size() != Y[i].size()) return false;
        for (size_t j = 0; j < X[i].size(); ++j) {
            if (std::abs(X[i][j] - Y[i][j]) > tol) return false;
        }
    }
    return true;
}

// Upper bound on the nnz of the product, used to size scratch buffers.
uint64_t product_capacity(const Csr& a, const Csr& b) {
    uint64_t max_row_b = 0;
    for (size_t i = 0; i < b.rows; ++i) {
        max_row_b = std::max<uint64_t>(max_row_b, b.row_ptr[i + 1] - b.row_ptr[i]);
    }
    return (a.nnz() + 1) * (max_row_b + 1) + b.rows + 16;
}

} // namespace

// ---------------------------------------------------------------------------
// Core property: C == A * B for randomised inputs
// ---------------------------------------------------------------------------

TEST(Property, MatchesDenseProductAcrossManyShapes) {
    std::mt19937_64 shape_rng(20260101);
    std::uniform_int_distribution<int> size_dist(1, 40);
    std::uniform_real_distribution<double> dens_dist(0.0, 0.6);

    PlusTimesDouble sem;
    int cases = 0;

    for (int trial = 0; trial < 200; ++trial) {
        const uint64_t n = static_cast<uint64_t>(size_dist(shape_rng));
        const double da = dens_dist(shape_rng);
        const double db = dens_dist(shape_rng);

        auto Am = random_matrix(n, da, static_cast<uint32_t>(trial * 7 + 1));
        auto Bm = random_matrix(n, db, static_cast<uint32_t>(trial * 13 + 5));

        const auto a = to_csr(to_dense(Am));
        const auto b = to_csr(to_dense(Bm));
        const auto ref = dense_product(to_dense(Am), to_dense(Bm));

        Csr c;
        c.rows = n; c.cols = n;
        c.row_ptr.assign(n + 1, 0);
        const uint64_t cap = product_capacity(a, b);
        c.col_idx.assign(cap, 0);
        c.values.assign(cap, 0.0);

        cpu::gustavson_spgemm<int64_t, double, PlusTimesDouble>(
            a.row_ptr.data(), a.col_idx.data(), a.values.data(),
            b.row_ptr.data(), b.col_idx.data(), b.values.data(),
            c.row_ptr.data(), c.col_idx.data(), c.values.data(),
            static_cast<int64_t>(n), static_cast<int64_t>(n),
            static_cast<int64_t>(n), sem);

        c.col_idx.resize(c.nnz());
        c.values.resize(c.nnz());

        const auto got = to_dense(c);
        const double scale = std::max(1.0, static_cast<double>(n));
        if (!same(got, ref, 1e-9 * scale)) {
            ++::hptest::failure_count();
            printf("    FAIL at trial %d: n=%llu da=%.3f db=%.3f\n", trial,
                   static_cast<unsigned long long>(n), da, db);
            break;
        }
        ++cases;
    }

    EXPECT_EQ(cases, 200);
}

// ---------------------------------------------------------------------------
// Invariants that must hold for every valid output
// ---------------------------------------------------------------------------

TEST(Property, RowPointersAreMonotoneAndTerminateAtNnz) {
    std::mt19937_64 rng(777);
    std::uniform_int_distribution<int> size_dist(1, 30);
    std::uniform_real_distribution<double> dens_dist(0.0, 0.8);

    PlusTimesDouble sem;
    int checked = 0;

    for (int trial = 0; trial < 100; ++trial) {
        const uint64_t n = static_cast<uint64_t>(size_dist(rng));
        auto Am = random_matrix(n, dens_dist(rng), static_cast<uint32_t>(trial));
        auto Bm = random_matrix(n, dens_dist(rng), static_cast<uint32_t>(trial + 500));

        const auto a = to_csr(to_dense(Am));
        const auto b = to_csr(to_dense(Bm));

        Csr c;
        c.rows = n; c.cols = n;
        c.row_ptr.assign(n + 1, 0);
        c.col_idx.assign(product_capacity(a, b), 0);
        c.values.assign(c.col_idx.size(), 0.0);

        cpu::gustavson_spgemm<int64_t, double, PlusTimesDouble>(
            a.row_ptr.data(), a.col_idx.data(), a.values.data(),
            b.row_ptr.data(), b.col_idx.data(), b.values.data(),
            c.row_ptr.data(), c.col_idx.data(), c.values.data(),
            static_cast<int64_t>(n), static_cast<int64_t>(n),
            static_cast<int64_t>(n), sem);

        if (c.row_ptr[0] != 0) { ++::hptest::failure_count(); break; }
        for (uint64_t i = 0; i < n; ++i) {
            if (c.row_ptr[i + 1] < c.row_ptr[i]) { ++::hptest::failure_count(); break; }
        }
        if (c.row_ptr[n] != c.col_idx.size() && c.row_ptr[n] > c.col_idx.size()) {
            ++::hptest::failure_count();
            break;
        }
        ++checked;
    }

    EXPECT_EQ(checked, 100);
}

TEST(Property, ColumnIndicesAreSortedAndInRange) {
    std::mt19937_64 rng(31337);
    std::uniform_int_distribution<int> size_dist(1, 35);
    std::uniform_real_distribution<double> dens_dist(0.0, 0.7);

    PlusTimesDouble sem;
    int checked = 0;

    for (int trial = 0; trial < 100; ++trial) {
        const uint64_t n = static_cast<uint64_t>(size_dist(rng));
        auto Am = random_matrix(n, dens_dist(rng), static_cast<uint32_t>(trial * 3));
        auto Bm = random_matrix(n, dens_dist(rng), static_cast<uint32_t>(trial * 3 + 9));

        const auto a = to_csr(to_dense(Am));
        const auto b = to_csr(to_dense(Bm));

        Csr c;
        c.rows = n; c.cols = n;
        c.row_ptr.assign(n + 1, 0);
        c.col_idx.assign(product_capacity(a, b), 0);
        c.values.assign(c.col_idx.size(), 0.0);

        cpu::gustavson_spgemm<int64_t, double, PlusTimesDouble>(
            a.row_ptr.data(), a.col_idx.data(), a.values.data(),
            b.row_ptr.data(), b.col_idx.data(), b.values.data(),
            c.row_ptr.data(), c.col_idx.data(), c.values.data(),
            static_cast<int64_t>(n), static_cast<int64_t>(n),
            static_cast<int64_t>(n), sem);

        bool ok = true;
        for (uint64_t i = 0; i < n && ok; ++i) {
            for (uint64_t p = c.row_ptr[i] + 1; p < c.row_ptr[i + 1]; ++p) {
                if (c.col_idx[p] <= c.col_idx[p - 1] || c.col_idx[p] >= n) {
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) { ++::hptest::failure_count(); break; }
        ++checked;
    }

    EXPECT_EQ(checked, 100);
}

TEST(Property, NoDuplicateColumnsWithinARow) {
    std::mt19937_64 rng(4242);
    std::uniform_real_distribution<double> dens_dist(0.05, 0.5);
    PlusTimesDouble sem;

    for (int trial = 0; trial < 60; ++trial) {
        const uint64_t n = 25;
        auto Am = random_matrix(n, dens_dist(rng), static_cast<uint32_t>(trial * 11));
        auto Bm = random_matrix(n, dens_dist(rng), static_cast<uint32_t>(trial * 11 + 4));

        const auto a = to_csr(to_dense(Am));
        const auto b = to_csr(to_dense(Bm));

        Csr c;
        c.rows = n; c.cols = n;
        c.row_ptr.assign(n + 1, 0);
        c.col_idx.assign(product_capacity(a, b), 0);
        c.values.assign(c.col_idx.size(), 0.0);

        cpu::gustavson_spgemm<int64_t, double, PlusTimesDouble>(
            a.row_ptr.data(), a.col_idx.data(), a.values.data(),
            b.row_ptr.data(), b.col_idx.data(), b.values.data(),
            c.row_ptr.data(), c.col_idx.data(), c.values.data(),
            static_cast<int64_t>(n), static_cast<int64_t>(n),
            static_cast<int64_t>(n), sem);

        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t p = c.row_ptr[i] + 1; p < c.row_ptr[i + 1]; ++p) {
                if (c.col_idx[p] == c.col_idx[p - 1]) {
                    ++::hptest::failure_count();
                    i = n;
                    break;
                }
            }
        }
    }
    EXPECT_TRUE(true);   // failures are counted above
}

// ---------------------------------------------------------------------------
// Algebraic properties
// ---------------------------------------------------------------------------

TEST(Property, MultiplicationIsAssociative) {
    std::mt19937_64 rng(9001);
    std::uniform_real_distribution<double> dens_dist(0.05, 0.3);
    PlusTimesDouble sem;

    auto mul = [](const Csr& x, const Csr& y, Csr& out) {
        out.rows = y.cols;
        out.cols = y.cols;
        out.row_ptr.assign(y.rows + 1, 0);
        out.col_idx.assign(product_capacity(x, y), 0);
        out.values.assign(out.col_idx.size(), 0.0);
        cpu::gustavson_spgemm<int64_t, double, PlusTimesDouble>(
            x.row_ptr.data(), x.col_idx.data(), x.values.data(),
            y.row_ptr.data(), y.col_idx.data(), y.values.data(),
            out.row_ptr.data(), out.col_idx.data(), out.values.data(),
            static_cast<int64_t>(x.rows), static_cast<int64_t>(y.cols),
            static_cast<int64_t>(x.cols), PlusTimesDouble{});
        out.col_idx.resize(out.nnz());
        out.values.resize(out.nnz());
    };

    int checked = 0;
    for (int trial = 0; trial < 30; ++trial) {
        const uint64_t n = 16;
        auto A = random_matrix(n, dens_dist(rng), static_cast<uint32_t>(trial * 3));
        auto B = random_matrix(n, dens_dist(rng), static_cast<uint32_t>(trial * 3 + 1));
        auto D = random_matrix(n, dens_dist(rng), static_cast<uint32_t>(trial * 3 + 2));

        const auto a = to_csr(to_dense(A));
        const auto b = to_csr(to_dense(B));
        const auto d = to_csr(to_dense(D));

        Csr ab, abd, bd, a_bd;
        mul(a, b, ab);
        mul(ab, d, abd);
        mul(b, d, bd);
        mul(a, bd, a_bd);

        if (!same(to_dense(abd), to_dense(a_bd), 1e-8)) {
            ++::hptest::failure_count();
            break;
        }
        ++checked;
    }
    EXPECT_EQ(checked, 30);
}

TEST(Property, MultiplicationByIdentityIsIdentity) {
    std::mt19937_64 rng(1234);
    std::uniform_real_distribution<double> dens_dist(0.0, 0.4);
    PlusTimesDouble sem;

    int checked = 0;
    for (int trial = 0; trial < 40; ++trial) {
        const uint64_t n = static_cast<uint64_t>(1 + (rng() % 30));
        auto A = random_matrix(n, dens_dist(rng), static_cast<uint32_t>(trial * 5));

        auto I = std::make_unique<Matrix<double, MatrixFormat::COO>>(n, n);
        I->nnz_ = n;
        I->coo_rows_.resize(n);
        I->coo_cols_.resize(n);
        I->coo_values_.resize(n);
        for (uint64_t i = 0; i < n; ++i) {
            I->coo_rows_[i] = i;
            I->coo_cols_[i] = i;
            I->coo_values_[i] = 1.0;
        }

        const auto a = to_csr(to_dense(A));
        const auto i = to_csr(to_dense(I));

        Csr c;
        c.rows = n; c.cols = n;
        c.row_ptr.assign(n + 1, 0);
        c.col_idx.assign(product_capacity(a, i), 0);
        c.values.assign(c.col_idx.size(), 0.0);

        cpu::gustavson_spgemm<int64_t, double, PlusTimesDouble>(
            a.row_ptr.data(), a.col_idx.data(), a.values.data(),
            i.row_ptr.data(), i.col_idx.data(), i.values.data(),
            c.row_ptr.data(), c.col_idx.data(), c.values.data(),
            static_cast<int64_t>(n), static_cast<int64_t>(n),
            static_cast<int64_t>(n), sem);
        c.col_idx.resize(c.nnz());
        c.values.resize(c.nnz());

        if (!same(to_dense(c), to_dense(a), 1e-12)) {
            ++::hptest::failure_count();
            break;
        }
        ++checked;
    }
    EXPECT_EQ(checked, 40);
}

TEST(Property, MultiplicationByZeroIsZero) {
    std::mt19937_64 rng(555);
    std::uniform_real_distribution<double> dens_dist(0.0, 0.5);
    PlusTimesDouble sem;

    int checked = 0;
    for (int trial = 0; trial < 30; ++trial) {
        const uint64_t n = static_cast<uint64_t>(1 + (rng() % 25));
        auto A = random_matrix(n, dens_dist(rng), static_cast<uint32_t>(trial * 7));

        const auto a = to_csr(to_dense(A));
        Csr z;
        z.rows = n; z.cols = n;
        z.row_ptr.assign(n + 1, 0);

        Csr c;
        c.rows = n; c.cols = n;
        c.row_ptr.assign(n + 1, 0);
        c.col_idx.assign(n + 1, 0);
        c.values.assign(n + 1, 0.0);

        cpu::gustavson_spgemm<int64_t, double, PlusTimesDouble>(
            a.row_ptr.data(), a.col_idx.data(), a.values.data(),
            z.row_ptr.data(), nullptr, nullptr,
            c.row_ptr.data(), c.col_idx.data(), c.values.data(),
            static_cast<int64_t>(n), static_cast<int64_t>(n),
            static_cast<int64_t>(n), sem);

        if (c.nnz() != 0) { ++::hptest::failure_count(); break; }
        ++checked;
    }
    EXPECT_EQ(checked, 30);
}

TEST(Property, DistributivityOverAddition) {
    std::mt19937_64 rng(8888);
    std::uniform_real_distribution<double> dens_dist(0.05, 0.25);
    PlusTimesDouble sem;

    auto mul = [](const Csr& x, const Csr& y, Csr& out) {
        out.rows = x.rows; out.cols = y.cols;
        out.row_ptr.assign(x.rows + 1, 0);
        out.col_idx.assign(product_capacity(x, y), 0);
        out.values.assign(out.col_idx.size(), 0.0);
        cpu::gustavson_spgemm<int64_t, double, PlusTimesDouble>(
            x.row_ptr.data(), x.col_idx.data(), x.values.data(),
            y.row_ptr.data(), y.col_idx.data(), y.values.data(),
            out.row_ptr.data(), out.col_idx.data(), out.values.data(),
            static_cast<int64_t>(x.rows), static_cast<int64_t>(y.cols),
            static_cast<int64_t>(x.cols), PlusTimesDouble{});
        out.col_idx.resize(out.nnz());
        out.values.resize(out.nnz());
    };

    int checked = 0;
    for (int trial = 0; trial < 25; ++trial) {
        const uint64_t n = 14;
        auto A = random_matrix(n, dens_dist(rng), static_cast<uint32_t>(trial * 2));
        auto B = random_matrix(n, dens_dist(rng), static_cast<uint32_t>(trial * 2 + 1));
        auto D = random_matrix(n, dens_dist(rng), static_cast<uint32_t>(trial * 2 + 2));

        const auto a = to_csr(to_dense(A));
        const auto b = to_csr(to_dense(B));
        const auto d = to_csr(to_dense(D));

        // C = (B + D), as a dense add.
        auto b_plus_d = to_dense(B);
        const auto dd = to_dense(D);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) b_plus_d[i][j] += dd[i][j];
        const auto c = to_csr(b_plus_d);

        Csr bc, lhs, bd, rhs;
        mul(b, a, bc);
        mul(d, a, bd);

        // rhs = A * (B + D) = (A*B) + (A*D)
        mul(a, c, rhs);

        // lhs = A*B + A*D
        auto lhs_dense = to_dense(bc);
        const auto bd_dense = to_dense(bd);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) lhs_dense[i][j] += bd_dense[i][j];

        if (!same(to_dense(rhs), lhs_dense, 1e-9)) {
            ++::hptest::failure_count();
            break;
        }
        ++checked;
    }
    EXPECT_EQ(checked, 25);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

TEST(Property, SingleElementMatrices) {
    PlusTimesDouble sem;
    // [a] * [b] == [a*b]
    const uint64_t rp1[2] = {0, 1};
    const uint64_t ci1[1] = {0};
    const double  v1[1] = {3.5};
    const double  v2[1] = {2.0};

    uint64_t out_rp[2] = {0, 0};
    uint64_t out_ci[1] = {0};
    double  out_v[1] = {0.0};

    cpu::gustavson_spgemm<int64_t, double, PlusTimesDouble>(
        rp1, ci1, v1, rp1, ci1, v2,
        out_rp, out_ci, out_v, 1, 1, 1, sem);

    EXPECT_EQ(out_rp[1], 1u);
    EXPECT_EQ(out_ci[0], 0u);
    EXPECT_NEAR(out_v[0], 7.0, 1e-12);
}

TEST(Property, EmptyRowTimesNonempty) {
    PlusTimesDouble sem;
    // A = [[0]] (nnz 0), B = [[5]]
    const uint64_t a_rp[2] = {0, 0};
    const uint64_t b_rp[2] = {0, 1};
    const uint64_t b_ci[1] = {0};
    const double  b_v[1]  = {5.0};

    uint64_t out_rp[2] = {0, 0};
    uint64_t out_ci[1] = {0};
    double  out_v[1] = {0.0};

    cpu::gustavson_spgemm<int64_t, double, PlusTimesDouble>(
        a_rp, nullptr, nullptr, b_rp, b_ci, b_v,
        out_rp, out_ci, out_v, 1, 1, 1, sem);

    EXPECT_EQ(out_rp[0], 0u);
    EXPECT_EQ(out_rp[1], 0u);
}

TEST(Property, NonEmptyTimesEmptyColumn) {
    PlusTimesDouble sem;
    // A = [[2]], B = [[]] (n = 0)
    const uint64_t a_rp[2] = {0, 1};
    const uint64_t a_ci[1] = {0};
    const double  a_v[1]  = {2.0};
    const uint64_t b_rp[1] = {0};

    uint64_t out_rp[2] = {0, 0};
    uint64_t out_ci[1] = {0};
    double  out_v[1] = {0.0};

    cpu::gustavson_spgemm<int64_t, double, PlusTimesDouble>(
        a_rp, a_ci, a_v, b_rp, nullptr, nullptr,
        out_rp, out_ci, out_v, 1, 0, 1, sem);

    EXPECT_EQ(out_rp[1], 0u);
}

TEST(Property, AsymmetricMatricesDoNotProduceSymmetricOutput) {
    PlusTimesDouble sem;
    // A is upper triangular, B is upper triangular; the product is too.
    const uint64_t n = 4;
    std::vector<uint64_t> a_rp(n + 1, 0), a_ci;
    std::vector<double> a_v;
    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = i; j < n; ++j) { a_ci.push_back(j); a_v.push_back(1.0); ++a_rp[i + 1]; }
    }
    for (uint64_t i = 0; i < n; ++i) a_rp[i + 1] += a_rp[i];

    const uint64_t* b_rp = a_rp.data();
    const uint64_t* b_ci = a_ci.data();
    const double*  b_v = a_v.data();

    std::vector<uint64_t> c_rp(n + 1, 0);
    std::vector<uint64_t> c_ci(256, 0);
    std::vector<double> c_v(256, 0.0);

    cpu::gustavson_spgemm<int64_t, double, PlusTimesDouble>(
        a_rp.data(), a_ci.data(), a_v.data(),
        b_rp, b_ci, b_v,
        c_rp.data(), c_ci.data(), c_v.data(),
        static_cast<int64_t>(n), static_cast<int64_t>(n),
        static_cast<int64_t>(n), sem);

    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t p = c_rp[i]; p < c_rp[i + 1]; ++p) {
            EXPECT_GE(c_ci[p], i);   // upper triangular
        }
    }
}

TEST(Property, HighLevelApiHandlesDegenerateInputs) {
    // n = 1 identity squared.
    auto I = std::make_unique<Matrix<double, MatrixFormat::COO>>(1, 1);
    I->nnz_ = 1;
    I->coo_rows_ = {0};
    I->coo_cols_ = {0};
    I->coo_values_ = {1.0};

    SpGEMMConfig cfg;
    cfg.use_cuda = false;
    auto s = create_spgemm(cfg);
    ASSERT_EQ(s->set_matrix_a<>(I.get()), 0);
    ASSERT_EQ(s->set_matrix_b<>(I.get()), 0);
    EXPECT_EQ(s->compute(), 0);
    EXPECT_EQ(s->stats().output_nnz, 1u);
}

TEST(Property, RandomConfigurationsStayInBounds) {
    // The autotuner's random sampling must never produce out-of-range values,
    // because the kernels index directly with them.
    ConfigurationSpace space = create_default_spgemm_space(2);
    std::mt19937_64 rng(2026);

    for (int i = 0; i < 5000; ++i) {
        const auto cfg = space.random_configuration(rng);

        const long tpb = std::stol(cfg.at("threads_per_block"));
        EXPECT_GE(tpb, 32);
        EXPECT_LE(tpb, 1024);
        // Block sizes must also be a multiple of the warp size.
        EXPECT_EQ(tpb % 32, 0);

        const long tpb_omp = std::stol(cfg.at("openmp_threads"));
        EXPECT_GE(tpb_omp, 1);
        EXPECT_LE(tpb_omp, 128);

        const long hms = std::stol(cfg.at("hashmap_size"));
        EXPECT_GE(hms, 64);
        EXPECT_LE(hms, 65536);

        const double lf = std::stod(cfg.at("load_factor"));
        EXPECT_GE(lf, 1.0);
        EXPECT_LE(lf, 4.0);
    }
}

HPC_TEST_MAIN()
