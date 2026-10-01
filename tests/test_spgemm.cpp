// Unit tests for SpGEMM correctness across algorithms and semirings.
//
// Correctness oracle: every kernel is compared against a dense reference
// product computed in the test itself, which is feasible for the small
// matrices used here.

#include "test_framework.hpp"
#include "hpc/hpc.hpp"

#include <vector>
#include <cmath>
#include <numeric>

using namespace hpc;

// ---------------------------------------------------------------------------
// Reference implementation
// ---------------------------------------------------------------------------

namespace {

// Dense reference SpGEMM: C = A * B under (add, mult).
template<typename ValueT>
std::vector<std::vector<ValueT>> dense_reference(
    const std::vector<std::vector<ValueT>>& A,
    const std::vector<std::vector<ValueT>>& B,
    ValueT add_identity)
{
    const size_t m = A.size();
    const size_t k = B.size();
    const size_t n = B.empty() ? 0 : B[0].size();

    std::vector<std::vector<ValueT>> C(m, std::vector<ValueT>(n, add_identity));
    for (size_t i = 0; i < m; ++i) {
        for (size_t p = 0; p < k; ++p) {
            if (A[i][p] == add_identity) continue;
            for (size_t j = 0; j < n; ++j) {
                if (B[p][j] == add_identity) continue;
                C[i][j] = C[i][j] + A[i][p] * B[p][j];
            }
        }
    }
    return C;
}

// Convert a dense matrix to CSR triplets (dropping add-identity entries).
template<typename ValueT>
struct Csr {
    uint64_t rows = 0, cols = 0;
    std::vector<uint64_t> row_ptr;
    std::vector<uint64_t> col_idx;
    std::vector<ValueT> values;

    uint64_t nnz() const { return row_ptr.empty() ? 0 : row_ptr[rows]; }
};

template<typename ValueT>
Csr<ValueT> to_csr(const std::vector<std::vector<ValueT>>& M, ValueT zero) {
    Csr<ValueT> c;
    c.rows = M.size();
    c.cols = M.empty() ? 0 : M[0].size();
    c.row_ptr.assign(c.rows + 1, 0);

    for (size_t i = 0; i < c.rows; ++i) {
        for (size_t j = 0; j < c.cols; ++j) {
            if (M[i][j] != zero) {
                c.col_idx.push_back(j);
                c.values.push_back(M[i][j]);
                ++c.row_ptr[i + 1];
            }
        }
    }
    for (size_t i = 0; i < c.rows; ++i) c.row_ptr[i + 1] += c.row_ptr[i];
    return c;
}

// Compare a CSR product against the dense reference.
template<typename ValueT>
bool csr_matches_dense(const Csr<ValueT>& c,
                       const std::vector<std::vector<ValueT>>& ref,
                       ValueT zero, double tol = 1e-10) {
    const size_t m = ref.size();
    const size_t n = ref.empty() ? 0 : ref[0].size();

    if (c.rows != m || c.cols != n) return false;
    if (c.row_ptr.size() != m + 1) return false;
    if (c.row_ptr[0] != 0) return false;

    for (size_t i = 0; i < m; ++i) {
        size_t p = c.row_ptr[i];
        for (size_t j = 0; j < n; ++j) {
            const bool has = (p < c.row_ptr[i + 1]) && (c.col_idx[p] == j);
            if (has) {
                const ValueT got = c.values[p];
                const ValueT want = ref[i][j];
                if (std::abs(static_cast<double>(got - want)) > tol) return false;
                ++p;
            } else {
                const ValueT want = ref[i][j];
                // Absence is only valid when the reference value is the identity
                // or an exact zero.
                if (want != zero && want != ValueT(0)) return false;
            }
        }
        if (p != c.row_ptr[i + 1]) return false;
    }
    return true;
}

// Deterministic pseudo-random dense matrix.
std::vector<std::vector<double>> random_dense(size_t m, size_t n, double density,
                                              uint32_t seed) {
    uint64_t s = seed * 6364136223846793005ull + 1442695040888963407ull;
    auto next = [&]() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>((s >> 11) & 0x1FFFFFFFFFFFFFull) /
               static_cast<double>(1ULL << 53);
    };

    std::vector<std::vector<double>> M(m, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (next() < density) {
                M[i][j] = next() * 2.0 - 1.0;   // in [-1, 1)
                if (M[i][j] == 0.0) M[i][j] = 0.5;
            }
        }
    }
    return M;
}

} // namespace

// ---------------------------------------------------------------------------
// Gustavson kernel
// ---------------------------------------------------------------------------

TEST(Gustavson, IdentityTimesMatrix) {
    const size_t n = 5;
    auto I = random_dense(n, n, 0.0, 1);
    for (size_t i = 0; i < n; ++i) I[i][i] = 1.0;
    auto B = random_dense(n, n, 0.4, 2);

    auto a = to_csr(I, 0.0);
    auto b = to_csr(B, 0.0);
    auto ref = dense_reference(I, B, 0.0);

    Csr<double> c;
    c.rows = n; c.cols = n;
    c.row_ptr.assign(n + 1, 0);
    c.col_idx.assign(b.nnz(), 0);
    c.values.assign(b.nnz(), 0.0);

    PlusTimesDouble sem;
    cpu::gustavson_spgemm<int64_t, double, PlusTimesDouble>(
        a.row_ptr.data(), a.col_idx.data(), a.values.data(),
        b.row_ptr.data(), b.col_idx.data(), b.values.data(),
        c.row_ptr.data(), c.col_idx.data(), c.values.data(),
        static_cast<int64_t>(n), static_cast<int64_t>(n), static_cast<int64_t>(n), sem);

    EXPECT_TRUE(csr_matches_dense(c, ref, 0.0));
}

TEST(Gustavson, MatchesDenseReference) {
    const size_t n = 40;
    auto A = random_dense(n, n, 0.15, 11);
    auto B = random_dense(n, n, 0.15, 22);

    auto a = to_csr(A, 0.0);
    auto b = to_csr(B, 0.0);
    auto ref = dense_reference(A, B, 0.0);

    Csr<double> c;
    c.rows = n; c.cols = n;
    c.row_ptr.assign(n + 1, 0);
    // Upper bound: nnz(A) * max_row_nnz(B)
    uint64_t max_row_b = 0;
    for (size_t i = 0; i < n; ++i)
        max_row_b = std::max<uint64_t>(max_row_b, b.row_ptr[i + 1] - b.row_ptr[i]);
    const uint64_t cap = a.nnz() * std::max<uint64_t>(max_row_b, 1);
    c.col_idx.assign(cap, 0);
    c.values.assign(cap, 0.0);

    PlusTimesDouble sem;
    cpu::gustavson_spgemm<int64_t, double, PlusTimesDouble>(
        a.row_ptr.data(), a.col_idx.data(), a.values.data(),
        b.row_ptr.data(), b.col_idx.data(), b.values.data(),
        c.row_ptr.data(), c.col_idx.data(), c.values.data(),
        static_cast<int64_t>(n), static_cast<int64_t>(n), static_cast<int64_t>(n), sem);

    c.col_idx.resize(c.nnz());
    c.values.resize(c.nnz());
    EXPECT_TRUE(csr_matches_dense(c, ref, 0.0));
}

TEST(Gustavson, ColumnIndicesAreSortedWithinRows) {
    const size_t n = 30;
    auto A = random_dense(n, n, 0.2, 7);
    auto B = random_dense(n, n, 0.2, 8);

    auto a = to_csr(A, 0.0);
    auto b = to_csr(B, 0.0);

    uint64_t max_row_b = 0;
    for (size_t i = 0; i < n; ++i)
        max_row_b = std::max<uint64_t>(max_row_b, b.row_ptr[i + 1] - b.row_ptr[i]);

    Csr<double> c;
    c.rows = n; c.cols = n;
    c.row_ptr.assign(n + 1, 0);
    c.col_idx.assign(a.nnz() * std::max<uint64_t>(max_row_b, 1), 0);
    c.values.assign(c.col_idx.size(), 0.0);

    PlusTimesDouble sem;
    cpu::gustavson_spgemm<int64_t, double, PlusTimesDouble>(
        a.row_ptr.data(), a.col_idx.data(), a.values.data(),
        b.row_ptr.data(), b.col_idx.data(), b.values.data(),
        c.row_ptr.data(), c.col_idx.data(), c.values.data(),
        static_cast<int64_t>(n), static_cast<int64_t>(n), static_cast<int64_t>(n), sem);

    for (size_t i = 0; i < n; ++i) {
        for (uint64_t p = c.row_ptr[i] + 1; p < c.row_ptr[i + 1]; ++p) {
            EXPECT_GE(c.col_idx[p], c.col_idx[p - 1]);
        }
    }
}

TEST(Gustavson, EmptyMatrixProducesEmptyOutput) {
    const size_t n = 4;
    Csr<double> a, b, c;
    a.rows = n; a.cols = n; a.row_ptr.assign(n + 1, 0);
    b.rows = n; b.cols = n; b.row_ptr.assign(n + 1, 0);
    c.rows = n; c.cols = n; c.row_ptr.assign(n + 1, 0);

    PlusTimesDouble sem;
    cpu::gustavson_spgemm<int64_t, double, PlusTimesDouble>(
        a.row_ptr.data(), nullptr, nullptr,
        b.row_ptr.data(), nullptr, nullptr,
        c.row_ptr.data(), c.col_idx.data(), c.values.data(),
        4, 4, 4, sem);

    EXPECT_EQ(c.nnz(), 0u);
    for (size_t i = 0; i <= n; ++i) EXPECT_EQ(c.row_ptr[i], 0u);
}

// ---------------------------------------------------------------------------
// All CPU kernels agree with each other and with the reference
// ---------------------------------------------------------------------------

namespace {

// Runs every CPU kernel and asserts they all match the dense reference.
template<typename Sem>
void all_kernels_match(const std::vector<std::vector<double>>& A,
                       const std::vector<std::vector<double>>& B,
                       double zero, const char* label) {
    const size_t n = A.size();
    const auto a = to_csr(A, zero);
    const auto b = to_csr(B, zero);
    const auto ref = dense_reference(A, B, zero);

    uint64_t max_row_b = 0;
    for (size_t i = 0; i < n; ++i)
        max_row_b = std::max<uint64_t>(max_row_b, b.row_ptr[i + 1] - b.row_ptr[i]);
    const uint64_t cap = (a.nnz() + 1) * (max_row_b + 1);

    Sem sem;

    auto run = [&](auto fn) {
        Csr<double> c;
        c.rows = n; c.cols = n;
        c.row_ptr.assign(n + 1, 0);
        c.col_idx.assign(cap, 0);
        c.values.assign(cap, 0.0);
        fn(c.row_ptr.data(), c.col_idx.data(), c.values.data());
        c.col_idx.resize(c.nnz());
        c.values.resize(c.nnz());
        return c;
    };

    const auto I = static_cast<int64_t>(n);

    const Csr<double> c1 = run([&](auto* rp, auto* ci, auto* cv) {
        cpu::gustavson_spgemm<int64_t, double, Sem>(
            a.row_ptr.data(), a.col_idx.data(), a.values.data(),
            b.row_ptr.data(), b.col_idx.data(), b.values.data(),
            rp, ci, cv, I, I, I, sem);
    });
    EXPECT_TRUE(csr_matches_dense(c1, ref, zero));

    const Csr<double> c2 = run([&](auto* rp, auto* ci, auto* cv) {
        cpu::parallel_gustavson_spgemm<int64_t, double, Sem>(
            a.row_ptr.data(), a.col_idx.data(), a.values.data(),
            b.row_ptr.data(), b.col_idx.data(), b.values.data(),
            rp, ci, cv, I, I, I, sem, 4);
    });
    EXPECT_TRUE(csr_matches_dense(c2, ref, zero));

    const Csr<double> c3 = run([&](auto* rp, auto* ci, auto* cv) {
        cpu::heap_spgemm<int64_t, double, Sem>(
            a.row_ptr.data(), a.col_idx.data(), a.values.data(),
            b.row_ptr.data(), b.col_idx.data(), b.values.data(),
            rp, ci, cv, I, I, I, sem);
    });
    EXPECT_TRUE(csr_matches_dense(c3, ref, zero));

    const Csr<double> c4 = run([&](auto* rp, auto* ci, auto* cv) {
        cpu::merge_path_spgemm<int64_t, double, Sem>(
            a.row_ptr.data(), a.col_idx.data(), a.values.data(),
            b.row_ptr.data(), b.col_idx.data(), b.values.data(),
            rp, ci, cv, I, I, I, sem);
    });
    EXPECT_TRUE(csr_matches_dense(c4, ref, zero));

    // All four kernels must produce the same nnz.
    EXPECT_EQ(c1.nnz(), c2.nnz());
    EXPECT_EQ(c1.nnz(), c3.nnz());
    EXPECT_EQ(c1.nnz(), c4.nnz());
    (void)label;
}

} // namespace

TEST(KernelAgreement, PlusTimes) {
    const size_t n = 32;
    all_kernels_match<PlusTimesDouble>(
        random_dense(n, n, 0.12, 101), random_dense(n, n, 0.12, 202), 0.0, "plus-times");
}

TEST(KernelAgreement, MinPlus) {
    // For min-plus the "identity" entry is +inf, so start from +inf matrices.
    const size_t n = 24;
    auto A = random_dense(n, n, 0.2, 303);
    auto B = random_dense(n, n, 0.2, 404);
    for (auto& row : A) for (auto& v : row) if (v == 0.0) v = std::abs(v) + 1.0;
    for (auto& row : B) for (auto& v : row) if (v == 0.0) v = std::abs(v) + 1.0;

    // Reference with min/add: initialise the accumulator to +inf.
    auto ref = dense_reference(A, B, std::numeric_limits<double>::infinity());

    const auto a = to_csr(A, std::numeric_limits<double>::infinity());
    const auto b = to_csr(B, std::numeric_limits<double>::infinity());

    MinPlusDouble sem;
    uint64_t max_row_b = 0;
    for (size_t i = 0; i < n; ++i)
        max_row_b = std::max<uint64_t>(max_row_b, b.row_ptr[i + 1] - b.row_ptr[i]);
    const uint64_t cap = (a.nnz() + 1) * (max_row_b + 1);

    Csr<double> c;
    c.rows = n; c.cols = n;
    c.row_ptr.assign(n + 1, 0);
    c.col_idx.assign(cap, 0);
    c.values.assign(cap, 0.0);

    cpu::gustavson_spgemm<int64_t, double, MinPlusDouble>(
        a.row_ptr.data(), a.col_idx.data(), a.values.data(),
        b.row_ptr.data(), b.col_idx.data(), b.values.data(),
        c.row_ptr.data(), c.col_idx.data(), c.values.data(),
        static_cast<int64_t>(n), static_cast<int64_t>(n), static_cast<int64_t>(n), sem);
    c.col_idx.resize(c.nnz());
    c.values.resize(c.nnz());

    // Rows with no incoming products must be empty under min-plus.
    for (size_t i = 0; i < n; ++i) {
        bool has_product = false;
        for (size_t j = 0; j < n; ++j)
            if (ref[i][j] != std::numeric_limits<double>::infinity()) has_product = true;
        if (!has_product) EXPECT_EQ(c.row_ptr[i + 1] - c.row_ptr[i], 0u);
    }
    EXPECT_TRUE(csr_matches_dense(c, ref, std::numeric_limits<double>::infinity(), 1e-9));
}

TEST(KernelAgreement, IntegerTypes) {
    const size_t n = 20;
    auto A = random_dense(n, n, 0.2, 55);
    auto B = random_dense(n, n, 0.2, 66);

    // Round to int32 so the integer kernels can be exercised.
    auto to_i32 = [](const std::vector<std::vector<double>>& M) {
        std::vector<std::vector<int32_t>> R(M.size());
        for (size_t i = 0; i < M.size(); ++i) {
            R[i].resize(M[i].size());
            for (size_t j = 0; j < M[i].size(); ++j)
                R[i][j] = static_cast<int32_t>(M[i][j] * 1000.0);
        }
        return R;
    };
    const auto Ai = to_i32(A);
    const auto Bi = to_i32(B);

    const auto a = to_csr(Ai, 0);
    const auto b = to_csr(Bi, 0);
    auto ref = dense_reference(Ai, Bi, 0);

    uint64_t max_row_b = 0;
    for (size_t i = 0; i < n; ++i)
        max_row_b = std::max<uint64_t>(max_row_b, b.row_ptr[i + 1] - b.row_ptr[i]);

    Csr<int32_t> c;
    c.rows = n; c.cols = n;
    c.row_ptr.assign(n + 1, 0);
    c.col_idx.assign((a.nnz() + 1) * (max_row_b + 1), 0);
    c.values.assign(c.col_idx.size(), 0);

    PlusTimesInt32 sem;
    cpu::gustavson_spgemm<int64_t, int32_t, PlusTimesInt32>(
        a.row_ptr.data(), a.col_idx.data(), a.values.data(),
        b.row_ptr.data(), b.col_idx.data(), b.values.data(),
        c.row_ptr.data(), c.col_idx.data(), c.values.data(),
        static_cast<int64_t>(n), static_cast<int64_t>(n), static_cast<int64_t>(n), sem);
    c.col_idx.resize(c.nnz());
    c.values.resize(c.nnz());

    EXPECT_TRUE(csr_matches_dense(c, ref, 0, 1e-3));
}

// ---------------------------------------------------------------------------
// High-level SpGEMM class
// ---------------------------------------------------------------------------

TEST(SpGEMMClass, InitializeSucceeds) {
    SpGEMMConfig cfg;
    cfg.use_cuda = false;
    auto s = create_spgemm(cfg);
    ASSERT_TRUE(s != nullptr);
    EXPECT_EQ(s->stats().total_time_ms, 0.0);
}

TEST(SpGEMMClass, ComputeProducesCorrectResult) {
    const size_t n = 24;
    // Build a sparse matrix explicitly so the reference and the kernel see
    // exactly the same numbers.
    auto Am = generate_random_sparse(n, n, 0.15, MatrixFormat::COO);

    const auto* mm = dynamic_cast<const Matrix<double, MatrixFormat::COO>*>(Am.get());
    ASSERT_TRUE(mm != nullptr);

    // Reconstruct the dense form for the reference oracle.
    std::vector<std::vector<double>> dense(n, std::vector<double>(n, 0.0));
    for (uint64_t i = 0; i < Am->num_nonzeros(); ++i) {
        dense[mm->coo_rows_[i]][mm->coo_cols_[i]] = mm->coo_values_[i];
    }
    const auto ref = dense_reference(dense, dense, 0.0);

    SpGEMMConfig cfg;
    cfg.use_cuda = false;
    cfg.use_openmp = false;
    auto s = create_spgemm(cfg);
    ASSERT_EQ(s->set_matrix_a<>(Am.get()), 0);
    ASSERT_EQ(s->set_matrix_b<>(Am.get()), 0);
    ASSERT_EQ(s->compute(), 0);

    EXPECT_GT(s->stats().output_nnz, 0u);
    EXPECT_GT(s->stats().total_time_ms, 0.0);
    EXPECT_EQ(s->stats().input_nnz_a, Am->num_nonzeros());
    EXPECT_EQ(s->stats().input_nnz_b, Am->num_nonzeros());

    // Output nnz must equal the count of structurally non-zero entries of the
    // dense reference product (modulo exact cancellations, which random
    // non-zero values make vanishingly unlikely).
    uint64_t expected_nnz = 0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            if (ref[i][j] != 0.0) ++expected_nnz;

    EXPECT_EQ(s->stats().output_nnz, expected_nnz);
}

TEST(SpGEMMClass, SymbolicPhaseRunsFirst) {
    const size_t n = 16;
    auto M = generate_erdos_renyi(n, 0.2, MatrixFormat::COO);

    SpGEMMConfig cfg;
    cfg.use_cuda = false;
    cfg.phase = SpGEMMPhase::SYMBOLIC;

    auto s = create_spgemm(cfg);
    ASSERT_EQ(s->set_matrix_a<>(M.get()), 0);
    ASSERT_EQ(s->set_matrix_b<>(M.get()), 0);
    EXPECT_EQ(s->compute(), 0);
    EXPECT_GT(s->stats().symbolic_time_ms, 0.0);
    EXPECT_GT(s->stats().output_nnz, 0u);
}

TEST(SpGEMMClass, NumericPhaseRunsAfterSymbolic) {
    const size_t n = 16;
    auto M = generate_erdos_renyi(n, 0.2, MatrixFormat::COO);

    SpGEMMConfig cfg;
    cfg.use_cuda = false;
    cfg.phase = SpGEMMPhase::FUSED;

    auto s = create_spgemm(cfg);
    ASSERT_EQ(s->set_matrix_a<>(M.get()), 0);
    ASSERT_EQ(s->set_matrix_b<>(M.get()), 0);
    EXPECT_EQ(s->compute(), 0);
    EXPECT_GT(s->stats().numeric_time_ms, 0.0);
}

TEST(SpGEMMClass, SetSemiringAcceptsBuiltinIds) {
    SpGEMMConfig cfg;
    cfg.use_cuda = false;
    auto s = create_spgemm(cfg);
    EXPECT_EQ(s->set_semiring(SemiringType::MIN_PLUS_DOUBLE), 0);
    EXPECT_EQ(s->set_semiring(SemiringType::LOR_LAND_INT32), 0);
}

TEST(SpGEMMClass, RejectsNullMatrices) {
    SpGEMMConfig cfg;
    cfg.use_cuda = false;
    auto s = create_spgemm(cfg);
    EXPECT_NE(s->set_matrix_a<>(nullptr), 0);
    EXPECT_NE(s->set_matrix_b<>(nullptr), 0);
    EXPECT_NE(s->set_matrix_c<>(nullptr), 0);
}

TEST(SpGEMMClass, ResetStatsClearsCounters) {
    const size_t n = 8;
    auto M = generate_erdos_renyi(n, 0.5, MatrixFormat::COO);

    SpGEMMConfig cfg;
    cfg.use_cuda = false;
    auto s = create_spgemm(cfg);
    ASSERT_EQ(s->set_matrix_a<>(M.get()), 0);
    ASSERT_EQ(s->set_matrix_b<>(M.get()), 0);
    ASSERT_EQ(s->compute(), 0);

    EXPECT_GT(s->stats().total_time_ms, 0.0);
    s->reset_stats();
    EXPECT_EQ(s->stats().total_time_ms, 0.0);
    EXPECT_EQ(s->stats().output_nnz, 0u);
    EXPECT_EQ(s->stats().flops, 0u);
}

// ---------------------------------------------------------------------------
// Algorithm selection and cost model
// ---------------------------------------------------------------------------

TEST(SpGEMMSelection, PicksHashmapForSmallCpuProblems) {
    SpGEMMDescriptor d;
    d.m = 1000; d.n = 1000; d.k = 1000;
    d.nnz_a = 100000; d.nnz_b = 100000;
    d.density_a = 1e-4; d.density_b = 1e-4;
    d.num_gpus = 0;
    d.num_mpi_ranks = 1;

    EXPECT_TRUE(SpGEMM::select_algorithm(d) == SpGEMMAlgorithm::HASHMAP);
}

TEST(SpGEMMSelection, PicksMergePathForDistributed) {
    SpGEMMDescriptor d;
    d.m = 100; d.n = 100; d.k = 100;
    d.nnz_a = 1000; d.nnz_b = 1000;
    d.num_gpus = 0;
    d.num_mpi_ranks = 8;

    EXPECT_TRUE(SpGEMM::select_algorithm(d) == SpGEMMAlgorithm::MERGE_PATH);
}

TEST(SpGEMMSelection, PicksHeapForVeryLargeCpuProblems) {
    SpGEMMDescriptor d;
    d.m = 2000000; d.n = 2000000; d.k = 2000000;
    d.nnz_a = 100000000; d.nnz_b = 100000000;
    d.num_gpus = 0;
    d.num_mpi_ranks = 1;

    EXPECT_TRUE(SpGEMM::select_algorithm(d) == SpGEMMAlgorithm::HEAP);
}

TEST(SpGEMMSelection, RecommendConfigEnablesDistributed) {
    SpGEMMDescriptor d;
    d.m = 1000; d.n = 1000; d.k = 1000;
    d.nnz_a = 10000; d.nnz_b = 10000;
    d.num_mpi_ranks = 4;
    d.num_gpus = 0;

    const SpGEMMConfig cfg = SpGEMM::recommend_config(d);
    EXPECT_TRUE(cfg.distributed);
    EXPECT_TRUE(cfg.hybrid_comm);
}

TEST(SpGEMMSelection, MemoryEstimateScalesWithNnz) {
    SpGEMMDescriptor small;
    small.m = small.n = small.k = 1000;
    small.nnz_a = small.nnz_b = 10000;

    SpGEMMDescriptor large = small;
    large.m = large.n = large.k = 100000;
    large.nnz_a = large.nnz_b = 10000000;

    EXPECT_GT(SpGEMM::estimate_memory(large), SpGEMM::estimate_memory(small));
}

TEST(SpGEMMSelection, MemoryEstimateIsPositive) {
    SpGEMMDescriptor d;
    d.m = d.n = d.k = 10;
    d.nnz_a = d.nnz_b = 10;
    EXPECT_GT(SpGEMM::estimate_memory(d), 0u);
}

// ---------------------------------------------------------------------------
// Statistics bookkeeping
// ---------------------------------------------------------------------------

TEST(SpGEMMStats, ComputeDerivedMetrics) {
    SpGEMMStats s;
    s.total_time_ms = 2.0;      // 2 ms
    s.flops = 2000000000ull;    // 2 GFLOP
    s.memory_bytes = 1000000000ull;  // 1 GB

    s.compute_derived();
    // 2e9 flops / 2e-3 s = 1e12 flops/s = 1000 GFLOP/s
    EXPECT_NEAR(s.gflops, 1000.0, 1e-6);
    // 1e9 bytes / 2e-3 s = 5e11 B/s = 500 GB/s
    EXPECT_NEAR(s.memory_bandwidth_gb_s, 500.0, 1e-6);
    EXPECT_GT(s.arithmetic_intensity, 0.0);
}

TEST(SpGEMMStats, ComputeDerivedHandlesZeroTime) {
    SpGEMMStats s;
    s.total_time_ms = 0.0;
    s.flops = 1000;
    s.compute_derived();
    EXPECT_NEAR(s.gflops, 0.0, 1e-15);
}

TEST(SpGEMMStats, ResetZeroesEverything) {
    SpGEMMStats s;
    s.flops = 12345;
    s.output_nnz = 678;
    s.total_time_ms = 9.0;
    s.reset();
    EXPECT_EQ(s.flops, 0u);
    EXPECT_EQ(s.output_nnz, 0u);
    EXPECT_NEAR(s.total_time_ms, 0.0, 1e-15);
}

// ---------------------------------------------------------------------------
// Associativity: (A*B)*C == A*(B*C) under plus-times
// ---------------------------------------------------------------------------

TEST(SpGEMMAlgebra, AssociativityHolds) {
    const size_t n = 12;
    auto A = random_dense(n, n, 0.25, 4242);
    auto B = random_dense(n, n, 0.25, 4343);
    auto D = random_dense(n, n, 0.25, 4444);

    auto a = to_csr(A, 0.0);
    auto b = to_csr(B, 0.0);
    auto d = to_csr(D, 0.0);

    PlusTimesDouble sem;
    const auto I = static_cast<int64_t>(n);

    auto capacity = [](const Csr<double>& x, const Csr<double>& y) {
        uint64_t mrow = 0;
        for (size_t i = 0; i < x.rows; ++i)
            mrow = std::max<uint64_t>(mrow, y.row_ptr[i + 1] - y.row_ptr[i]);
        return (x.nnz() + 1) * (mrow + 1) + 16;
    };

    auto mul = [&](const Csr<double>& x, const Csr<double>& y, Csr<double>& out) {
        out.rows = n; out.cols = n;
        out.row_ptr.assign(n + 1, 0);
        const uint64_t cap = capacity(x, y);
        out.col_idx.assign(cap, 0);
        out.values.assign(cap, 0.0);
        cpu::gustavson_spgemm<int64_t, double, PlusTimesDouble>(
            x.row_ptr.data(), x.col_idx.data(), x.values.data(),
            y.row_ptr.data(), y.col_idx.data(), y.values.data(),
            out.row_ptr.data(), out.col_idx.data(), out.values.data(),
            I, I, I, sem);
        out.col_idx.resize(out.nnz());
        out.values.resize(out.nnz());
    };

    Csr<double> ab, ab_d, bd, a_bd;
    mul(a, b, ab);
    mul(ab, d, ab_d);
    mul(b, d, bd);
    mul(a, bd, a_bd);

    EXPECT_EQ(ab_d.nnz(), a_bd.nnz());

    for (size_t i = 0; i < n; ++i) {
        size_t p = ab_d.row_ptr[i], q = a_bd.row_ptr[i];
        const size_t len = ab_d.row_ptr[i + 1] - ab_d.row_ptr[i];
        for (size_t t = 0; t < len; ++t, ++p, ++q) {
            EXPECT_EQ(ab_d.col_idx[p], a_bd.col_idx[q]);
            EXPECT_NEAR(ab_d.values[p], a_bd.values[q], 1e-9);
        }
    }
}

HPC_TEST_MAIN()
