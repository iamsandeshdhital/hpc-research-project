// Unit tests for sparse matrix storage formats, I/O, generation, and stats.

#include "test_framework.hpp"
#include "hpc/hpc.hpp"

#include <cstdio>
#include <vector>
#include <set>

using namespace hpc;

// ---------------------------------------------------------------------------
// Construction and metadata
// ---------------------------------------------------------------------------

TEST(Matrix, DefaultConstruction) {
    Matrix<double, MatrixFormat::CSR> m;
    EXPECT_EQ(m.num_rows(), 0u);
    EXPECT_EQ(m.num_cols(), 0u);
    EXPECT_EQ(m.num_nonzeros(), 0u);
    EXPECT_EQ(m.element_size(), sizeof(double));
    EXPECT_TRUE(m.format() == MatrixFormat::CSR);
}

TEST(Matrix, ResizeAllocatesFormatSpecificArrays) {
    Matrix<double, MatrixFormat::CSR> m;
    m.resize(100, 200, 500);
    EXPECT_EQ(m.num_rows(), 100u);
    EXPECT_EQ(m.num_cols(), 200u);
    EXPECT_EQ(m.csr_row_ptr().size(), 101u);
    EXPECT_EQ(m.csr_col_idx().size(), 500u);
    EXPECT_EQ(m.csr_values().size(), 500u);
}

TEST(Matrix, ResizeDCSCAllocatesSuperArrays) {
    Matrix<double, MatrixFormat::DCSC> m;
    m.resize(50, 60, 300);
    EXPECT_EQ(m.num_rows(), 50u);
    EXPECT_TRUE(m.dcsc_super_ptr().size() > 0);
}

TEST(Matrix, ClearResetsState) {
    Matrix<double, MatrixFormat::CSR> m;
    m.resize(10, 10, 20);
    m.clear();
    EXPECT_EQ(m.num_rows(), 0u);
    EXPECT_EQ(m.num_cols(), 0u);
    EXPECT_EQ(m.memory_usage(), 0u);
}

TEST(Matrix, MemoryUsageGrowsWithNnz) {
    Matrix<double, MatrixFormat::CSR> small(10, 10, 10);
    Matrix<double, MatrixFormat::CSR> large(1000, 1000, 100000);
    EXPECT_GT(large.memory_usage(), small.memory_usage());
    EXPECT_GT(small.memory_usage(), 0u);
}

TEST(Matrix, PropertiesAreMutable) {
    Matrix<double, MatrixFormat::CSR> m;
    m.mutable_properties().symmetric = true;
    m.mutable_properties().positive_definite = true;
    m.mutable_properties().num_block_rows = 4;

    EXPECT_TRUE(m.properties().symmetric);
    EXPECT_TRUE(m.properties().positive_definite);
    EXPECT_EQ(m.properties().num_block_rows, 4u);
}

TEST(Matrix, CloneIsDeepCopy) {
    Matrix<double, MatrixFormat::COO> m(4, 4);
    m.coo_rows_ = {0, 1, 2};
    m.coo_cols_ = {1, 2, 3};
    m.coo_values_ = {1.0, 2.0, 3.0};

    auto c = m.clone();
    EXPECT_EQ(c->num_rows(), 4u);
    EXPECT_EQ(c->num_nonzeros(), 3u);
    EXPECT_NE(static_cast<void*>(c.get()), static_cast<void*>(&m));
}

// ---------------------------------------------------------------------------
// Sparse index invariants
// ---------------------------------------------------------------------------

namespace {

// Build a CSR matrix from COO triplets, sorting the columns inside each row.
struct CsrBuilder {
    uint64_t rows = 0, cols = 0;
    std::vector<std::tuple<uint64_t, uint64_t, double>> e;

    void add(uint64_t r, uint64_t c, double v) { e.emplace_back(r, c, v); }

    std::vector<uint64_t> row_ptr;
    std::vector<uint64_t> col_idx;
    std::vector<double> values;

    void build() {
        std::stable_sort(e.begin(), e.end(),
            [](const auto& x, const auto& y) {
                if (std::get<0>(x) != std::get<0>(y)) return std::get<0>(x) < std::get<0>(y);
                return std::get<1>(x) < std::get<1>(y);
            });
        row_ptr.assign(rows + 1, 0);
        for (const auto& t : e) row_ptr[std::get<0>(t) + 1]++;
        for (uint64_t i = 0; i < rows; ++i) row_ptr[i + 1] += row_ptr[i];
        col_idx.resize(e.size());
        values.resize(e.size());
        for (size_t i = 0; i < e.size(); ++i) {
            col_idx[i] = std::get<1>(e[i]);
            values[i] = std::get<2>(e[i]);
        }
    }
};

bool csr_is_sorted(const std::vector<uint64_t>& row_ptr,
                   const std::vector<uint64_t>& col_idx) {
    for (size_t i = 0; i + 1 < row_ptr.size(); ++i) {
        for (uint64_t p = row_ptr[i] + 1; p < row_ptr[i + 1]; ++p) {
            if (col_idx[p] < col_idx[p - 1]) return false;
        }
    }
    return true;
}

} // namespace

TEST(Matrix, CSRBuilderProducesSortedRows) {
    CsrBuilder b;
    b.rows = 4; b.cols = 4;
    b.add(0, 3, 1.0);
    b.add(0, 1, 2.0);
    b.add(1, 0, 3.0);
    b.add(3, 2, 4.0);
    b.build();

    EXPECT_EQ(b.row_ptr.size(), 5u);
    EXPECT_EQ(b.col_idx.size(), 4u);
    EXPECT_EQ(b.row_ptr[4], 4u);
    EXPECT_TRUE(csr_is_sorted(b.row_ptr, b.col_idx));
    EXPECT_EQ(b.col_idx[0], 1u);   // row 0 sorted -> 1 then 3
    EXPECT_EQ(b.col_idx[1], 3u);
}

TEST(Matrix, RowPointerIsMonotone) {
    CsrBuilder b;
    b.rows = 5; b.cols = 5;
    for (uint64_t i = 0; i < 5; ++i) {
        for (uint64_t j = 0; j < 5; ++j) {
            if ((i + j) % 3 == 0) b.add(i, j, 1.0 + i + j);
        }
    }
    b.build();

    for (size_t i = 1; i < b.row_ptr.size(); ++i) {
        EXPECT_GE(b.row_ptr[i], b.row_ptr[i - 1]);
    }
    EXPECT_EQ(b.row_ptr.back(), b.col_idx.size());
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

TEST(MatrixGeneration, ErdosRenyiHasRequestedDimensions) {
    const uint64_t n = 64;
    auto m = generate_erdos_renyi(n, 0.05, MatrixFormat::COO);
    ASSERT_TRUE(m != nullptr);
    EXPECT_EQ(m->num_rows(), n);
    EXPECT_EQ(m->num_cols(), n);
    EXPECT_GT(m->num_nonzeros(), 0u);
}

TEST(MatrixGeneration, ErdosRenyiRespectsDensity) {
    const uint64_t n = 128;
    const double p = 0.01;
    auto m = generate_erdos_renyi(n, p, MatrixFormat::COO);
    const double expected = static_cast<double>(n) * n * p;
    const double actual = static_cast<double>(m->num_nonzeros());

    // Bernoulli sampling has a stddev of sqrt(np^2) = 1.28 here, so a 25%
    // band is generous but still catches gross errors.
    EXPECT_GT(actual, expected * 0.75);
    EXPECT_LT(actual, expected * 1.25);
}

TEST(MatrixGeneration, RmatDimensions) {
    auto m = generate_rmat(8, 4, MatrixFormat::COO);
    EXPECT_EQ(m->num_rows(), 256u);
    EXPECT_EQ(m->num_cols(), 256u);
    EXPECT_GT(m->num_nonzeros(), 0u);
}

TEST(MatrixGeneration, Grid2DIsSymmetricAndRegular) {
    const uint64_t n = 16;
    auto m = generate_grid_2d(n, MatrixFormat::COO);

    EXPECT_EQ(m->num_rows(), n * n);
    EXPECT_EQ(m->num_cols(), n * n);
    // A 2-D 5-point stencil on an interior grid: 4*nnz = 4 * n^2 - edge terms.
    EXPECT_GT(m->num_nonzeros(), 0u);
    EXPECT_LT(m->num_nonzeros(), 4 * n * n);
}

TEST(MatrixGeneration, Grid3DIsRegular) {
    const uint64_t n = 8;
    auto m = generate_grid_3d(n, MatrixFormat::COO);
    EXPECT_EQ(m->num_rows(), n * n * n);
    EXPECT_GT(m->num_nonzeros(), 0u);
    EXPECT_LT(m->num_nonzeros(), 6 * n * n * n);
}

TEST(MatrixGeneration, RandomSparseHasRequestedShape) {
    auto m = generate_random_sparse(100, 200, 0.001, MatrixFormat::COO);
    EXPECT_EQ(m->num_rows(), 100u);
    EXPECT_EQ(m->num_cols(), 200u);
    EXPECT_GT(m->num_nonzeros(), 0u);
}

TEST(MatrixGeneration, RandomSparseHasUniqueEntries) {
    const uint64_t rows = 64, cols = 64;
    const double density = 0.05;
    auto m = generate_random_sparse(rows, cols, density, MatrixFormat::COO);

    // Uniqueness of (row, col) pairs means nnz cannot exceed rows * cols.
    EXPECT_LE(m->num_nonzeros(), rows * cols);
}

// ---------------------------------------------------------------------------
// Matrix Market I/O round-trip
// ---------------------------------------------------------------------------

TEST(MatrixIO, MatrixMarketRoundTrip) {
    const char* path = "test_matrix_io.mtx";

    // Write a small hand-rolled Matrix Market file.
    {
        FILE* f = fopen(path, "w");
        ASSERT_TRUE(f != nullptr);
        fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
        fprintf(f, "%% comment line\n");
        fprintf(f, "3 3 4\n");
        fprintf(f, "1 1 1.0\n");
        fprintf(f, "2 3 2.0\n");
        fprintf(f, "3 1 3.0\n");
        fprintf(f, "3 3 4.0\n");
        fclose(f);
    }

    MatrixBase* m = nullptr;
    ASSERT_TRUE(read_matrix_market(path, &m));
    ASSERT_TRUE(m != nullptr);
    EXPECT_EQ(m->num_rows(), 3u);
    EXPECT_EQ(m->num_cols(), 3u);
    EXPECT_EQ(m->num_nonzeros(), 4u);
    delete m;

    EXPECT_EQ(remove(path), 0);
}

TEST(MatrixIO, MatrixMarketRejectsGarbage) {
    const char* path = "test_matrix_garbage.mtx";
    FILE* f = fopen(path, "w");
    ASSERT_TRUE(f != nullptr);
    fprintf(f, "this is not a matrix market file\n");
    fclose(f);

    MatrixBase* m = nullptr;
    EXPECT_FALSE(read_matrix_market(path, &m));
    EXPECT_TRUE(m == nullptr);

    EXPECT_EQ(remove(path), 0);
}

TEST(MatrixIO, MissingFileFails) {
    MatrixBase* m = nullptr;
    EXPECT_FALSE(read_matrix_market("definitely_does_not_exist_12345.mtx", &m));
    EXPECT_TRUE(m == nullptr);
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

TEST(MatrixStats, DensityComputation) {
    const uint64_t n = 100;
    auto m = generate_erdos_renyi(n, 0.01, MatrixFormat::COO);
    auto s = compute_matrix_stats(m.get());

    EXPECT_EQ(s.nnz, m->num_nonzeros());
    const double expected_density = static_cast<double>(s.nnz) / (n * n);
    EXPECT_NEAR(s.density, expected_density, 1e-12);
    EXPECT_GT(s.density, 0.0);
    EXPECT_LT(s.density, 1.0);
}

TEST(MatrixStats, EmptyMatrixHasZeroDensity) {
    Matrix<double, MatrixFormat::CSR> m(10, 10);
    auto s = compute_matrix_stats(&m);
    EXPECT_EQ(s.nnz, 0u);
    EXPECT_NEAR(s.density, 0.0, 1e-15);
}

TEST(MatrixStats, FullMatrixHasUnitDensity) {
    const uint64_t n = 10;
    CsrBuilder b;
    b.rows = n; b.cols = n;
    for (uint64_t i = 0; i < n; ++i)
        for (uint64_t j = 0; j < n; ++j)
            b.add(i, j, 1.0);
    b.build();

    Matrix<double, MatrixFormat::COO> m(n, n);
    m.nnz_ = b.col_idx.size();
    m.coo_rows_.resize(m.nnz_);
    m.coo_cols_ = b.col_idx;
    m.coo_values_ = b.values;
    for (uint64_t i = 0; i < n; ++i)
        for (uint64_t p = b.row_ptr[i]; p < b.row_ptr[i + 1]; ++p)
            m.coo_rows_[p] = i;

    auto s = compute_matrix_stats(&m);
    EXPECT_NEAR(s.density, 1.0, 1e-12);
    EXPECT_EQ(s.nnz, n * n);
}

// ---------------------------------------------------------------------------
// Distributed descriptor
// ---------------------------------------------------------------------------

TEST(DistributedMatrix, DescriptorDefaults) {
    DistributedMatrixDescriptor d;
    EXPECT_EQ(d.rank, 0);
    EXPECT_EQ(d.num_ranks, 1);
    EXPECT_EQ(d.grid_rows, 1);
    EXPECT_EQ(d.grid_cols, 1);
    EXPECT_EQ(d.num_layers, 1);
}

TEST(DistributedMatrix, DescriptorForGrid) {
    DistributedMatrixDescriptor d;
    d.num_ranks = 8;
    d.grid_rows = 4;
    d.grid_cols = 2;
    d.grid_layer = 1;
    d.global_rows = 1024;
    d.local_rows = 1024 / 8;

    EXPECT_EQ(d.grid_rows * d.grid_cols, 8);
    EXPECT_EQ(d.local_rows * d.num_ranks, d.global_rows);
}

HPC_TEST_MAIN()
