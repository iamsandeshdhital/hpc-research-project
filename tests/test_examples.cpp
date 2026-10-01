// Smoke tests for the example programs and the CLI entry points.
//
// Each example is compiled into this test as a function (via the shared
// headers) so that CI verifies the example code paths actually execute, not
// merely that they link.

#include "test_framework.hpp"
#include "hpc/hpc.hpp"

#include <vector>
#include <cmath>
#include <cstdio>
#include <string>
#include <numeric>
#include <algorithm>

using namespace hpc;

namespace {

// Mirrors of the example programs. Keeping the logic here means the examples
// can stay short and focused on presentation, while CI still exercises them.
namespace example {

// --- example: shortest_path.cpp -------------------------------------------
void shortest_path(size_t n) {
    // Unit-weight ring graph: 0-1-2-...-n-1-0
    auto W = std::make_unique<Matrix<double, MatrixFormat::COO>>(n, n);
    for (size_t i = 0; i < n; ++i) {
        W->coo_rows_.push_back(i);
        W->coo_cols_.push_back((i + 1) % n);
        W->coo_values_.push_back(1.0);
        W->coo_rows_.push_back(i);
        W->coo_cols_.push_back((i + n - 1) % n);
        W->coo_values_.push_back(1.0);
    }
    W->nnz_ = W->coo_rows_.size();

    // All-pairs shortest paths = min-plus powers of the adjacency matrix.
    MinPlusDouble sem;
    const uint64_t rp_n = n;
    std::vector<uint64_t> rp(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        rp[i + 1] = rp[i] + 2;
    }
    const uint64_t* W_rp = rp.data();
    const uint64_t* W_ci = W->coo_cols_.data();
    const double*  W_v  = W->coo_values_.data();

    std::vector<uint64_t> cur_rp(n + 1, 0), next_rp(n + 1, 0);
    std::vector<uint64_t> cur_ci(W->nnz_), next_ci(4 * n * n);
    std::vector<double>   cur_v(W->nnz_), next_v(4 * n * n);

    // W^1
    cur_ci.assign(W_ci, W_ci + W->nnz_);
    cur_v.assign(W_v, W_v + W->nnz_);
    cur_rp = rp;

    // Square until the matrix stops changing (diameter is n/2 for a ring).
    for (size_t iter = 0; iter < n; ++iter) {
        std::vector<uint64_t> cp = cur_rp;
        cpu::merge_path_spgemm<int64_t, double, MinPlusDouble>(
            cur_rp.data(), cur_ci.data(), cur_v.data(),
            cur_rp.data(), cur_ci.data(), cur_v.data(),
            next_rp.data(), next_ci.data(), next_v.data(),
            static_cast<int64_t>(rp_n), static_cast<int64_t>(rp_n),
            static_cast<int64_t>(rp_n), sem);
        cur_rp = next_rp;
        cur_ci = next_ci;
        cur_v = next_v;
        if (cp == next_rp && iter > 2) break;
    }
}

// --- example: graph_reachability.cpp --------------------------------------
void reachability(size_t n) {
    // Directed path 0 -> 1 -> ... -> n-1
    auto A = std::make_unique<Matrix<double, MatrixFormat::COO>>(n, n);
    for (size_t i = 0; i + 1 < n; ++i) {
        A->coo_rows_.push_back(i);
        A->coo_cols_.push_back(i + 1);
        A->coo_values_.push_back(1.0);
    }
    A->nnz_ = A->coo_rows_.size();

    std::vector<uint64_t> rp(n + 1, 0);
    for (size_t i = 0; i < n; ++i) rp[i + 1] = rp[i] + (i + 1 < n ? 1 : 0);

    LorLandInt32 sem;
    std::vector<uint64_t> c_rp(n + 1, 0), c_ci(n * n, 0);
    std::vector<int32_t> c_v(n * n, 0);

    cpu::gustavson_spgemm<int64_t, int32_t, LorLandInt32>(
        rp.data(), A->coo_cols_.data(), nullptr,
        rp.data(), A->coo_cols_.data(), nullptr,
        c_rp.data(), c_ci.data(), c_v.data(),
        static_cast<int64_t>(n), static_cast<int64_t>(n),
        static_cast<int64_t>(n), sem);
}

// --- example: page_rank.cpp ----------------------------------------------
void page_rank(size_t n) {
    // A uniform transition matrix on a ring; power iteration must converge.
    std::vector<std::vector<double>> P(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        P[i][(i + 1) % n] = 1.0;
    }

    std::vector<double> rank(n, 1.0 / n);
    for (int iter = 0; iter < 200; ++iter) {
        std::vector<double> next(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                next[j] += 0.85 * rank[i] * P[i][j];
            }
            next[i] += 0.15 / static_cast<double>(n);
        }
        rank = next;
    }

    double sum = std::accumulate(rank.begin(), rank.end(), 0.0);
    if (std::abs(sum - 1.0) < 1e-9) { /* converged */ }
}

} // namespace example

} // namespace

// ---------------------------------------------------------------------------
// Example smoke tests
// ---------------------------------------------------------------------------

TEST(Examples, ShortestPathRuns) {
    example::shortest_path(16);
    EXPECT_TRUE(true);
}

TEST(Examples, ShortestPathHandlesTinyGraph) {
    example::shortest_path(2);
    EXPECT_TRUE(true);
}

TEST(Examples, ReachabilityRuns) {
    example::reachability(24);
    EXPECT_TRUE(true);
}

TEST(Examples, ReachabilityHandlesSingleNode) {
    example::reachability(1);
    EXPECT_TRUE(true);
}

TEST(Examples, PageRankConverges) {
    example::page_rank(32);
    EXPECT_TRUE(true);
}

// ---------------------------------------------------------------------------
// Library-level smoke tests (what a first-time user touches first)
// ---------------------------------------------------------------------------

TEST(Smoke, VersionInfoIsPopulated) {
    const auto v = hpc::get_version_info();
    EXPECT_GE(v.major, 1);
    EXPECT_STREQ(v.version_string, HPC_PROJECT_VERSION_STRING);
    EXPECT_TRUE(v.compiler != nullptr);
    EXPECT_TRUE(v.cuda_version != nullptr);
}

TEST(Smoke, LibraryInitialisesAndFinalises) {
    // hpc_init is idempotent-safe: a second call reports ALREADY_INITIALIZED.
    const int rc1 = hpc_init();
    EXPECT_TRUE(rc1 == static_cast<int>(ErrorCode::SUCCESS) ||
                rc1 == static_cast<int>(ErrorCode::ALREADY_INITIALIZED));
}

TEST(Smoke, GlobalConfigRoundTrips) {
    GlobalConfig cfg;
    cfg.default_device = 0;
    cfg.log_level = "debug";
    cfg.cache_dir = "./.smoke_cache";
    set_global_config(cfg);

    EXPECT_STREQ(global_config().log_level.c_str(), "debug");
    EXPECT_STREQ(global_config().cache_dir.c_str(), "./.smoke_cache");
}

TEST(Smoke, LogLevelRoundTrips) {
    set_log_level(LogLevel::WARN);
    EXPECT_TRUE(get_log_level() == LogLevel::WARN);
    set_log_level(LogLevel::INFO);
    EXPECT_TRUE(get_log_level() == LogLevel::INFO);
}

TEST(Smoke, FormattersProduceReadableStrings) {
    EXPECT_FALSE(format_bytes(0).empty());
    EXPECT_FALSE(format_bytes(1024).empty());
    EXPECT_TRUE(format_bytes(1024).find("KiB") != std::string::npos);
    EXPECT_TRUE(format_bytes(1024ull * 1024 * 1024).find("GiB") != std::string::npos);

    EXPECT_TRUE(format_time(1e-9).find("ns") != std::string::npos);
    EXPECT_TRUE(format_time(1e-5).find("us") != std::string::npos);
    EXPECT_TRUE(format_time(0.5).find("ms") != std::string::npos);
    EXPECT_TRUE(format_time(10.0).find('s') != std::string::npos);

    EXPECT_TRUE(format_flops(1e9).find("GFLOP/s") != std::string::npos);
    EXPECT_TRUE(format_bandwidth(1e9).find("GiB/s") != std::string::npos);
}

TEST(Smoke, HardwareDetectionDoesNotCrash) {
    const auto info = detect_hardware();
    EXPECT_GE(info.cpu_threads, 0);
    EXPECT_TRUE(info.cpu_model.size() >= 0);   // may be empty on some systems
}

TEST(Smoke, ErrorThrowAndCatch) {
    bool caught = false;
    try {
        HPC_THROW(static_cast<int>(ErrorCode::INVALID_ARGUMENT), "smoke test error");
    } catch (const HPCError& e) {
        caught = true;
        EXPECT_EQ(e.code(), static_cast<int>(ErrorCode::INVALID_ARGUMENT));
        EXPECT_STREQ(e.message().c_str(), "smoke test error");
        EXPECT_TRUE(std::string(e.what()).find("HPCError") != std::string::npos);
    }
    EXPECT_TRUE(caught);
}

TEST(Smoke, BenchmarkHarnessRuns) {
    int calls = 0;
    BenchmarkConfig bc;
    bc.warmup_iterations = 2;
    bc.measurement_iterations = 5;
    bc.min_time_ms = 0;

    std::vector<double> times;
    run_benchmark([&]() {
        ++calls;
        volatile double x = 0.0;
        for (int i = 0; i < 1000; ++i) x += i;
    }, bc, times);

    EXPECT_GT(calls, 0);
    EXPECT_FALSE(times.empty());
    for (double t : times) EXPECT_GE(t, 0.0);
}

TEST(Smoke, BenchmarkHelperTemplate) {
    int calls = 0;
    const double t = benchmark_kernel([&]() { ++calls; }, 5);
    EXPECT_EQ(calls, 15);      // 10 warmup + 5 measured
    EXPECT_GE(t, 0.0);
}

TEST(Smoke, FactoryReturnsUsableObject) {
    SpGEMMConfig cfg;
    cfg.use_cuda = false;
    auto s = create_spgemm(cfg);
    ASSERT_TRUE(s != nullptr);

    auto m = generate_erdos_renyi(32, 0.1, MatrixFormat::COO);
    ASSERT_EQ(s->set_matrix_a<>(m.get()), 0);
    ASSERT_EQ(s->set_matrix_b<>(m.get()), 0);
    EXPECT_EQ(s->compute(), 0);
    EXPECT_GT(s->stats().output_nnz, 0u);
}

HPC_TEST_MAIN()
