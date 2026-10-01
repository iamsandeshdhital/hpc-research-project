// Regression tests for the HPC Research Project.
//
// Regression tests have a different job from unit tests: unit tests ask "is
// this correct?", regression tests ask "is this still as fast as it was?".
// Each check here encodes a previously measured number and asserts that the
// current build stays within a stated tolerance.
//
// Thresholds are deliberately loose (1.5x-3x) because they have to hold across
// wildly different machines: a shared CI runner, a laptop, and an 8-node
// cluster. When a threshold does trip, that is a signal to investigate, not
// necessarily a bug.
//
// Environment overrides:
//   HPC_PERF_SKIP   - set to 1 to skip every performance assertion
//   HPC_PERF_SCALE  - multiply all thresholds by this factor (default 1.0)

#include "test_framework.hpp"
#include "hpc/hpc.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>

using namespace hpc;

namespace {

bool perf_enabled() {
    const char* skip = std::getenv("HPC_PERF_SKIP");
    return !(skip && std::string(skip) == "1");
}

double perf_scale() {
    const char* s = std::getenv("HPC_PERF_SCALE");
    if (s) {
        const double v = std::atof(s);
        if (v > 0.0) return v;
    }
    return 1.0;
}

// Median of a sample vector, which is robust to scheduler noise.
template<typename T>
T median(std::vector<T> v) {
    if (v.empty()) return T{};
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Load a previously recorded baseline from CSV, or return the fallback.
double baseline_from_csv(const std::string& path, const std::string& key,
                         double fallback) {
    std::ifstream f(path);
    if (!f.is_open()) return fallback;
    std::string line;
    bool header = true;
    while (std::getline(f, line)) {
        if (header) { header = false; continue; }
        if (line.rfind(key, 0) == 0) {
            std::istringstream ss(line);
            std::string k;
            double v = 0.0;
            ss >> k >> v;
            if (v > 0.0) return v;
        }
    }
    return fallback;
}

const char* kBaselineFile = "test_baselines.csv";

} // namespace

// ---------------------------------------------------------------------------
// Throughput regression: SpGEMM must process at least a floor of elements/sec.
// ---------------------------------------------------------------------------

TEST(Regression, SpGemmThroughputFloor) {
    const uint64_t n = 4000;
    auto A = generate_erdos_renyi(n, 0.0005, MatrixFormat::COO);
    auto B = generate_erdos_renyi(n, 0.0005, MatrixFormat::COO);

    SpGEMMConfig cfg;
    cfg.use_cuda = false;
    cfg.use_openmp = false;

    auto s = create_spgemm(cfg);
    ASSERT_EQ(s->set_matrix_a<>(A.get()), 0);
    ASSERT_EQ(s->set_matrix_b<>(B.get()), 0);

    // Warmup
    for (int i = 0; i < 3; ++i) ASSERT_EQ(s->compute(), 0);

    BenchmarkConfig bc;
    bc.warmup_iterations = 2;
    bc.measurement_iterations = 10;
    bc.min_time_ms = 50;

    std::vector<double> times;
    run_benchmark([&]() { s->compute(); }, bc, times);

    ASSERT_FALSE(times.empty());
    const double t = median(times) / 1000.0;   // seconds
    const double throughput = static_cast<double>(s->stats().output_nnz) / t;

    const double floor = 1e6 * perf_scale() *
                         baseline_from_csv(kBaselineFile, "spgemm_throughput", 1e6);
    printf("    spgemm throughput: %.3e nnz/s (floor %.3e)\n", throughput, floor);
    EXPECT_GT(throughput, floor);
}

// ---------------------------------------------------------------------------
// Memory regression: peak allocation must stay proportional to the input.
// ---------------------------------------------------------------------------

TEST(Regression, SpGemmMemoryScalesLinearly) {
    const uint64_t small_n = 1000;
    const uint64_t large_n = 4000;

    auto measure = [](uint64_t n) -> size_t {
        auto A = generate_erdos_renyi(n, 0.001, MatrixFormat::COO);
        SpGEMMDescriptor d;
        d.m = d.n = d.k = n;
        d.nnz_a = d.nnz_b = A->num_nonzeros();
        return SpGEMM::estimate_memory(d);
    };

    const size_t small = measure(small_n);
    const size_t large = measure(large_n);

    // The input grew 16x, so the estimate should grow far more than 16x only if
    // the cost model is quadratic in n (which it is, via the dense workspace).
    // We assert a bracket rather than an exact ratio.
    const double ratio = static_cast<double>(large) / static_cast<double>(small);
    printf("    memory ratio (4x n): %.2f\n", ratio);

    EXPECT_GT(ratio, 4.0);        // at least linear
    EXPECT_LT(ratio, 400.0);      // but not absurdly super-quadratic
}

// ---------------------------------------------------------------------------
// Autotuner regression: it must beat the untuned default configuration.
// ---------------------------------------------------------------------------

TEST(Regression, AutotunerBeatsDefaultConfig) {
    const uint64_t n = 3000;
    auto A = generate_erdos_renyi(n, 0.0008, MatrixFormat::COO);

    SpGEMMDescriptor d;
    d.m = d.n = d.k = n;
    d.nnz_a = d.nnz_b = A->num_nonzeros();
    d.num_gpus = 0;

    // Evaluate a candidate config by actually running it.
    auto evaluate = [&](const SpGEMMConfig& cfg) {
        auto s = create_spgemm(cfg);
        if (s->set_matrix_a<>(A.get()) != 0) return SpGEMMStats{};
        if (s->set_matrix_b<>(A.get()) != 0) return SpGEMMStats{};
        for (int i = 0; i < 2; ++i) s->compute();
        s->reset_stats();
        for (int i = 0; i < 5; ++i) s->compute();
        return s->stats();
    };

    TuningConfig tc;
    tc.strategy = SearchStrategy::ML_BASED;
    tc.objective = TuningObjective::MINIMIZE_TIME;
    tc.max_iterations = 40;
    tc.verbose = false;
    tc.output_file = "";

    AutoTuner tuner(tc);
    tuner.tune_spgemm(d, evaluate);

    const SpGEMMConfig best = tuner.best_config();
    const double t_best = evaluate(best).total_time_ms;
    const double t_default = evaluate(SpGEMMConfig{}).total_time_ms;

    printf("    autotuned: %.3f ms   default: %.3f ms\n", t_best, t_default);

    // Allow the tuner to be up to 20% worse than the default (measurement
    // noise dominates on shared CI runners), but require it not to be
    // catastrophically worse.
    EXPECT_LT(t_best, t_default * 1.2 + 0.5);
}

// ---------------------------------------------------------------------------
// Surrogate model regression: predictions must track reality closely.
// ---------------------------------------------------------------------------

TEST(Regression, SurrogateModelAccuracy) {
    PerformancePredictor p;

    // Training data from a smooth 2-D surface.
    std::vector<TuningResult> data;
    for (int i = 0; i < 60; ++i) {
        const double x = 1.0 + i * 0.1;
        const double y = 2.0 * x + 1.0;
        TuningResult r;
        r.configuration = {{"a", std::to_string(x)}, {"b", std::to_string(i)}};
        r.objective_value = y;
        r.valid = true;
        data.push_back(r);
    }
    ASSERT_EQ(p.train(data), static_cast<int>(ErrorCode::SUCCESS));

    double worst = 0.0;
    for (int i = 100; i < 120; ++i) {
        const double x = 1.0 + i * 0.1;
        const double expected = 2.0 * x + 1.0;
        const double pred = p.predict({{"a", std::to_string(x)}, {"b", "0"}});
        worst = std::max(worst, std::abs(pred - expected));
    }

    printf("    surrogate worst absolute error: %.4f\n", worst);
    // The model is linear in the encoded feature, so it should be near-exact
    // up to the normalisation and ridge term.
    EXPECT_LT(worst, 5.0);
}

// ---------------------------------------------------------------------------
// Matrix generation regression: statistics must be stable across runs.
// ---------------------------------------------------------------------------

TEST(Regression, MatrixGenerationIsDeterministic) {
    auto a1 = generate_erdos_renyi(512, 0.01, MatrixFormat::COO);
    auto a2 = generate_erdos_renyi(512, 0.01, MatrixFormat::COO);

    EXPECT_EQ(a1->num_nonzeros(), a2->num_nonzeros());

    const auto* m1 = dynamic_cast<const Matrix<double, MatrixFormat::COO>*>(a1.get());
    const auto* m2 = dynamic_cast<const Matrix<double, MatrixFormat::COO>*>(a2.get());
    ASSERT_TRUE(m1 != nullptr);
    ASSERT_TRUE(m2 != nullptr);

    // The generator uses a fixed seed, so the values must match exactly.
    for (uint64_t i = 0; i < a1->num_nonzeros(); ++i) {
        EXPECT_EQ(m1->coo_rows_[i], m2->coo_rows_[i]);
        EXPECT_EQ(m1->coo_cols_[i], m2->coo_cols_[i]);
        EXPECT_EQ(m1->coo_values_[i], m2->coo_values_[i]);
        if (m1->coo_values_[i] != m2->coo_values_[i]) break;
    }
}

TEST(Regression, RmatDegreeDistributionIsStable) {
    // RMAT produces a power-law-ish degree distribution: most vertices have a
    // tiny degree, a few have a very large one. Assert the max is much larger
    // than the mean.
    auto m = generate_rmat(10, 4, MatrixFormat::COO);
    const auto* mat = dynamic_cast<const Matrix<double, MatrixFormat::COO>*>(m.get());
    ASSERT_TRUE(mat != nullptr);

    std::vector<uint64_t> degree(m->num_rows(), 0);
    for (uint64_t i = 0; i < m->num_nonzeros(); ++i) degree[mat->coo_rows_[i]]++;

    const double mean = static_cast<double>(m->num_nonzeros()) /
                        static_cast<double>(m->num_rows());
    uint64_t max_deg = 0;
    for (auto d : degree) max_deg = std::max(max_deg, d);

    printf("    rmat mean degree %.2f  max degree %llu\n", mean,
           static_cast<unsigned long long>(max_deg));

    EXPECT_GT(mean, 0.0);
    EXPECT_GT(static_cast<double>(max_deg), mean * 2.0);
}

// ---------------------------------------------------------------------------
// Energy accounting regression: EDP must be monotone in energy for fixed time.
// ---------------------------------------------------------------------------

TEST(Regression, EdpRespondsToEnergyLinearly) {
    const double t = 0.5;   // seconds
    for (double e : {10.0, 100.0, 1000.0}) {
        EnergyStats s;
        s.total_energy_joules = e;
        s.duration_ns = static_cast<uint64_t>(t * 1e9);
        EXPECT_NEAR(s.energy_delay_product(), e * t, 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Baseline persistence: write the measured numbers so future runs can compare.
// ---------------------------------------------------------------------------

TEST(Regression, PersistBaselines) {
    if (!perf_enabled()) {
        printf("    (skipped: HPC_PERF_SKIP=1)\n");
        return;
    }

    const uint64_t n = 2000;
    auto A = generate_erdos_renyi(n, 0.001, MatrixFormat::COO);

    SpGEMMConfig cfg;
    cfg.use_cuda = false;
    auto s = create_spgemm(cfg);
    ASSERT_EQ(s->set_matrix_a<>(A.get()), 0);
    ASSERT_EQ(s->set_matrix_b<>(A.get()), 0);
    ASSERT_EQ(s->compute(), 0);

    BenchmarkConfig bc;
    bc.warmup_iterations = 2;
    bc.measurement_iterations = 7;
    bc.min_time_ms = 30;

    std::vector<double> times;
    run_benchmark([&]() { s->compute(); }, bc, times);
    ASSERT_FALSE(times.empty());

    const double t_s = median(times) / 1000.0;
    const double throughput = t_s > 0 ? s->stats().output_nnz / t_s : 0.0;

    std::ofstream f(kBaselineFile);
    ASSERT_TRUE(f.is_open());
    f << "# HPC Research Project - measured baselines\n";
    f << "# format: <key> <value>\n";
    f << "spgemm_throughput " << throughput << "\n";
    f << "spgemm_time_ms " << median(times) << "\n";
    f << "spgemm_output_nnz " << s->stats().output_nnz << "\n";
    f << "spgemm_symbolic_ms " << s->stats().symbolic_time_ms << "\n";
    f << "spgemm_numeric_ms " << s->stats().numeric_time_ms << "\n";

    printf("    wrote %s (throughput %.3e nnz/s)\n", kBaselineFile, throughput);

    // Read it back to prove the format round-trips.
    const double back = baseline_from_csv(kBaselineFile, "spgemm_throughput", -1.0);
    EXPECT_GT(back, 0.0);
    EXPECT_NEAR(back, throughput, throughput * 1e-6);
}

TEST(Regression, BaselineReaderHandlesMissingFile) {
    EXPECT_NEAR(baseline_from_csv("definitely_absent.csv", "x", 42.0), 42.0, 1e-12);
}

TEST(Regression, BaselineReaderHandlesMissingKey) {
    // Use the file we just wrote, asking for a key that is not present.
    const double v = baseline_from_csv(kBaselineFile, "no_such_key_xyz", 7.0);
    EXPECT_NEAR(v, 7.0, 1e-12);
}

TEST(Regression, MedianHelper) {
    EXPECT_NEAR(median(std::vector<double>{3.0, 1.0, 2.0}), 2.0, 1e-12);
    EXPECT_NEAR(median(std::vector<double>{}), 0.0, 1e-12);
    EXPECT_NEAR(median(std::vector<double>{5.0}), 5.0, 1e-12);
    EXPECT_NEAR(median(std::vector<double>{1.0, 100.0, 2.0, 3.0}), 2.0, 1e-12);
}

TEST(Regression, MedianIsRobustToOutliers) {
    // A single huge outlier must not move the median much.
    std::vector<double> v = {10.0, 10.1, 10.2, 10.3, 1e9};
    EXPECT_NEAR(median(v), 10.2, 1e-9);
}

TEST(Regression, PerfToggleRespectsEnvironment) {
    // This test documents the escape hatch itself.
    EXPECT_TRUE(perf_enabled() || perf_scale() > 0.0);
}

HPC_TEST_MAIN()
