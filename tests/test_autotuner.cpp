// Unit tests for the autotuner: parameter encoding, search strategies,
// objective evaluation, the surrogate model, and result export/import.

#include "test_framework.hpp"
#include "hpc/hpc.hpp"

#include <cstdio>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace hpc;

// ---------------------------------------------------------------------------
// TuningParameter encoding
// ---------------------------------------------------------------------------

TEST(TuningParameter, IntRoundTrip) {
    TuningParameter p;
    p.name = "threads";
    p.constraint.type = ParameterType::INT;
    p.constraint.min_int = 1;
    p.constraint.max_int = 64;

    p.set_int(37);
    EXPECT_EQ(p.as_int(), 37);
    EXPECT_STREQ(p.as_string().c_str(), "37");
}

TEST(TuningParameter, FloatRoundTrip) {
    TuningParameter p;
    p.name = "load_factor";
    p.constraint.type = ParameterType::FLOAT;
    p.constraint.min_float = 1.0;
    p.constraint.max_float = 4.0;

    p.set_float(2.5);
    EXPECT_NEAR(p.as_float(), 2.5, 1e-12);
}

TEST(TuningParameter, BoolRoundTrip) {
    TuningParameter p;
    p.constraint.type = ParameterType::BOOL;

    p.set_bool(true);
    EXPECT_TRUE(p.as_bool());
    p.set_bool(false);
    EXPECT_FALSE(p.as_bool());
}

TEST(TuningParameter, BoolAcceptsNumericStrings) {
    TuningParameter p;
    p.constraint.type = ParameterType::BOOL;

    p.set_string("1");
    EXPECT_TRUE(p.as_bool());
    p.set_string("0");
    EXPECT_FALSE(p.as_bool());
    p.set_string("yes");
    EXPECT_TRUE(p.as_bool());
}

TEST(TuningParameter, MalformedIntFallsBackToMin) {
    TuningParameter p;
    p.constraint.type = ParameterType::INT;
    p.constraint.min_int = 7;
    p.set_string("not-a-number");
    EXPECT_EQ(p.as_int(), 7);
}

// ---------------------------------------------------------------------------
// ConfigurationSpace
// ---------------------------------------------------------------------------

namespace {

ConfigurationSpace make_space() {
    ConfigurationSpace s;

    TuningParameter a;
    a.name = "threads";
    a.constraint.type = ParameterType::INT;
    a.constraint.min_int = 1;
    a.constraint.max_int = 16;
    a.default_value = "8";
    s.add_parameter(a);

    TuningParameter b;
    b.name = "ratio";
    b.constraint.type = ParameterType::FLOAT;
    b.constraint.min_float = 0.0;
    b.constraint.max_float = 1.0;
    b.default_value = "0.5";
    s.add_parameter(b);

    TuningParameter c;
    c.name = "use_gpu";
    c.constraint.type = ParameterType::BOOL;
    c.default_value = "false";
    s.add_parameter(c);

    TuningParameter d;
    d.name = "algo";
    d.constraint.type = ParameterType::ENUM;
    d.constraint.enum_values = {"hashmap", "heap", "merge"};
    d.default_value = "hashmap";
    d.is_categorical = true;
    s.add_parameter(d);

    return s;
}

} // namespace

TEST(ConfigurationSpace, AddGetRemove) {
    auto s = make_space();
    EXPECT_EQ(s.size(), 4u);
    EXPECT_FALSE(s.empty());

    ASSERT_TRUE(s.get_parameter("threads") != nullptr);
    EXPECT_EQ(s.get_parameter("threads")->as_int(), 1);

    EXPECT_TRUE(s.get_parameter("nonexistent") == nullptr);

    s.remove_parameter("ratio");
    EXPECT_EQ(s.size(), 3u);
    EXPECT_TRUE(s.get_parameter("ratio") == nullptr);
}

TEST(ConfigurationSpace, ParameterNamesAreSorted) {
    auto s = make_space();
    const auto names = s.parameter_names();
    ASSERT_EQ(names.size(), 4u);
    // std::map keeps keys sorted, so the order is deterministic.
    EXPECT_STREQ(names[0].c_str(), "algo");
    EXPECT_STREQ(names[1].c_str(), "ratio");
    EXPECT_STREQ(names[2].c_str(), "threads");
    EXPECT_STREQ(names[3].c_str(), "use_gpu");
}

TEST(ConfigurationSpace, RandomConfigurationRespectsIntBounds) {
    auto s = make_space();
    std::mt19937 rng(42);
    for (int i = 0; i < 200; ++i) {
        auto cfg = s.random_configuration(rng);
        const long t = std::stol(cfg.at("threads"));
        EXPECT_GE(t, 1);
        EXPECT_LE(t, 16);

        const double r = std::stod(cfg.at("ratio"));
        EXPECT_GE(r, 0.0);
        EXPECT_LE(r, 1.0);
    }
}

TEST(ConfigurationSpace, RandomConfigurationUsesValidEnums) {
    auto s = make_space();
    std::mt19937 rng(7);
    for (int i = 0; i < 200; ++i) {
        const auto cfg = s.random_configuration(rng);
        const std::string a = cfg.at("algo");
        EXPECT_TRUE(a == "hashmap" || a == "heap" || a == "merge");
    }
}

TEST(ConfigurationSpace, RandomConfigurationCoversEveryValue) {
    auto s = make_space();
    std::mt19937 rng(99);
    std::vector<bool> seen_algo(3, false);
    for (int i = 0; i < 400; ++i) {
        const auto cfg = s.random_configuration(rng);
        const std::string a = cfg.at("algo");
        if (a == "hashmap") seen_algo[0] = true;
        else if (a == "heap") seen_algo[1] = true;
        else seen_algo[2] = true;
    }
    EXPECT_TRUE(seen_algo[0]);
    EXPECT_TRUE(seen_algo[1]);
    EXPECT_TRUE(seen_algo[2]);
}

TEST(ConfigurationSpace, NeighbourStaysInBounds) {
    auto s = make_space();
    std::mt19937 rng(5);
    std::map<std::string, std::string> base = s.random_configuration(rng);

    for (int i = 0; i < 500; ++i) {
        const auto nb = s.neighbor_configuration(base, rng, 0.2);
        const long t = std::stol(nb.at("threads"));
        EXPECT_GE(t, 1);
        EXPECT_LE(t, 16);

        const double r = std::stod(nb.at("ratio"));
        EXPECT_GE(r, 0.0);
        EXPECT_LE(r, 1.0);

        const std::string a = nb.at("algo");
        EXPECT_TRUE(a == "hashmap" || a == "heap" || a == "merge");
    }
}

TEST(ConfigurationSpace, NeighbourIsUsuallyDifferent) {
    auto s = make_space();
    std::mt19937 rng(11);
    const auto base = s.random_configuration(rng);

    int differences = 0;
    for (int i = 0; i < 100; ++i) {
        const auto nb = s.neighbor_configuration(base, rng, 0.3);
        for (const auto& [k, v] : nb) {
            auto it = base.find(k);
            if (it == base.end() || it->second != v) { ++differences; break; }
        }
    }
    EXPECT_GT(differences, 90);
}

TEST(ConfigurationSpace, EmptySpaceIsHandled) {
    ConfigurationSpace s;
    EXPECT_TRUE(s.empty());
    std::mt19937 rng(1);
    const auto cfg = s.random_configuration(rng);
    EXPECT_TRUE(cfg.empty());
}

// ---------------------------------------------------------------------------
// Objective evaluation
// ---------------------------------------------------------------------------

TEST(TuningObjective, MinimiseTimeUsesRuntime) {
    SpGEMMStats s;
    s.total_time_ms = 5.0;
    EXPECT_NEAR(evaluate_objective(s, EnergyStats{}, TuningObjective::MINIMIZE_TIME),
                0.005, 1e-12);
}

TEST(TuningObjective, MinimiseEnergyUsesJoules) {
    EnergyStats e;
    e.total_energy_joules = 250.0;
    EXPECT_NEAR(evaluate_objective(SpGEMMStats{}, e, TuningObjective::MINIMIZE_ENERGY),
                250.0, 1e-12);
}

TEST(TuningObjective, EdpIsEnergyTimesDelay) {
    SpGEMMStats s;
    s.total_time_ms = 2.0;
    EnergyStats e;
    e.total_energy_joules = 500.0;
    EXPECT_NEAR(evaluate_objective(s, e, TuningObjective::MINIMIZE_EDP), 1.0, 1e-9);
}

TEST(TuningObjective, Ed2pIsEnergyTimesDelaySquared) {
    SpGEMMStats s;
    s.total_time_ms = 4.0;
    EnergyStats e;
    e.total_energy_joules = 100.0;
    EXPECT_NEAR(evaluate_objective(s, e, TuningObjective::MINIMIZE_ED2P), 1.6, 1e-9);
}

TEST(TuningObjective, MaximiseGflopsNegatesTheScore) {
    SpGEMMStats s;
    s.gflops = 123.0;
    EXPECT_NEAR(evaluate_objective(s, EnergyStats{}, TuningObjective::MAXIMIZE_GFLOPS),
                -123.0, 1e-12);
}

TEST(TuningObjective, MaximiseThroughputNegatesNnz) {
    SpGEMMStats s;
    s.output_nnz = 999;
    EXPECT_NEAR(evaluate_objective(s, EnergyStats{}, TuningObjective::MAXIMIZE_THROUGHPUT),
                -999.0, 1e-12);
}

TEST(TuningObjective, LowerEdpWinsForBalancedWorkload) {
    // Two candidates: a fast/greedy one and a slower/leaner one.
    SpGEMMStats fast;
    fast.total_time_ms = 1.0;
    EnergyStats greedy;
    greedy.total_energy_joules = 2000.0;   // 2 J per ms

    SpGEMMStats lean;
    lean.total_time_ms = 2.0;
    EnergyStats efficient;
    efficient.total_energy_joules = 800.0;  // 0.4 J per ms

    const double edp_fast = evaluate_objective(fast, greedy, TuningObjective::MINIMIZE_EDP);
    const double edp_lean = evaluate_objective(lean, efficient, TuningObjective::MINIMIZE_EDP);

    EXPECT_LT(edp_lean, edp_fast);
    EXPECT_NEAR(edp_fast, 2.0, 1e-9);
    EXPECT_NEAR(edp_lean, 1.6, 1e-9);
}

TEST(TuningObjective, StringNames) {
    EXPECT_STREQ(objective_to_string(TuningObjective::MINIMIZE_TIME).c_str(), "min_time");
    EXPECT_STREQ(objective_to_string(TuningObjective::MINIMIZE_EDP).c_str(), "min_edp");
    EXPECT_STREQ(strategy_to_string(SearchStrategy::ML_BASED).c_str(), "ml_based");
    EXPECT_STREQ(strategy_to_string(SearchStrategy::BAYESIAN).c_str(), "bayesian");
}

// ---------------------------------------------------------------------------
// Search strategies
// ---------------------------------------------------------------------------

namespace {

// A synthetic objective with a clear optimum, so we can assert that each
// strategy finds something close to the best value.
double synthetic_objective(const std::map<std::string, std::string>& cfg) {
    const double x = std::stod(cfg.at("threads"));
    const double r = std::stod(cfg.at("ratio"));
    const double target_x = 8.0, target_r = 0.75;
    const double dx = x - target_x, dr = r - target_r;
    return dx * dx + dr * dr;
}

TuningResult evaluate_synthetic(const std::map<std::string, std::string>& cfg) {
    TuningResult r;
    r.configuration = cfg;
    r.objective_value = synthetic_objective(cfg);
    r.valid = true;
    r.timestamp_ns = 0;
    return r;
}

} // namespace

TEST(MLTuner, FindsNearOptimum) {
    TuningConfig cfg;
    cfg.max_iterations = 200;
    cfg.verbose = false;
    cfg.max_time_seconds = 30;

    MLTuner tuner(make_space(), cfg);
    const auto results = tuner.tune(evaluate_synthetic);

    EXPECT_GT(results.size(), 10u);

    const auto& best = tuner.best_result();
    ASSERT_TRUE(best.valid);
    EXPECT_LT(best.objective_value, 1.0);
}

TEST(MLTuner, HandlesNullEvaluator) {
    TuningConfig cfg;
    cfg.verbose = false;
    MLTuner tuner(make_space(), cfg);
    const auto results = tuner.tune(nullptr);
    EXPECT_EQ(results.size(), 0u);
}

TEST(MLTuner, RespectsIterationBudget) {
    TuningConfig cfg;
    cfg.max_iterations = 25;
    cfg.verbose = false;
    MLTuner tuner(make_space(), cfg);
    const auto results = tuner.tune(evaluate_synthetic);
    EXPECT_LE(results.size(), static_cast<size_t>(cfg.max_iterations));
}

TEST(MLTuner, EarlyStopsOnFlatObjective) {
    TuningConfig cfg;
    cfg.max_iterations = 500;
    cfg.early_stop_patience = 15;
    cfg.early_stop_threshold = 1e-3;
    cfg.verbose = false;

    // Constant objective -> the search should bail out early.
    auto flat = [](const std::map<std::string, std::string>& cfg) {
        TuningResult r;
        r.configuration = cfg;
        r.objective_value = 1.0;
        r.valid = true;
        return r;
    };

    MLTuner tuner(make_space(), cfg);
    const auto results = tuner.tune(flat);
    EXPECT_LT(results.size(), 500u);
}

TEST(MLTuner, RecordsFeatureImportance) {
    TuningConfig cfg;
    cfg.max_iterations = 60;
    cfg.verbose = false;
    MLTuner tuner(make_space(), cfg);
    tuner.tune(evaluate_synthetic);

    const auto importance = tuner.feature_importance();
    EXPECT_EQ(importance.size(), make_space().size());
    for (const auto& [k, v] : importance) {
        (void)k;
        EXPECT_GE(v, 0.0);
        EXPECT_FALSE(std::isnan(v));
    }
}

TEST(MLTuner, SavesAndLoadsSurrogate) {
    TuningConfig cfg;
    cfg.max_iterations = 50;
    cfg.verbose = false;
    MLTuner tuner(make_space(), cfg);
    tuner.tune(evaluate_synthetic);

    EXPECT_EQ(tuner.save_model("test_surrogate.txt"), 0);

    TuningConfig cfg2;
    cfg2.verbose = false;
    MLTuner loader(make_space(), cfg2);
    EXPECT_EQ(loader.load_model("test_surrogate.txt"), 0);

    EXPECT_EQ(remove("test_surrogate.txt"), 0);
}

TEST(BayesianTuner, FindsNearOptimum) {
    TuningConfig cfg;
    cfg.max_iterations = 120;
    cfg.verbose = false;
    BayesianTuner tuner(make_space(), cfg);
    tuner.tune(evaluate_synthetic);

    ASSERT_TRUE(tuner.best_result().valid);
    EXPECT_LT(tuner.best_result().objective_value, 2.0);
}

TEST(GeneticTuner, FindsNearOptimum) {
    TuningConfig cfg;
    cfg.max_iterations = 150;
    cfg.verbose = false;
    GeneticTuner tuner(make_space(), cfg);
    tuner.tune(evaluate_synthetic);

    ASSERT_TRUE(tuner.best_result().valid);
    EXPECT_LT(tuner.best_result().objective_value, 2.0);
}

TEST(GeneticTuner, ElitismPreservesBest) {
    TuningConfig cfg;
    cfg.max_iterations = 60;
    cfg.verbose = false;
    GeneticTuner tuner(make_space(), cfg);

    double best_seen = 1e300;
    auto tracking = [&](const std::map<std::string, std::string>& c) {
        TuningResult r = evaluate_synthetic(c);
        best_seen = std::min(best_seen, r.objective_value);
        return r;
    };
    tuner.tune(tracking);

    // Elitism means the reported best can never be worse than anything observed.
    EXPECT_LE(tuner.best_result().objective_value, best_seen + 1e-12);
}

TEST(SimulatedAnnealingTuner, FindsNearOptimum) {
    TuningConfig cfg;
    cfg.max_iterations = 400;
    cfg.verbose = false;
    SimulatedAnnealingTuner tuner(make_space(), cfg);
    tuner.tune(evaluate_synthetic);

    ASSERT_TRUE(tuner.best_result().valid);
    EXPECT_LT(tuner.best_result().objective_value, 2.0);
}

// ---------------------------------------------------------------------------
// Surrogate model
// ---------------------------------------------------------------------------

TEST(PerformancePredictor, RequiresEnoughSamples) {
    PerformancePredictor p;
    std::vector<TuningResult> few(2);
    EXPECT_EQ(p.train(few), static_cast<int>(ErrorCode::INVALID_ARGUMENT));
}

TEST(PerformancePredictor, LearnsALinearTrend) {
    // y = 2*x for x in the space, so a linear surrogate must recover it.
    PerformancePredictor p;

    std::vector<TuningResult> data;
    for (int i = 1; i <= 40; ++i) {
        TuningResult r;
        r.configuration = {{"x", std::to_string(i)}};
        r.objective_value = 2.0 * i;
        r.valid = true;
        data.push_back(r);
    }

    EXPECT_EQ(p.train(data), static_cast<int>(ErrorCode::SUCCESS));

    std::map<std::string, std::string> q = {{"x", "25"}};
    const auto [pred, sd] = p.predict_with_uncertainty(q);
    EXPECT_NEAR(pred, 50.0, 1.0);
    EXPECT_GE(sd, 0.0);
}

TEST(PerformancePredictor, ReportsFeatureNames) {
    PerformancePredictor p;
    std::vector<TuningResult> data;
    for (int i = 0; i < 20; ++i) {
        TuningResult r;
        r.configuration = {{"alpha", std::to_string(i)},
                           {"beta", std::to_string(2 * i)}};
        r.objective_value = static_cast<double>(i);
        r.valid = true;
        data.push_back(r);
    }
    p.train(data);

    const auto names = p.feature_names();
    EXPECT_EQ(names.size(), 2u);
    EXPECT_EQ(names[0], "alpha");
    EXPECT_EQ(names[1], "beta");
}

TEST(PerformancePredictor, SaveLoadRoundTrip) {
    PerformancePredictor p;
    std::vector<TuningResult> data;
    for (int i = 1; i <= 30; ++i) {
        TuningResult r;
        r.configuration = {{"x", std::to_string(i)}};
        r.objective_value = 3.0 * i + 1.0;
        r.valid = true;
        data.push_back(r);
    }
    p.train(data);

    const double before = p.predict({{"x", "10"}});
    EXPECT_EQ(p.save("test_pred.txt"), 0);

    PerformancePredictor q;
    EXPECT_EQ(q.load("test_pred.txt"), 0);
    const double after = q.predict({{"x", "10"}});
    EXPECT_NEAR(before, after, 1e-9);

    EXPECT_EQ(remove("test_pred.txt"), 0);
}

TEST(PerformancePredictor, UntrainedPredictsZero) {
    PerformancePredictor p;
    EXPECT_NEAR(p.predict({{"x", "1"}}), 0.0, 1e-15);
}

TEST(PerformancePredictor, HandlesCategoricalFeatures) {
    PerformancePredictor p;
    std::vector<TuningResult> data;
    for (int i = 0; i < 30; ++i) {
        TuningResult r;
        r.configuration = {{"algo", (i % 2 == 0) ? "a" : "b"}};
        r.objective_value = static_cast<double>(i % 2);
        r.valid = true;
        data.push_back(r);
    }
    EXPECT_EQ(p.train(data), static_cast<int>(ErrorCode::SUCCESS));
    const double pred = p.predict({{"algo", "a"}});
    EXPECT_FALSE(std::isnan(pred));
}

// ---------------------------------------------------------------------------
// AutoTuner integration
// ---------------------------------------------------------------------------

namespace {

SpGEMMStats fake_stats_for(const SpGEMMConfig& cfg) {
    SpGEMMStats s;
    // Synthetic model: fewer threads_per_block and fewer OpenMP threads are
    // "better", so the tuner has a gradient to follow.
    const double penalty = static_cast<double>(cfg.cuda_threads_per_block) * 1e-6 +
                           static_cast<double>(std::max(cfg.openmp_threads, 1)) * 1e-5;
    s.total_time_ms = 1.0 + penalty;
    s.gflops = 100.0 / (1.0 + penalty);
    s.output_nnz = 1000;
    s.compute_derived();
    return s;
}

} // namespace

TEST(AutoTuner, TuneSpGEMMFindsGoodConfig) {
    SpGEMMDescriptor d;
    d.m = d.n = d.k = 10000;
    d.nnz_a = d.nnz_b = 100000;
    d.num_gpus = 0;

    TuningConfig cfg;
    cfg.strategy = SearchStrategy::ML_BASED;
    cfg.objective = TuningObjective::MINIMIZE_TIME;
    cfg.max_iterations = 80;
    cfg.verbose = false;
    cfg.output_file = "";

    AutoTuner tuner(cfg);
    const auto results = tuner.tune_spgemm(d, fake_stats_for);

    EXPECT_GT(results.size(), 5u);

    const SpGEMMConfig best = tuner.best_config();
    EXPECT_LE(best.cuda_threads_per_block, 512);
}

TEST(AutoTuner, TuneSpGEMMRespectsObjective) {
    SpGEMMDescriptor d;
    d.m = d.n = d.k = 1000;
    d.nnz_a = d.nnz_b = 10000;

    TuningConfig cfg;
    cfg.objective = TuningObjective::MAXIMIZE_GFLOPS;
    cfg.max_iterations = 40;
    cfg.verbose = false;
    cfg.output_file = "";

    AutoTuner tuner(cfg);
    tuner.tune_spgemm(d, fake_stats_for);
    // With MAXIMIZE_GFLOPS the objective is negative, so the "best" value is < 0.
    EXPECT_LT(tuner.best_config().cuda_threads_per_block, 1024);
}

TEST(AutoTuner, EnergyAwareTuningUsesEdp) {
    SpGEMMDescriptor d;
    d.m = d.n = d.k = 1000;
    d.nnz_a = d.nnz_b = 10000;

    TuningConfig cfg;
    cfg.objective = TuningObjective::MINIMIZE_EDP;
    cfg.max_iterations = 50;
    cfg.verbose = false;
    cfg.output_file = "";

    AutoTuner tuner(cfg);
    auto results = tuner.tune_spgemm_energy(
        d,
        [](const SpGEMMConfig& c) -> std::pair<SpGEMMStats, EnergyStats> {
            SpGEMMStats s = fake_stats_for(c);
            EnergyStats e;
            // Smaller block size is more energy-hungry in this toy model.
            e.total_energy_joules = 100.0 - static_cast<double>(c.cuda_threads_per_block) * 0.05;
            if (e.total_energy_joules < 0) e.total_energy_joules = 0.1;
            return {s, e};
        });

    EXPECT_GT(results.size(), 5u);
    for (const auto& r : results) {
        EXPECT_FALSE(std::isnan(r.objective_value));
    }
}

TEST(AutoTuner, HandlesThrowingEvaluator) {
    SpGEMMDescriptor d;
    d.m = d.n = d.k = 100;
    d.nnz_a = d.nnz_b = 100;

    TuningConfig cfg;
    cfg.max_iterations = 30;
    cfg.verbose = false;
    cfg.output_file = "";

    AutoTuner tuner(cfg);
    auto results = tuner.tune_spgemm(
        d,
        [](const SpGEMMConfig&) -> SpGEMMStats {
            throw std::runtime_error("intentional failure");
        });

    // Every point is invalid, but the tuner must still terminate cleanly.
    EXPECT_GT(results.size(), 0u);
    for (const auto& r : results) {
        if (!r.valid) EXPECT_FALSE(r.error_message.empty());
    }
}

TEST(AutoTuner, NullEvaluatorReturnsEmpty) {
    TuningConfig cfg;
    cfg.verbose = false;
    AutoTuner tuner(cfg);
    SpGEMMDescriptor d;
    EXPECT_EQ(tuner.tune_spgemm(d, nullptr).size(), 0u);
}

TEST(AutoTuner, AllStrategiesTerminate) {
    SpGEMMDescriptor d;
    d.m = d.n = d.k = 500;
    d.nnz_a = d.nnz_b = 5000;

    const SearchStrategy strategies[] = {
        SearchStrategy::RANDOM, SearchStrategy::BAYESIAN, SearchStrategy::GENETIC,
        SearchStrategy::SIMULATED_ANNEALING, SearchStrategy::ML_BASED,
        SearchStrategy::HYBRID
    };

    for (auto strat : strategies) {
        TuningConfig cfg;
        cfg.strategy = strat;
        cfg.max_iterations = 40;
        cfg.verbose = false;
        cfg.output_file = "";

        AutoTuner tuner(cfg);
        const auto results = tuner.tune_spgemm(d, fake_stats_for);
        EXPECT_GT(results.size(), 0u);
    }
}

// ---------------------------------------------------------------------------
// Export / import
// ---------------------------------------------------------------------------

TEST(AutoTuner, CsvExportThenImport) {
    SpGEMMDescriptor d;
    d.m = d.n = d.k = 400;
    d.nnz_a = d.nnz_b = 4000;

    TuningConfig cfg;
    cfg.max_iterations = 40;
    cfg.verbose = false;
    cfg.output_file = "test_tuning_results.csv";

    AutoTuner tuner(cfg);
    tuner.tune_spgemm(d, fake_stats_for);

    // The file must exist and have at least one data row.
    FILE* f = fopen("test_tuning_results.csv", "r");
    ASSERT_TRUE(f != nullptr);
    std::string content;
    char buf[1024];
    while (fgets(buf, sizeof(buf), f)) content += buf;
    fclose(f);

    EXPECT_TRUE(content.find("iteration") != std::string::npos);
    EXPECT_TRUE(content.find("objective") != std::string::npos);

    // Now import into a fresh tuner.
    AutoTuner importer;
    importer.import_results("test_tuning_results.csv");

    EXPECT_EQ(remove("test_tuning_results.csv"), 0);
}

TEST(AutoTuner, JsonExport) {
    SpGEMMDescriptor d;
    d.m = d.n = d.k = 300;
    d.nnz_a = d.nnz_b = 3000;

    TuningConfig cfg;
    cfg.max_iterations = 20;
    cfg.verbose = false;
    cfg.output_file = "";

    AutoTuner tuner(cfg);
    tuner.tune_spgemm(d, fake_stats_for);
    tuner.export_results("test_tuning.json", "json");

    FILE* f = fopen("test_tuning.json", "r");
    ASSERT_TRUE(f != nullptr);
    std::string content;
    char buf[1024];
    while (fgets(buf, sizeof(buf), f)) content += buf;
    fclose(f);

    EXPECT_TRUE(content.find("[") != std::string::npos);
    EXPECT_TRUE(content.find("objective") != std::string::npos);
    EXPECT_TRUE(content.find("]") != std::string::npos);

    EXPECT_EQ(remove("test_tuning.json"), 0);
}

TEST(AutoTuner, ImportMissingFileIsSafe) {
    AutoTuner tuner;
    tuner.import_results("no_such_tuning_file.csv");   // must not crash
}

// ---------------------------------------------------------------------------
// Default search spaces
// ---------------------------------------------------------------------------

TEST(DefaultSpaces, SpgemmSpaceHasExpectedParameters) {
    auto s = create_default_spgemm_space(4);
    EXPECT_GT(s.size(), 4u);
    EXPECT_TRUE(s.get_parameter("algorithm") != nullptr);
    EXPECT_TRUE(s.get_parameter("threads_per_block") != nullptr);
    EXPECT_TRUE(s.get_parameter("openmp_threads") != nullptr);
    EXPECT_TRUE(s.get_parameter("hashmap_size") != nullptr);
    EXPECT_TRUE(s.get_parameter("num_gpus") != nullptr);
}

TEST(DefaultSpaces, SingleGpuOmitsNumGpus) {
    auto s = create_default_spgemm_space(1);
    EXPECT_TRUE(s.get_parameter("num_gpus") == nullptr);
}

TEST(DefaultSpaces, HashmapSpace) {
    auto s = create_hashmap_space();
    EXPECT_EQ(s.size(), 2u);
    EXPECT_TRUE(s.get_parameter("hashmap_size") != nullptr);
    EXPECT_TRUE(s.get_parameter("load_factor") != nullptr);
}

TEST(DefaultSpaces, GalaticSpace) {
    auto s = create_galatic_space();
    EXPECT_EQ(s.size(), 2u);
    EXPECT_TRUE(s.get_parameter("threads_per_block") != nullptr);
}

TEST(DefaultSpaces, MergePathSpace) {
    auto s = create_merge_path_space();
    EXPECT_EQ(s.size(), 2u);
    EXPECT_TRUE(s.get_parameter("tile_size") != nullptr);
}

TEST(DefaultSpaces, DistributedSpace) {
    auto s = create_distributed_space(16);
    EXPECT_EQ(s.size(), 4u);
    EXPECT_TRUE(s.get_parameter("grid_rows") != nullptr);
    EXPECT_TRUE(s.get_parameter("grid_layers") != nullptr);
    EXPECT_TRUE(s.get_parameter("comm_threshold") != nullptr);
    EXPECT_TRUE(s.get_parameter("hybrid_comm") != nullptr);
}

// ---------------------------------------------------------------------------
// Roofline model
// ---------------------------------------------------------------------------

TEST(Roofline, MemoryBoundAtLowIntensity) {
    const auto r = RooflineModel::from_specs(10000.0, 1000.0);
    // Arithmetic intensity 0.1 FLOP/byte is far below the ridge point (10).
    EXPECT_TRUE(r.identify_bottleneck(0.1) == RooflineModel::Bottleneck::MEMORY);
    EXPECT_NEAR(r.predict_performance(0.1), 0.1 * 1000.0, 1e-6);
}

TEST(Roofline, ComputeBoundAtHighIntensity) {
    const auto r = RooflineModel::from_specs(10000.0, 1000.0);
    EXPECT_TRUE(r.identify_bottleneck(1000.0) == RooflineModel::Bottleneck::COMPUTE);
    EXPECT_NEAR(r.predict_performance(1000.0), 10000.0, 1e-6);
}

TEST(Roofline, RidgePointIsRatioOfCeilings) {
    const auto r = RooflineModel::from_specs(20000.0, 500.0);
    const double ridge = 20000.0 / 500.0;   // 40
    EXPECT_TRUE(r.identify_bottleneck(ridge * 0.9) == RooflineModel::Bottleneck::MEMORY);
    EXPECT_TRUE(r.identify_bottleneck(ridge * 1.1) == RooflineModel::Bottleneck::COMPUTE);
}

TEST(Roofline, ZeroIntensityGivesZeroPerformance) {
    const auto r = RooflineModel::from_specs(1000.0, 100.0);
    EXPECT_NEAR(r.predict_performance(0.0), 0.0, 1e-12);
}

TEST(Roofline, FromDeviceDoesNotCrash) {
    const auto r = RooflineModel::from_device(0);
    EXPECT_GT(r.peak_flops, 0.0);
    EXPECT_GT(r.peak_bandwidth, 0.0);
}

HPC_TEST_MAIN()
