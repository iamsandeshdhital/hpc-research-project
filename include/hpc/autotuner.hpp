#pragma once

#include "config.hpp"
#include "spgemm.hpp"
#include "energy.hpp"
#include <cstdint>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <functional>
#include <random>

HPC_NAMESPACE_BEGIN

// Tuning parameter types
enum class ParameterType : uint8_t {
    INT = 0,
    FLOAT = 1,
    BOOL = 2,
    ENUM = 3,
    CATEGORICAL = 4
};

// Parameter constraint
struct HPC_API ParameterConstraint {
    ParameterType type = ParameterType::INT;
    int64_t min_int = 0;
    int64_t max_int = 100;
    double min_float = 0.0;
    double max_float = 1.0;
    std::vector<std::string> enum_values;
    std::vector<std::string> categorical_values;
    std::function<bool(const std::string&)> validator = nullptr;
};

// Tuning parameter
struct HPC_API TuningParameter {
    std::string name;
    ParameterConstraint constraint;
    std::string current_value;
    std::string default_value;
    std::string description;
    bool is_categorical = false;

    // Get typed value
    int64_t as_int() const;
    double as_float() const;
    bool as_bool() const;
    std::string as_string() const;

    // Set typed value
    void set_int(int64_t value);
    void set_float(double value);
    void set_bool(bool value);
    void set_string(const std::string& value);
};

// Configuration space
struct HPC_API ConfigurationSpace {
    std::map<std::string, TuningParameter> parameters;

    void add_parameter(const TuningParameter& param);
    void remove_parameter(const std::string& name);
    TuningParameter* get_parameter(const std::string& name);
    const TuningParameter* get_parameter(const std::string& name) const;
    std::vector<std::string> parameter_names() const;
    size_t size() const { return parameters.size(); }
    bool empty() const { return parameters.empty(); }

    // Generate random configuration
    std::map<std::string, std::string> random_configuration(std::mt19937& rng) const;

    // Generate neighbor configuration
    std::map<std::string, std::string> neighbor_configuration(
        const std::map<std::string, std::string>& current,
        std::mt19937& rng,
        double step_size = 0.1
    ) const;
};

// Search strategy types
enum class SearchStrategy : uint8_t {
    RANDOM = 0,
    GRID = 1,
    BAYESIAN = 2,
    GENETIC = 3,
    SIMULATED_ANNEALING = 4,
    PARTICLE_SWARM = 5,
    ML_BASED = 6,       // ML-based (PerfDojo-style)
    HYBRID = 7
};

// Tuning objective
enum class TuningObjective : uint8_t {
    MINIMIZE_TIME = 0,
    MINIMIZE_ENERGY = 1,
    MINIMIZE_EDP = 2,           // Energy-Delay Product
    MINIMIZE_ED2P = 3,          // Energy-Delay^2 Product
    MAXIMIZE_GFLOPS = 4,
    MAXIMIZE_THROUGHPUT = 5,
    MAXIMIZE_EFFICIENCY = 6,
    CUSTOM = 7
};

// Tuning result
struct HPC_API TuningResult {
    std::map<std::string, std::string> configuration;
    double objective_value = 0.0;
    SpGEMMStats stats;
    EnergyStats energy_stats;
    bool valid = false;
    std::string error_message;
    uint64_t timestamp_ns = 0;
    int iteration = 0;
};

// Tuning configuration
struct HPC_API TuningConfig {
    SearchStrategy strategy = SearchStrategy::ML_BASED;
    TuningObjective objective = TuningObjective::MINIMIZE_EDP;
    int max_iterations = 100;
    int max_time_seconds = 3600;
    int warmup_runs = 3;
    int measurement_runs = 5;
    int early_stop_patience = 20;
    double early_stop_threshold = 1e-4;
    int num_threads = 1;
    bool parallel_evaluation = false;
    bool verbose = true;
    std::string output_file = "tuning_results.csv";
    std::string checkpoint_file = "tuning_checkpoint.bin";
    int checkpoint_interval = 10;
    bool use_surrogate_model = true;
    int surrogate_update_interval = 5;
};

// ML-based tuner (PerfDojo-style)
class HPC_API MLTuner {
public:
    MLTuner(const ConfigurationSpace& space, const TuningConfig& config);
    ~MLTuner();

    // Run tuning
    std::vector<TuningResult> tune(
        const std::function<TuningResult(const std::map<std::string, std::string>&)>& evaluate
    );

    // Get best result
    const TuningResult& best_result() const;

    // Get all results
    const std::vector<TuningResult>& all_results() const;

    // Save/load model
    int save_model(const std::string& path);
    int load_model(const std::string& path);

    // Feature importance
    std::map<std::string, double> feature_importance() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Bayesian optimization tuner
class HPC_API BayesianTuner {
public:
    BayesianTuner(const ConfigurationSpace& space, const TuningConfig& config);
    ~BayesianTuner();

    std::vector<TuningResult> tune(
        const std::function<TuningResult(const std::map<std::string, std::string>&)>& evaluate
    );

    const TuningResult& best_result() const;
    const std::vector<TuningResult>& all_results() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Genetic algorithm tuner
class HPC_API GeneticTuner {
public:
    GeneticTuner(const ConfigurationSpace& space, const TuningConfig& config);
    ~GeneticTuner();

    std::vector<TuningResult> tune(
        const std::function<TuningResult(const std::map<std::string, std::string>&)>& evaluate
    );

    const TuningResult& best_result() const;
    const std::vector<TuningResult>& all_results() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Simulated annealing tuner
class HPC_API SimulatedAnnealingTuner {
public:
    SimulatedAnnealingTuner(const ConfigurationSpace& space, const TuningConfig& config);
    ~SimulatedAnnealingTuner();

    std::vector<TuningResult> tune(
        const std::function<TuningResult(const std::map<std::string, std::string>&)>& evaluate
    );

    const TuningResult& best_result() const;
    const std::vector<TuningResult>& all_results() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Auto-tuner factory
class HPC_API AutoTuner {
public:
    AutoTuner(const TuningConfig& config = TuningConfig{});
    ~AutoTuner();

    // Set search space
    void set_configuration_space(const ConfigurationSpace& space);

    // Run autotuning for SpGEMM
    std::vector<TuningResult> tune_spgemm(
        const SpGEMMDescriptor& descriptor,
        const std::function<SpGEMMStats(const SpGEMMConfig&)>& evaluate
    );

    // Run autotuning with energy awareness
    std::vector<TuningResult> tune_spgemm_energy(
        const SpGEMMDescriptor& descriptor,
        const std::function<std::pair<SpGEMMStats, EnergyStats>(const SpGEMMConfig&)>& evaluate
    );

    // Get best configuration
    SpGEMMConfig best_config() const;

    // Export results
    void export_results(const std::string& filename, const char* format = "csv") const;

    // Import results
    void import_results(const std::string& filename);

private:
    TuningConfig config_;
    ConfigurationSpace space_;
    std::unique_ptr<MLTuner> ml_tuner_;
    std::unique_ptr<BayesianTuner> bayesian_tuner_;
    std::unique_ptr<GeneticTuner> genetic_tuner_;
    std::unique_ptr<SimulatedAnnealingTuner> sa_tuner_;
    std::vector<TuningResult> results_;
    TuningResult best_result_;
};

// Default SpGEMM configuration space
HPC_API ConfigurationSpace create_default_spgemm_space(int num_gpus = 1);

// Predefined spaces for specific algorithms
HPC_API ConfigurationSpace create_hashmap_space();
HPC_API ConfigurationSpace create_galatic_space();
HPC_API ConfigurationSpace create_merge_path_space();
HPC_API ConfigurationSpace create_distributed_space(int num_ranks);

// Utility functions
HPC_API double evaluate_objective(
    const SpGEMMStats& stats,
    const EnergyStats& energy,
    TuningObjective objective
);

HPC_API std::string objective_to_string(TuningObjective obj);
HPC_API std::string strategy_to_string(SearchStrategy strat);

// ML model for performance prediction (simplified)
class HPC_API PerformancePredictor {
public:
    PerformancePredictor();
    ~PerformancePredictor();

    // Train on historical data
    int train(const std::vector<TuningResult>& data);

    // Predict performance for configuration
    double predict(const std::map<std::string, std::string>& config) const;

    // Predict with uncertainty
    std::pair<double, double> predict_with_uncertainty(const std::map<std::string, std::string>& config) const;

    // Save/load
    int save(const std::string& path);
    int load(const std::string& path);

    // Feature names
    std::vector<std::string> feature_names() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Roofline model integration
struct HPC_API RooflineModel {
    double peak_flops = 0.0;
    double peak_bandwidth = 0.0;
    double compute_ceilings[4] = {0};  // FP64, FP32, FP16, INT8
    double memory_ceilings[4] = {0};

    // Compute roofline prediction
    double predict_performance(double arithmetic_intensity, int precision = 1) const;

    // Find bottleneck
    enum class Bottleneck { COMPUTE, MEMORY, BALANCED };
    Bottleneck identify_bottleneck(double arithmetic_intensity, int precision = 1) const;

    // Load from hardware specs
    static RooflineModel from_device(int device_id);
    static RooflineModel from_specs(double peak_flops, double peak_bandwidth);
};

HPC_NAMESPACE_END