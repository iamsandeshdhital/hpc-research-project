#include "hpc/autotuner.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <cstring>
#include <chrono>

HPC_NAMESPACE_BEGIN

namespace {
inline uint64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

// Normalise a value into [0,1] for hashing/encoding
inline double normalize(double v, double lo, double hi) {
    if (hi <= lo) return 0.5;
    return std::max(0.0, std::min(1.0, (v - lo) / (hi - lo)));
}
} // namespace

// ---------------------------------------------------------------------------
// TuningParameter
// ---------------------------------------------------------------------------

int64_t TuningParameter::as_int() const {
    try { return std::stoll(current_value); } catch (...) { return min_int; }
}
double TuningParameter::as_float() const {
    try { return std::stod(current_value); } catch (...) { return min_float; }
}
bool TuningParameter::as_bool() const {
    return current_value == "true" || current_value == "1" || current_value == "yes";
}
std::string TuningParameter::as_string() const { return current_value; }

void TuningParameter::set_int(int64_t value) { current_value = std::to_string(value); }
void TuningParameter::set_float(double value) { current_value = std::to_string(value); }
void TuningParameter::set_bool(bool value) { current_value = value ? "true" : "false"; }
void TuningParameter::set_string(const std::string& value) { current_value = value; }

// ---------------------------------------------------------------------------
// ConfigurationSpace
// ---------------------------------------------------------------------------

void ConfigurationSpace::add_parameter(const TuningParameter& param) {
    parameters[param.name] = param;
}

void ConfigurationSpace::remove_parameter(const std::string& name) {
    parameters.erase(name);
}

TuningParameter* ConfigurationSpace::get_parameter(const std::string& name) {
    auto it = parameters.find(name);
    return it == parameters.end() ? nullptr : &it->second;
}

const TuningParameter* ConfigurationSpace::get_parameter(const std::string& name) const {
    auto it = parameters.find(name);
    return it == parameters.end() ? nullptr : &it->second;
}

std::vector<std::string> ConfigurationSpace::parameter_names() const {
    std::vector<std::string> names;
    names.reserve(parameters.size());
    for (const auto& [k, v] : parameters) names.push_back(k);
    return names;
}

std::map<std::string, std::string> ConfigurationSpace::random_configuration(
    std::mt19937& rng) const {
    std::map<std::string, std::string> cfg;
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    for (const auto& [name, p] : parameters) {
        switch (p.constraint.type) {
            case ParameterType::INT: {
                const double f = uni(rng);
                const int64_t lo = p.constraint.min_int, hi = p.constraint.max_int;
                cfg[name] = std::to_string(lo + static_cast<int64_t>(f * (hi - lo)));
                break;
            }
            case ParameterType::FLOAT: {
                const double f = uni(rng);
                const double lo = p.constraint.min_float, hi = p.constraint.max_float;
                cfg[name] = std::to_string(lo + f * (hi - lo));
                break;
            }
            case ParameterType::BOOL:
                cfg[name] = uni(rng) < 0.5 ? "true" : "false";
                break;
            case ParameterType::ENUM:
            case ParameterType::CATEGORICAL: {
                const auto& vals = p.constraint.type == ParameterType::ENUM
                    ? p.constraint.enum_values : p.constraint.categorical_values;
                if (!vals.empty()) {
                    const size_t idx = static_cast<size_t>(uni(rng) * vals.size());
                    cfg[name] = vals[std::min(idx, vals.size() - 1)];
                } else {
                    cfg[name] = p.default_value;
                }
                break;
            }
        }
    }
    return cfg;
}

std::map<std::string, std::string> ConfigurationSpace::neighbor_configuration(
    const std::map<std::string, std::string>& current,
    std::mt19937& rng,
    double step_size) const {
    std::map<std::string, std::string> cfg = current;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::uniform_int_distribution<int> coin(0, 1);

    for (const auto& [name, p] : parameters) {
        auto it = cfg.find(name);
        const std::string val = (it != cfg.end()) ? it->second : p.default_value;

        switch (p.constraint.type) {
            case ParameterType::INT: {
                int64_t v = std::atoll(val.c_str());
                const int64_t span = p.constraint.max_int - p.constraint.min_int;
                const int64_t delta = std::max<int64_t>(1, static_cast<int64_t>(span * step_size));
                v += (coin(rng) ? +1 : -1) * static_cast<int64_t>(uni(rng) * delta);
                v = std::max(p.constraint.min_int, std::min(p.constraint.max_int, v));
                cfg[name] = std::to_string(v);
                break;
            }
            case ParameterType::FLOAT: {
                double v = std::stod(val);
                const double span = p.constraint.max_float - p.constraint.min_float;
                v += (coin(rng) ? +1 : -1) * uni(rng) * span * step_size;
                v = std::max(p.constraint.min_float, std::min(p.constraint.max_float, v));
                cfg[name] = std::to_string(v);
                break;
            }
            case ParameterType::BOOL:
                cfg[name] = (coin(rng) == 0) ? "true" : "false";
                break;
            case ParameterType::ENUM:
            case ParameterType::CATEGORICAL: {
                const auto& vals = p.constraint.type == ParameterType::ENUM
                    ? p.constraint.enum_values : p.constraint.categorical_values;
                if (!vals.empty()) {
                    size_t idx = 0;
                    for (size_t i = 0; i < vals.size(); ++i) if (vals[i] == val) idx = i;
                    idx = (idx + (coin(rng) ? 1 : vals.size() - 1)) % vals.size();
                    cfg[name] = vals[idx];
                }
                break;
            }
        }
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Objective evaluation
// ---------------------------------------------------------------------------

double evaluate_objective(const SpGEMMStats& stats, const EnergyStats& energy,
                          TuningObjective objective) {
    const double time_s = stats.total_time_ms / 1000.0;
    const double e = energy.total_energy_joules > 0 ? energy.total_energy_joules : 0.0;

    switch (objective) {
        case TuningObjective::MINIMIZE_TIME:
            return time_s;
        case TuningObjective::MINIMIZE_ENERGY:
            return e;
        case TuningObjective::MINIMIZE_EDP:
            return e * time_s;
        case TuningObjective::MINIMIZE_ED2P:
            return e * time_s * time_s;
        case TuningObjective::MAXIMIZE_GFLOPS:
            return -stats.gflops;
        case TuningObjective::MAXIMIZE_THROUGHPUT:
            return -static_cast<double>(stats.output_nnz);
        case TuningObjective::MAXIMIZE_EFFICIENCY:
            return -stats.memory_bandwidth_gb_s;
        case TuningObjective::CUSTOM:
            return time_s;
    }
    return time_s;
}

std::string objective_to_string(TuningObjective obj) {
    switch (obj) {
        case TuningObjective::MINIMIZE_TIME:       return "min_time";
        case TuningObjective::MINIMIZE_ENERGY:     return "min_energy";
        case TuningObjective::MINIMIZE_EDP:        return "min_edp";
        case TuningObjective::MINIMIZE_ED2P:       return "min_ed2p";
        case TuningObjective::MAXIMIZE_GFLOPS:     return "max_gflops";
        case TuningObjective::MAXIMIZE_THROUGHPUT: return "max_throughput";
        case TuningObjective::MAXIMIZE_EFFICIENCY: return "max_efficiency";
        case TuningObjective::CUSTOM:             return "custom";
    }
    return "unknown";
}

std::string strategy_to_string(SearchStrategy strat) {
    switch (strat) {
        case SearchStrategy::RANDOM:             return "random";
        case SearchStrategy::GRID:               return "grid";
        case SearchStrategy::BAYESIAN:           return "bayesian";
        case SearchStrategy::GENETIC:            return "genetic";
        case SearchStrategy::SIMULATED_ANNEALING:return "simulated_annealing";
        case SearchStrategy::PARTICLE_SWARM:     return "particle_swarm";
        case SearchStrategy::ML_BASED:           return "ml_based";
        case SearchStrategy::HYBRID:             return "hybrid";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// PerformancePredictor  (ridge-regression surrogate on encoded features)
// ---------------------------------------------------------------------------

struct PerformancePredictor::Impl {
    std::vector<std::string> feature_names_;
    std::vector<double> weights_;
    double bias = 0.0;
    double residual_stddev = 0.0;
    bool trained = false;

    void encode(const std::map<std::string, std::string>& config,
                std::vector<double>& features) const {
        features.assign(feature_names_.size(), 0.0);
        for (size_t i = 0; i < feature_names_.size(); ++i) {
            const auto& name = feature_names_[i];
            auto it = config.find(name);
            if (it == config.end()) continue;
            try {
                // Treat the value as a plain number when possible.
                features[i] = std::stod(it->second);
            } catch (...) {
                // Categorical: stable hash into [0,1)
                features[i] = (static_cast<double>(
                    std::hash<std::string>{}(it->second) % 1000) / 1000.0);
            }
        }
    }
};

PerformancePredictor::PerformancePredictor() : impl_(std::make_unique<Impl>()) {}
PerformancePredictor::~PerformancePredictor() = default;

int PerformancePredictor::train(const std::vector<TuningResult>& data) {
    if (data.size() < 3) return static_cast<int>(ErrorCode::INVALID_ARGUMENT);

    // Determine the feature set from the first configuration.
    impl_->feature_names_.clear();
    for (const auto& [k, v] : data.front().configuration) {
        (void)v;
        impl_->feature_names_.push_back(k);
    }
    std::sort(impl_->feature_names_.begin(), impl_->feature_names_.end());

    const size_t n = data.size();
    const size_t d = impl_->feature_names_.size() + 1; // + bias

    // Design matrix with bias column and per-feature normalisation.
    std::vector<std::vector<double>> X(n, std::vector<double>(d, 0.0));
    std::vector<double> y(n, 0.0);

    std::vector<double> means(impl_->feature_names_.size(), 0.0);
    std::vector<double> stds(impl_->feature_names_.size(), 1.0);

    std::vector<std::vector<double>> raw(n);
    for (size_t i = 0; i < n; ++i) {
        impl_->encode(data[i].configuration, raw[i]);
        for (size_t j = 0; j < raw[i].size(); ++j) means[j] += raw[i][j];
    }
    for (auto& m : means) m /= static_cast<double>(n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < raw[i].size(); ++j) {
            const double dv = raw[i][j] - means[j];
            stds[j] += dv * dv;
        }
    }
    for (auto& s : stds) s = std::sqrt(s / static_cast<double>(n)) + 1e-9;

    for (size_t i = 0; i < n; ++i) {
        impl_->encode(data[i].configuration, raw[i]);
        for (size_t j = 0; j < impl_->feature_names_.size(); ++j) {
            X[i][j] = (raw[i][j] - means[j]) / stds[j];
        }
        X[i][d - 1] = 1.0;
        y[i] = data[i].objective_value;
    }

    // Solve normal equations for ridge regression: (X'X + lambda I) w = X'y
    const double lambda = 1e-3;
    std::vector<std::vector<double>> XtX(d, std::vector<double>(d, 0.0));
    std::vector<double> Xty(d, 0.0);

    for (size_t i = 0; i < n; ++i) {
        for (size_t a = 0; a < d; ++a) {
            Xty[a] += X[i][a] * y[i];
            for (size_t b = 0; b < d; ++b) {
                XtX[a][b] += X[i][a] * X[i][b];
            }
        }
    }
    for (size_t a = 0; a < d; ++a) XtX[a][a] += lambda * n / d;

    // Gaussian elimination with partial pivoting
    std::vector<double> w(d, 0.0);
    {
        std::vector<std::vector<double>> A = XtX;
        std::vector<double> b = Xty;
        for (size_t col = 0; col < d; ++col) {
            size_t pivot = col;
            for (size_t r = col + 1; r < d; ++r) {
                if (std::abs(A[r][col]) > std::abs(A[pivot][col])) pivot = r;
            }
            std::swap(A[col], A[pivot]);
            std::swap(b[col], b[pivot]);

            const double pv = A[col][col];
            if (std::abs(pv) < 1e-14) continue;
            for (size_t r = 0; r < d; ++r) {
                if (r == col) continue;
                const double f = A[r][col] / pv;
                if (f == 0.0) continue;
                for (size_t c = col; c < d; ++c) A[r][c] -= f * A[col][c];
                b[r] -= f * b[col];
            }
        }
        for (size_t i = 0; i < d; ++i) w[i] = std::abs(A[i][i]) > 1e-14 ? b[i] / A[i][i] : 0.0;
    }

    impl_->weights_ = w;
    impl_->bias = w[d - 1];
    impl_->weights_.resize(impl_->feature_names_.size());

    // Residual standard deviation gives the predictive uncertainty.
    double ss = 0.0;
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> f;
        impl_->encode(data[i].configuration, f);
        double pred = impl_->bias;
        for (size_t j = 0; j < f.size(); ++j) pred += impl_->weights_[j] * f[j];
        ss += (y[i] - pred) * (y[i] - pred);
    }
    impl_->residual_stddev = std::sqrt(ss / static_cast<double>(n));
    impl_->trained = true;

    return static_cast<int>(ErrorCode::SUCCESS);
}

double PerformancePredictor::predict(const std::map<std::string, std::string>& config) const {
    if (!impl_->trained) return 0.0;
    std::vector<double> f;
    impl_->encode(config, f);
    double pred = impl_->bias;
    for (size_t j = 0; j < f.size() && j < impl_->weights_.size(); ++j) {
        pred += impl_->weights_[j] * f[j];
    }
    return pred;
}

std::pair<double, double> PerformancePredictor::predict_with_uncertainty(
    const std::map<std::string, std::string>& config) const {
    const double p = predict(config);
    return {p, impl_->residual_stddev};
}

int PerformancePredictor::save(const std::string& path) {
    std::ofstream f(path);
    if (!f.is_open()) return static_cast<int>(ErrorCode::FILE_ERROR);
    f << impl_->feature_names_.size() << "\n";
    for (const auto& n : impl_->feature_names_) f << n << "\n";
    f << impl_->weights_.size() << "\n";
    for (double w : impl_->weights_) f << w << "\n";
    f << impl_->bias << "\n" << impl_->residual_stddev << "\n";
    return static_cast<int>(ErrorCode::SUCCESS);
}

int PerformancePredictor::load(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return static_cast<int>(ErrorCode::FILE_ERROR);
    size_t nfeat = 0;
    f >> nfeat;
    impl_->feature_names_.resize(nfeat);
    for (auto& n : impl_->feature_names_) f >> n;
    size_t nw = 0;
    f >> nw;
    impl_->weights_.resize(nw);
    for (auto& w : impl_->weights_) f >> w;
    f >> impl_->bias >> impl_->residual_stddev;
    impl_->trained = true;
    return static_cast<int>(ErrorCode::SUCCESS);
}

std::vector<std::string> PerformancePredictor::feature_names() const {
    return impl_->feature_names_;
}

// ---------------------------------------------------------------------------
// MLTuner  (PerfDojo-style: surrogate-guided search with exploitation of the
// best-so-far plus exploration around promising regions)
// ---------------------------------------------------------------------------

struct MLTuner::Impl {
    ConfigurationSpace space;
    TuningConfig config;
    PerformancePredictor predictor;
    std::vector<TuningResult> results;
    TuningResult best;

    void update_best() {
        for (const auto& r : results) {
            if (!r.valid) continue;
            if (!best.valid || r.objective_value < best.objective_value) best = r;
        }
    }
};

MLTuner::MLTuner(const ConfigurationSpace& space, const TuningConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->space = space;
    impl_->config = config;
}

MLTuner::~MLTuner() = default;

std::vector<TuningResult> MLTuner::tune(
    const std::function<TuningResult(const std::map<std::string, std::string>&)>& evaluate) {

    if (!evaluate) return {};

    std::mt19937 rng(0xC0FFEEu);
    const uint64_t t_start = now_ns();

    impl_->results.clear();
    impl_->best = TuningResult{};

    // Phase 1: initial design of experiments (Latin-hypercube-ish random sample)
    const int n_init = std::min<int>(std::max(4, impl_->config.max_iterations / 5),
                                     impl_->config.max_iterations);
    for (int i = 0; i < n_init; ++i) {
        auto cfg = impl_->space.random_configuration(rng);
        TuningResult r = evaluate(cfg);
        r.iteration = i;
        r.timestamp_ns = now_ns();
        impl_->results.push_back(r);
    }
    impl_->update_best();

    // Phase 2: surrogate-guided search
    for (int it = n_init; it < impl_->config.max_iterations; ++it) {
        const uint64_t elapsed_s = (now_ns() - t_start) / 1'000'000'000ull;
        if (impl_->config.max_time_seconds > 0 &&
            static_cast<int>(elapsed_s) >= impl_->config.max_time_seconds) break;

        // Early stopping check
        if (it >= impl_->config.early_stop_patience) {
            const auto& recent = impl_->results;
            double spread = 0.0;
            size_t lo = recent.size() > static_cast<size_t>(impl_->config.early_stop_patience)
                            ? recent.size() - impl_->config.early_stop_patience : 0;
            if (lo < recent.size() - 1) {
                double mn = recent[lo].objective_value, mx = mn;
                for (size_t i = lo; i < recent.size(); ++i) {
                    mn = std::min(mn, recent[i].objective_value);
                    mx = std::max(mx, recent[i].objective_value);
                }
                const double base = std::abs(impl_->best.objective_value) + 1e-12;
                spread = (mx - mn) / base;
            }
            if (spread < impl_->config.early_stop_threshold) break;
        }

        // Refit surrogate periodically
        if (impl_->config.use_surrogate_model &&
            (it % std::max(1, impl_->config.surrogate_update_interval) == 0)) {
            std::vector<TuningResult> valid;
            for (const auto& r : impl_->results) if (r.valid) valid.push_back(r);
            if (valid.size() >= 3) impl_->predictor.train(valid);
        }

        std::map<std::string, std::string> cfg;
        if (impl_->config.use_surrogate_model && impl_->results.size() >= 3 && impl_->best.valid) {
            // Exploitation: perturb the incumbent.
            const double temperature = 1.0 - static_cast<double>(it) /
                                              std::max(1, impl_->config.max_iterations);
            cfg = impl_->space.neighbor_configuration(impl_->best.configuration, rng,
                                                      std::max(0.02, temperature));
        } else {
            // Exploration: fresh random point.
            cfg = impl_->space.random_configuration(rng);
        }

        TuningResult r = evaluate(cfg);
        r.iteration = it;
        r.timestamp_ns = now_ns();
        impl_->results.push_back(r);
        impl_->update_best();

        if (impl_->config.verbose) {
            fprintf(stderr, "[ml-tuner] iter %3d  obj=%.6g  best=%.6g\n",
                    it, r.objective_value, impl_->best.objective_value);
        }
    }

    // Save the surrogate
    if (!impl_->config.checkpoint_file.empty()) {
        impl_->predictor.save(impl_->config.checkpoint_file);
    }

    return impl_->results;
}

const TuningResult& MLTuner::best_result() const { return impl_->best; }
const std::vector<TuningResult>& MLTuner::all_results() const { return impl_->results; }

int MLTuner::save_model(const std::string& path) { return impl_->predictor.save(path); }
int MLTuner::load_model(const std::string& path) { return impl_->predictor.load(path); }

std::map<std::string, double> MLTuner::feature_importance() const {
    std::map<std::string, double> importance;
    const auto names = impl_->predictor.feature_names();
    // Surrogate weight magnitude is a reasonable first-order importance proxy.
    // Recompute via a small perturbation study for correctness.
    for (size_t i = 0; i < names.size(); ++i) {
        importance[names[i]] = 0.0;
    }
    if (impl_->results.size() < 2 || !impl_->best.valid) return importance;

    std::mt19937 rng(1234);
    const double base = impl_->predictor.predict(impl_->best.configuration);
    for (const auto& name : names) {
        double acc = 0.0;
        const int trials = 8;
        for (int t = 0; t < trials; ++t) {
            auto pert = impl_->space.neighbor_configuration(impl_->best.configuration, rng, 0.25);
            auto it = pert.find(name);
            if (it == pert.end()) continue;
            acc += std::abs(impl_->predictor.predict(pert) - base);
        }
        importance[name] = acc / trials;
    }
    return importance;
}

// ---------------------------------------------------------------------------
// BayesianTuner (simplified GP-free expected-improvement over a KDE)
// ---------------------------------------------------------------------------

struct BayesianTuner::Impl {
    ConfigurationSpace space;
    TuningConfig config;
    std::vector<TuningResult> results;
    TuningResult best;

    void update_best() {
        for (const auto& r : results) {
            if (!r.valid) continue;
            if (!best.valid || r.objective_value < best.objective_value) best = r;
        }
    }
};

BayesianTuner::BayesianTuner(const ConfigurationSpace& space, const TuningConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->space = space;
    impl_->config = config;
}
BayesianTuner::~BayesianTuner() = default;

std::vector<TuningResult> BayesianTuner::tune(
    const std::function<TuningResult(const std::map<std::string, std::string>&)>& evaluate) {
    if (!evaluate) return {};

    std::mt19937 rng(0xBAFEu);
    const int n_init = std::max(4, impl_->config.max_iterations / 5);

    for (int i = 0; i < n_init; ++i) {
        auto cfg = impl_->space.random_configuration(rng);
        TuningResult r = evaluate(cfg);
        r.iteration = i;
        impl_->results.push_back(r);
    }
    impl_->update_best();

    const int candidates = 64;
    for (int it = n_init; it < impl_->config.max_iterations; ++it) {
        // Sample candidates and pick the one furthest from the incumbent
        // (maximin / discrepancy-style exploration).
        std::map<std::string, std::string> chosen;
        double chosen_dist = -1.0;
        for (int c = 0; c < candidates; ++c) {
            auto cand = impl_->space.random_configuration(rng);
            double dist = 0.0;
            for (const auto& [k, v] : cand) {
                auto itb = impl_->best.configuration.find(k);
                if (itb == impl_->best.configuration.end()) continue;
                try {
                    dist += std::abs(std::stod(v) - std::stod(itb->second));
                } catch (...) {
                    dist += (v == itb->second) ? 0.0 : 1.0;
                }
            }
            if (dist > chosen_dist) { chosen_dist = dist; chosen = cand; }
        }
        TuningResult r = evaluate(chosen);
        r.iteration = it;
        impl_->results.push_back(r);
        impl_->update_best();
    }
    return impl_->results;
}

const TuningResult& BayesianTuner::best_result() const { return impl_->best; }
const std::vector<TuningResult>& BayesianTuner::all_results() const { return impl_->results; }

// ---------------------------------------------------------------------------
// GeneticTuner
// ---------------------------------------------------------------------------

struct GeneticTuner::Impl {
    ConfigurationSpace space;
    TuningConfig config;
    std::vector<TuningResult> results;   // archive of all evaluated
    TuningResult best;
    size_t population_size = 32;

    void update_best() {
        for (const auto& r : results) {
            if (!r.valid) continue;
            if (!best.valid || r.objective_value < best.objective_value) best = r;
        }
    }

    std::map<std::string, std::string> mutate(const std::map<std::string, std::string>& g,
                                              std::mt19937& rng, double rate) {
        auto out = space.neighbor_configuration(g, rng, 0.3);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        for (auto& [k, v] : out) {
            if (uni(rng) < rate) v = space.random_configuration(rng).at(k);
        }
        return out;
    }

    std::map<std::string, std::string> crossover(
        const std::map<std::string, std::string>& a,
        const std::map<std::string, std::string>& b,
        std::mt19937& rng) {
        std::map<std::string, std::string> out;
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        for (const auto& [k, va] : a) {
            auto itb = b.find(k);
            out[k] = (itb != b.end() && uni(rng) < 0.5) ? va : (itb != b.end() ? itb->second : va);
        }
        return out;
    }
};

GeneticTuner::GeneticTuner(const ConfigurationSpace& space, const TuningConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->space = space;
    impl_->config = config;
}
GeneticTuner::~GeneticTuner() = default;

std::vector<TuningResult> GeneticTuner::tune(
    const std::function<TuningResult(const std::map<std::string, std::string>&)>& evaluate) {
    if (!evaluate) return {};

    std::mt19937 rng(0x6E37u);
    const size_t P = impl_->population_size;

    // Initial population
    std::vector<std::pair<std::map<std::string, std::string>, TuningResult>> pop;
    for (size_t i = 0; i < P; ++i) {
        auto cfg = impl_->space.random_configuration(rng);
        TuningResult r = evaluate(cfg);
        r.iteration = static_cast<int>(i);
        impl_->results.push_back(r);
        pop.emplace_back(cfg, r);
    }
    impl_->update_best();

    std::sort(pop.begin(), pop.end(),
              [](const auto& x, const auto& y) { return x.second.objective_value < y.second.objective_value; });

    int generation = 0;
    for (int it = static_cast<int>(P); it < impl_->config.max_iterations; ++it) {
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::vector<std::pair<std::map<std::string, std::string>, TuningResult>> next;

        // Elitism
        for (size_t i = 0; i < std::min<size_t>(4, pop.size()); ++i) next.push_back(pop[i]);

        while (next.size() < P) {
            const auto& pa = pop[static_cast<size_t>(uni(rng) * pop.size()) % pop.size()];
            const auto& pb = pop[static_cast<size_t>(uni(rng) * pop.size()) % pop.size()];
            auto child = impl_->crossover(pa.first, pb.first, rng);
            child = impl_->mutate(child, rng, 0.15);
            TuningResult r = evaluate(child);
            r.iteration = it;
            impl_->results.push_back(r);
            next.emplace_back(child, r);
        }

        std::sort(next.begin(), next.end(),
                  [](const auto& x, const auto& y) { return x.second.objective_value < y.second.objective_value; });
        pop = std::move(next);
        impl_->update_best();
        ++generation;

        if (impl_->config.verbose) {
            fprintf(stderr, "[ga-tuner] gen %3d  best=%.6g\n", generation, impl_->best.objective_value);
        }
    }

    return impl_->results;
}

const TuningResult& GeneticTuner::best_result() const { return impl_->best; }
const std::vector<TuningResult>& GeneticTuner::all_results() const { return impl_->results; }

// ---------------------------------------------------------------------------
// SimulatedAnnealingTuner
// ---------------------------------------------------------------------------

struct SimulatedAnnealingTuner::Impl {
    ConfigurationSpace space;
    TuningConfig config;
    std::vector<TuningResult> results;
    TuningResult best;
    TuningResult current;
};

SimulatedAnnealingTuner::SimulatedAnnealingTuner(const ConfigurationSpace& space,
                                                 const TuningConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->space = space;
    impl_->config = config;
}
SimulatedAnnealingTuner::~SimulatedAnnealingTuner() = default;

std::vector<TuningResult> SimulatedAnnealingTuner::tune(
    const std::function<TuningResult(const std::map<std::string, std::string>&)>& evaluate) {
    if (!evaluate) return {};

    std::mt19937 rng(0x5A4Du);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    auto cfg = impl_->space.random_configuration(rng);
    TuningResult r = evaluate(cfg);
    r.iteration = 0;
    impl_->current = r;
    impl_->best = r;
    impl_->results.push_back(r);

    // Initial temperature from the spread of a few random probes
    double t0 = 0.0;
    for (int i = 0; i < 5 && impl_->results.size() < 32; ++i) {
        auto c = impl_->space.random_configuration(rng);
        TuningResult p = evaluate(c);
        impl_->results.push_back(p);
        if (p.valid) t0 = std::max(t0, std::abs(p.objective_value));
    }
    if (t0 <= 0) t0 = 1.0;
    double temperature = t0;

    const double cooling = std::pow(1e-4, 1.0 / std::max(1, impl_->config.max_iterations));

    for (int it = 1; it < impl_->config.max_iterations; ++it) {
        auto neighbor = impl_->space.neighbor_configuration(impl_->current.configuration,
                                                            rng, 0.15);
        TuningResult cand = evaluate(neighbor);
        cand.iteration = it;
        impl_->results.push_back(cand);

        const double delta = cand.objective_value - impl_->current.objective_value;
        const bool accept = cand.valid &&
                            (delta < 0 || uni(rng) < std::exp(-delta / temperature));
        if (accept) impl_->current = cand;
        if (cand.valid && (!impl_->best.valid || cand.objective_value < impl_->best.objective_value)) {
            impl_->best = cand;
        }
        temperature *= cooling;
    }
    return impl_->results;
}

const TuningResult& SimulatedAnnealingTuner::best_result() const { return impl_->best; }
const std::vector<TuningResult>& SimulatedAnnealingTuner::all_results() const { return impl_->results; }

// ---------------------------------------------------------------------------
// Default search spaces
// ---------------------------------------------------------------------------

ConfigurationSpace create_default_spgemm_space(int num_gpus) {
    ConfigurationSpace space;

    {
        TuningParameter p;
        p.name = "algorithm";
        p.description = "SpGEMM kernel algorithm";
        p.constraint.type = ParameterType::ENUM;
        p.constraint.enum_values = {"hashmap", "heap", "merge_path", "galatic"};
        p.default_value = "hashmap";
        p.is_categorical = true;
        space.add_parameter(p);
    }
    {
        TuningParameter p;
        p.name = "threads_per_block";
        p.description = "CUDA block size";
        p.constraint.type = ParameterType::INT;
        p.constraint.min_int = 32;
        p.constraint.max_int = 1024;
        p.default_value = "256";
        space.add_parameter(p);
    }
    {
        TuningParameter p;
        p.name = "openmp_threads";
        p.description = "OpenMP thread count";
        p.constraint.type = ParameterType::INT;
        p.constraint.min_int = 1;
        p.constraint.max_int = 128;
        p.default_value = "0";
        space.add_parameter(p);
    }
    {
        TuningParameter p;
        p.name = "hashmap_size";
        p.description = "Initial per-row hashmap capacity";
        p.constraint.type = ParameterType::INT;
        p.constraint.min_int = 64;
        p.constraint.max_int = 65536;
        p.default_value = "1024";
        space.add_parameter(p);
    }
    {
        TuningParameter p;
        p.name = "load_factor";
        p.description = "Hashmap load factor";
        p.constraint.type = ParameterType::FLOAT;
        p.constraint.min_float = 1.0;
        p.constraint.max_float = 4.0;
        p.default_value = "2.0";
        space.add_parameter(p);
    }
    {
        TuningParameter p;
        p.name = "use_managed_memory";
        p.description = "Use CUDA managed (unified) memory";
        p.constraint.type = ParameterType::BOOL;
        p.default_value = "false";
        space.add_parameter(p);
    }
    {
        TuningParameter p;
        p.name = "remove_zeros";
        p.description = "Drop structurally-numerical zeros";
        p.constraint.type = ParameterType::BOOL;
        p.default_value = "true";
        space.add_parameter(p);
    }
    {
        TuningParameter p;
        p.name = "sm_clock_mhz";
        p.description = "Target GPU SM clock for DVFS";
        p.constraint.type = ParameterType::INT;
        p.constraint.min_int = 300;
        p.constraint.max_int = 1980;
        p.default_value = "1200";
        space.add_parameter(p);
    }
    if (num_gpus > 1) {
        TuningParameter p;
        p.name = "num_gpus";
        p.description = "Number of GPUs to use";
        p.constraint.type = ParameterType::INT;
        p.constraint.min_int = 1;
        p.constraint.max_int = num_gpus;
        p.default_value = "1";
        space.add_parameter(p);
    }

    return space;
}

ConfigurationSpace create_hashmap_space() {
    ConfigurationSpace space;
    {
        TuningParameter p;
        p.name = "hashmap_size";
        p.constraint.type = ParameterType::INT;
        p.constraint.min_int = 64; p.constraint.max_int = 65536;
        p.default_value = "1024";
        space.add_parameter(p);
    }
    {
        TuningParameter p;
        p.name = "load_factor";
        p.constraint.type = ParameterType::FLOAT;
        p.constraint.min_float = 1.0; p.constraint.max_float = 4.0;
        p.default_value = "2.0";
        space.add_parameter(p);
    }
    return space;
}

ConfigurationSpace create_galatic_space() {
    ConfigurationSpace space;
    {
        TuningParameter p;
        p.name = "threads_per_block";
        p.constraint.type = ParameterType::INT;
        p.constraint.min_int = 32; p.constraint.max_int = 1024;
        p.default_value = "256";
        space.add_parameter(p);
    }
    {
        TuningParameter p;
        p.name = "num_blocks_per_sm";
        p.constraint.type = ParameterType::INT;
        p.constraint.min_int = 1; p.constraint.max_int = 32;
        p.default_value = "4";
        space.add_parameter(p);
    }
    return space;
}

ConfigurationSpace create_merge_path_space() {
    ConfigurationSpace space;
    {
        TuningParameter p;
        p.name = "tile_size";
        p.constraint.type = ParameterType::INT;
        p.constraint.min_int = 32; p.constraint.max_int = 8192;
        p.default_value = "512";
        space.add_parameter(p);
    }
    {
        TuningParameter p;
        p.name = "openmp_threads";
        p.constraint.type = ParameterType::INT;
        p.constraint.min_int = 1; p.constraint.max_int = 128;
        p.default_value = "8";
        space.add_parameter(p);
    }
    return space;
}

ConfigurationSpace create_distributed_space(int num_ranks) {
    ConfigurationSpace space;
    {
        TuningParameter p;
        p.name = "grid_rows";
        p.description = "Process-grid rows";
        p.constraint.type = ParameterType::INT;
        p.constraint.min_int = 1; p.constraint.max_int = num_ranks;
        p.default_value = "1";
        space.add_parameter(p);
    }
    {
        TuningParameter p;
        p.name = "grid_layers";
        p.description = "2.5D replication factor";
        p.constraint.type = ParameterType::INT;
        p.constraint.min_int = 1; p.constraint.max_int = std::max(1, num_ranks / 2);
        p.default_value = "1";
        space.add_parameter(p);
    }
    {
        TuningParameter p;
        p.name = "comm_threshold";
        p.description = "Hybrid-comm crossover (bytes)";
        p.constraint.type = ParameterType::INT;
        p.constraint.min_int = 4096; p.constraint.max_int = 16 * 1024 * 1024;
        p.default_value = "65536";
        space.add_parameter(p);
    }
    {
        TuningParameter p;
        p.name = "hybrid_comm";
        p.constraint.type = ParameterType::BOOL;
        p.default_value = "true";
        space.add_parameter(p);
    }
    return space;
}

// ---------------------------------------------------------------------------
// RooflineModel
// ---------------------------------------------------------------------------

double RooflineModel::predict_performance(double arithmetic_intensity, int precision) const {
    const int p = std::clamp(precision, 0, 3);
    const double roof = std::min(compute_ceilings[p], arithmetic_intensity * memory_ceilings[p]);
    return roof;
}

RooflineModel::Bottleneck RooflineModel::identify_bottleneck(double arithmetic_intensity,
                                                            int precision) const {
    const int p = std::clamp(precision, 0, 3);
    if (arithmetic_intensity * memory_ceilings[p] < compute_ceilings[p]) {
        return Bottleneck::MEMORY;
    }
    return Bottleneck::COMPUTE;
}

RooflineModel RooflineModel::from_specs(double peak_flops, double peak_bandwidth) {
    RooflineModel m;
    m.peak_flops = peak_flops;
    m.peak_bandwidth = peak_bandwidth;
    // FP64 = FP32 / 2 for consumer parts, equal for datacentre parts; assume
    // a conservative datacentre ratio of 1:2:4:8.
    m.compute_ceilings[0] = peak_flops;
    m.compute_ceilings[1] = peak_flops;
    m.compute_ceilings[2] = peak_flops * 2.0;
    m.compute_ceilings[3] = peak_flops * 4.0;
    m.memory_ceilings[0] = peak_bandwidth;
    m.memory_ceilings[1] = peak_bandwidth;
    m.memory_ceilings[2] = peak_bandwidth * 1.5;
    m.memory_ceilings[3] = peak_bandwidth * 2.0;
    return m;
}

RooflineModel RooflineModel::from_device(int device_id) {
#if HPC_HAVE_CUDA
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
        const double gflops_fp32 =
            2.0 * prop.multiProcessorCount * prop.clockRate * 2.0 / 1e6; // rough
        const double bw = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
        return from_specs(gflops_fp32, bw);
    }
#else
    (void)device_id;
#endif
    return from_specs(1000.0, 100.0);
}

// ---------------------------------------------------------------------------
// AutoTuner
// ---------------------------------------------------------------------------

AutoTuner::AutoTuner(const TuningConfig& config) : config_(config) {}
AutoTuner::~AutoTuner() = default;

void AutoTuner::set_configuration_space(const ConfigurationSpace& space) {
    space_ = space;
}

namespace {

SpGEMMConfig config_from_map(const std::map<std::string, std::string>& cfg,
                             const SpGEMMConfig& base) {
    SpGEMMConfig out = base;

    auto get = [&](const char* key) -> const std::string* {
        auto it = cfg.find(key);
        return it == cfg.end() ? nullptr : &it->second;
    };

    if (const std::string* v = get("algorithm")) {
        if      (*v == "hashmap")   out.algorithm = SpGEMMAlgorithm::HASHMAP;
        else if (*v == "heap")      out.algorithm = SpGEMMAlgorithm::HEAP;
        else if (*v == "merge_path")out.algorithm = SpGEMMAlgorithm::MERGE_PATH;
        else if (*v == "galatic")   out.algorithm = SpGEMMAlgorithm::GALATIC;
        else if (*v == "cusparse")  out.algorithm = SpGEMMAlgorithm::CUSPARSE;
    }
    if (const std::string* v = get("threads_per_block"))
        out.cuda_threads_per_block = static_cast<int>(std::stoi(*v));
    if (const std::string* v = get("openmp_threads"))
        out.openmp_threads = static_cast<int>(std::stoi(*v));
    if (const std::string* v = get("hashmap_size"))
        out.initial_hashmap_size = static_cast<size_t>(std::stoul(*v));
    if (const std::string* v = get("load_factor"))
        out.hashmap_load_factor = static_cast<size_t>(std::stod(*v));
    if (const std::string* v = get("use_managed_memory"))
        out.use_managed_memory = (*v == "true" || *v == "1");
    if (const std::string* v = get("remove_zeros"))
        out.remove_zeros = (*v == "true" || *v == "1");
    if (const std::string* v = get("comm_threshold"))
        out.comm_threshold = static_cast<size_t>(std::stoul(*v));
    if (const std::string* v = get("hybrid_comm"))
        out.hybrid_comm = (*v == "true" || *v == "1");

    return out;
}

} // namespace

std::vector<TuningResult> AutoTuner::tune_spgemm(
    const SpGEMMDescriptor& descriptor,
    const std::function<SpGEMMStats(const SpGEMMConfig&)>& evaluate) {

    if (!evaluate) return {};
    if (space_.empty()) space_ = create_default_spgemm_space(descriptor.num_gpus);

    const SpGEMMConfig base = SpGEMM::recommend_config(descriptor);

    auto eval_fn = [&](const std::map<std::string, std::string>& cfg) -> TuningResult {
        TuningResult r;
        r.configuration = cfg;
        try {
            SpGEMMConfig c = config_from_map(cfg, base);
            r.stats = evaluate(c);
            r.objective_value = evaluate_objective(r.stats, EnergyStats{}, config_.objective);
            r.valid = true;
        } catch (const std::exception& e) {
            r.valid = false;
            r.error_message = e.what();
        }
        return r;
    };

    switch (config_.strategy) {
        case SearchStrategy::BAYESIAN: {
            bayesian_tuner_ = std::make_unique<BayesianTuner>(space_, config_);
            results_ = bayesian_tuner_->tune(eval_fn);
            best_result_ = bayesian_tuner_->best_result();
            break;
        }
        case SearchStrategy::GENETIC: {
            genetic_tuner_ = std::make_unique<GeneticTuner>(space_, config_);
            results_ = genetic_tuner_->tune(eval_fn);
            best_result_ = genetic_tuner_->best_result();
            break;
        }
        case SearchStrategy::SIMULATED_ANNEALING: {
            sa_tuner_ = std::make_unique<SimulatedAnnealingTuner>(space_, config_);
            results_ = sa_tuner_->tune(eval_fn);
            best_result_ = sa_tuner_->best_result();
            break;
        }
        case SearchStrategy::ML_BASED:
        case SearchStrategy::HYBRID:
        case SearchStrategy::RANDOM:
        default: {
            ml_tuner_ = std::make_unique<MLTuner>(space_, config_);
            results_ = ml_tuner_->tune(eval_fn);
            best_result_ = ml_tuner_->best_result();
            break;
        }
    }

    if (!config_.output_file.empty()) export_results(config_.output_file);
    return results_;
}

std::vector<TuningResult> AutoTuner::tune_spgemm_energy(
    const SpGEMMDescriptor& descriptor,
    const std::function<std::pair<SpGEMMStats, EnergyStats>(const SpGEMMConfig&)>& evaluate) {

    if (!evaluate) return {};
    if (space_.empty()) space_ = create_default_spgemm_space(descriptor.num_gpus);

    const SpGEMMConfig base = SpGEMM::recommend_config(descriptor);

    auto eval_fn = [&](const std::map<std::string, std::string>& cfg) -> TuningResult {
        TuningResult r;
        r.configuration = cfg;
        try {
            SpGEMMConfig c = config_from_map(cfg, base);
            auto [s, e] = evaluate(c);
            r.stats = s;
            r.energy_stats = e;
            r.objective_value = evaluate_objective(s, e, config_.objective);
            r.valid = true;
        } catch (const std::exception& ex) {
            r.valid = false;
            r.error_message = ex.what();
        }
        return r;
    };

    ml_tuner_ = std::make_unique<MLTuner>(space_, config_);
    results_ = ml_tuner_->tune(eval_fn);
    best_result_ = ml_tuner_->best_result();
    return results_;
}

SpGEMMConfig AutoTuner::best_config() const {
    return config_from_map(best_result_.configuration, SpGEMMConfig{});
}

void AutoTuner::export_results(const std::string& filename, const char* format) const {
    const std::string fmt = format ? format : "csv";
    if (fmt == "json") {
        std::ofstream f(filename);
        if (!f.is_open()) return;
        f << "[\n";
        for (size_t i = 0; i < results_.size(); ++i) {
            const auto& r = results_[i];
            f << "  {\"iteration\": " << r.iteration
              << ", \"objective\": " << r.objective_value
              << ", \"time_ms\": " << r.stats.total_time_ms
              << ", \"gflops\": " << r.stats.gflops
              << ", \"valid\": " << (r.valid ? "true" : "false")
              << ", \"config\": {";
            bool first = true;
            for (const auto& [k, v] : r.configuration) {
                if (!first) f << ", ";
                f << '"' << k << "\": \"" << v << '"';
                first = false;
            }
            f << "}}" << (i + 1 < results_.size() ? "," : "") << "\n";
        }
        f << "]\n";
        return;
    }

    std::ofstream f(filename);
    if (!f.is_open()) return;
    const auto names = space_.parameter_names();
    f << "iteration,objective,total_time_ms,gflops,nnz,valid";
    for (const auto& n : names) f << ',' << n;
    f << '\n';

    for (const auto& r : results_) {
        f << r.iteration << ',' << r.objective_value << ','
          << r.stats.total_time_ms << ',' << r.stats.gflops << ','
          << r.stats.output_nnz << ',' << (r.valid ? 1 : 0);
        for (const auto& n : names) {
            auto it = r.configuration.find(n);
            f << ',' << (it != r.configuration.end() ? it->second : std::string());
        }
        f << '\n';
    }
}

void AutoTuner::import_results(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open()) return;

    std::string line;
    if (!std::getline(f, line)) return;

    // Parse header to discover parameter columns
    std::vector<std::string> headers;
    {
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) headers.push_back(tok);
    }

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<std::string> toks;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) toks.push_back(tok);
        if (toks.size() < 4) continue;

        TuningResult r;
        try {
            r.iteration = std::stoi(toks[0]);
            r.objective_value = std::stod(toks[1]);
            r.stats.total_time_ms = std::stod(toks[2]);
            r.stats.gflops = std::stod(toks[3]);
            r.valid = true;
        } catch (...) { continue; }

        for (size_t i = 6; i < toks.size() && i < headers.size(); ++i) {
            r.configuration[headers[i]] = toks[i];
        }
        results_.push_back(r);
    }
}

HPC_NAMESPACE_END