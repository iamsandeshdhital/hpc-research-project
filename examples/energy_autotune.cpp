// energy_autotune - energy-aware autotuning of an SpGEMM configuration.
//
// Demonstrates the two energy-aware knobs this project exposes:
//
//   1. DVFS policy   - pick a clock/power point per phase (NVML on NVIDIA,
//                      CPU RAPL on Intel). Trades runtime for joules.
//   2. Configuration - use the ML-based tuner (PerfDojo-style: surrogate-guided
//                      search) with an energy-delay objective instead of a pure
//                      runtime objective.
//
// Reference: PerfDojo (ETH Zurich, SC'25) for the surrogate-guided search;
// EnergyUCB (2024) for treating energy as a first-class tuning dimension.

#include "hpc/hpc.hpp"

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

using namespace hpc;

namespace {

const char* flag(const char* s, const char* key) {
    const size_t kl = std::strlen(key);
    return std::strncmp(s, key, kl) == 0 ? s + kl : nullptr;
}

struct Trial {
    SpGEMMConfig config;
    double time_ms = 0.0;
    double energy_j = 0.0;
    double edp = 0.0;
};

// Evaluate one configuration, measuring both runtime and energy.
Trial run_trial(const std::unique_ptr<MatrixBase>& A,
                const std::unique_ptr<MatrixBase>& B,
                const SpGEMMConfig& cfg,
                EnergyManager& em,
                DVFSPolicy policy,
                int reps) {
    Trial t;
    t.config = cfg;

    DVFSConfig dvfs;
    dvfs.policy = policy;
    em.apply_dvfs_policy(dvfs);

    auto s = create_spgemm(cfg);
    if (s->set_matrix_a<>(A.get()) != 0 || s->set_matrix_b<>(B.get()) != 0) return t;

    // Warmup (excluded from both measurements).
    for (int i = 0; i < 2; ++i) s->compute();

    em.start_measurement("trial");

    BenchmarkConfig bc;
    bc.warmup_iterations = 0;
    bc.measurement_iterations = reps;
    bc.min_time_ms = 0;
    std::vector<double> times;
    run_benchmark([&]() { s->compute(); }, bc, times);

    const EnergyStats es = em.stop_measurement();

    std::sort(times.begin(), times.end());
    t.time_ms = times.empty() ? 0.0 : times[times.size() / 2];
    t.energy_j = es.total_energy_joules;
    t.edp = es.energy_delay_product();
    return t;
}

const char* algo_name(SpGEMMAlgorithm a) {
    switch (a) {
        case SpGEMMAlgorithm::HASHMAP:   return "hashmap";
        case SpGEMMAlgorithm::HEAP:      return "heap";
        case SpGEMMAlgorithm::MERGE_PATH:return "merge_path";
        case SpGEMMAlgorithm::GALATIC:   return "galatic";
        case SpGEMMAlgorithm::CUSPARSE:  return "cusparse";
        default:                         return "auto";
    }
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
        MPI_Finalize();
        return 0;
    }

    uint64_t n = 20000;
    double density = 1e-4;
    int reps = 5;
    int iters = 40;
    bool with_energy = true;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (const char* v = flag(a, "--n="))            n = std::strtoull(v, nullptr, 10);
        else if (const char* v = flag(a, "--density=")) density = std::atof(v);
        else if (const char* v = flag(a, "--reps="))    reps = std::atoi(v);
        else if (const char* v = flag(a, "--iters="))   iters = std::atoi(v);
        else if (std::strcmp(a, "--no-energy") == 0)    with_energy = false;
        else if (std::strcmp(a, "--help") == 0) {
            printf("Usage: %s [--n=N] [--density=P] [--reps=R] [--iters=K] "
                   "[--no-energy]\n", argv[0]);
            MPI_Finalize();
            return 0;
        }
    }

    printf("Energy-aware SpGEMM autotuning\n");
    printf("  n=%llu density=%.3g reps=%d iterations=%d\n",
           static_cast<unsigned long long>(n), density, reps, iters);

    const auto info = detect_hardware();
    print_hardware_info(info);

    EnergyManager em;
    EnergyConfig ec;
    ec.continuous_sampling = false;
    const bool have_energy =
        with_energy && em.initialize(ec) == static_cast<int>(ErrorCode::SUCCESS);

    if (with_energy && !have_energy) {
        printf("  no energy counters available: falling back to runtime-only "
               "tuning (EDP degenerates to runtime)\n");
    }

    auto A = generate_random_sparse(n, n, density, MatrixFormat::COO);
    auto B = generate_random_sparse(n, n, density, MatrixFormat::COO);

    SpGEMMDescriptor d;
    d.m = d.n = d.k = n;
    d.nnz_a = A->num_nonzeros();
    d.nnz_b = B->num_nonzeros();
    d.num_gpus = info.gpu_names.empty() ? 0 : 1;

    // ------------------------------------------------------------------
    // Stage 1: sweep the DVFS policies at the recommended configuration.
    // ------------------------------------------------------------------
    printf("\n=== stage 1: DVFS policy sweep ===\n");
    printf("%-20s %12s %12s %12s\n", "policy", "time[ms]", "energy[J]", "EDP[J*s]");
    printf("--------------------------------------------------------\n");

    SpGEMMConfig base = SpGEMM::recommend_config(d);
    base.use_cuda = false;   // keep this example runnable without a GPU

    const DVFSPolicy policies[] = {
        DVFSPolicy::PERFORMANCE, DVFSPolicy::BALANCED, DVFSPolicy::ENERGY_EFFICIENT
    };
    const char* policy_names[] = {"performance", "balanced", "energy_efficient"};

    std::vector<Trial> sweep;
    for (size_t i = 0; i < 3; ++i) {
        const Trial t = run_trial(A, B, base, em, policies[i], reps);
        sweep.push_back(t);
        printf("%-20s %12.3f %12.3f %12.4f\n", policy_names[i],
               t.time_ms, t.energy_j, t.edp);
    }

    size_t best_policy = 0;
    for (size_t i = 1; i < sweep.size(); ++i) {
        const double ref_edp = sweep[best_policy].energy_j > 0
            ? sweep[best_policy].edp : sweep[best_policy].time_ms;
        const double cur_edp = sweep[i].energy_j > 0 ? sweep[i].edp : sweep[i].time_ms;
        if (cur_edp < ref_edp) best_policy = i;
    }
    printf("  -> best policy for EDP: %s\n", policy_names[best_policy]);

    // ------------------------------------------------------------------
    // Stage 2: surrogate-guided configuration search under EDP.
    // ------------------------------------------------------------------
    printf("\n=== stage 2: autotuning %d configurations (objective: %s) ===\n",
           iters, have_energy ? "energy-delay product" : "runtime");

    TuningConfig tc;
    tc.strategy = SearchStrategy::ML_BASED;
    tc.objective = have_energy ? TuningObjective::MINIMIZE_EDP
                              : TuningObjective::MINIMIZE_TIME;
    tc.max_iterations = iters;
    tc.verbose = true;
    tc.output_file = "energy_autotune_results.csv";

    std::vector<Trial> log;
    AutoTuner tuner(tc);

    if (have_energy) {
        tuner.tune_spgemm_energy(
            d,
            [&](const SpGEMMConfig& c) -> std::pair<SpGEMMStats, EnergyStats> {
                SpGEMMConfig cc = c;
                cc.use_cuda = false;
                const Trial t = run_trial(A, B, cc, em, policies[best_policy], reps);
                log.push_back(t);

                SpGEMMStats s;
                s.total_time_ms = t.time_ms;
                s.compute_derived();
                EnergyStats e;
                e.total_energy_joules = t.energy_j;
                return {s, e};
            });
    } else {
        tuner.tune_spgemm(
            d,
            [&](const SpGEMMConfig& c) -> SpGEMMStats {
                SpGEMMConfig cc = c;
                cc.use_cuda = false;
                const Trial t = run_trial(A, B, cc, em, policies[best_policy], reps);
                log.push_back(t);
                SpGEMMStats s;
                s.total_time_ms = t.time_ms;
                s.compute_derived();
                return s;
            });
    }

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    printf("\n=== summary ===\n");
    if (!log.empty()) {
        const auto* fastest = &log[0];
        for (const auto& t : log) if (t.time_ms < fastest->time_ms) fastest = &t;

        const auto* leanest = &log[0];
        for (const auto& t : log) if (t.energy_j < leanest->energy_j) leanest = &t;

        printf("  configurations tried : %zu\n", log.size());
        printf("  fastest              : %.3f ms  (algo=%s, block=%d, omp=%d)\n",
               fastest->time_ms, algo_name(fastest->config.algorithm),
               fastest->config.cuda_threads_per_block, fastest->config.openmp_threads);
        if (have_energy) {
            printf("  lowest energy        : %.3f J   (algo=%s, block=%d, omp=%d)\n",
                   leanest->energy_j, algo_name(leanest->config.algorithm),
                   leanest->config.cuda_threads_per_block,
                   leanest->config.openmp_threads);
        }

        const SpGEMMConfig best = tuner.best_config();
        printf("  tuner choice         : algo=%s block=%d omp=%d hashmap=%zu\n",
               algo_name(best.algorithm), best.cuda_threads_per_block,
               best.openmp_threads, best.initial_hashmap_size);
        printf("\n  wrote %s\n", tc.output_file.c_str());
        printf("  NOTE: the tuner's objective is EDP, so its pick is neither the\n"
               "  fastest nor the leanest configuration. That is the point: it is\n"
               "  the best available trade-off for the objective you specified.\n");
    }

    MPI_Finalize();
    return 0;
}
