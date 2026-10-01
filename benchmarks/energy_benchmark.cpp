// energy_benchmark - measures energy and runtime across DVFS policies.
//
// Usage:
//   ./energy_benchmark [--n=20000] [--density=1e-4] [--reps=5]
//                      [--policies=performance,balanced,energy]
//
// Reports energy-to-solution, EDP and ED2P, which is the metric the
// energy-aware autotuner optimises (see EnergyUCB-style UCB work and the
// CEEC "Harvesting energy consumption on European HPC systems" guidance).

#include "hpc/hpc.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <algorithm>

using namespace hpc;

namespace {

struct Opts {
    uint64_t n = 20000;
    double density = 1e-4;
    int reps = 5;
    std::string policies = "performance,balanced,energy";
};

const char* flag(const char* s, const char* key) {
    const size_t kl = std::strlen(key);
    return std::strncmp(s, key, kl) == 0 ? s + kl : nullptr;
}

DVFSPolicy parse_policy(const std::string& s) {
    if (s == "performance" || s == "perf") return DVFSPolicy::PERFORMANCE;
    if (s == "energy" || s == "eco")     return DVFSPolicy::ENERGY_EFFICIENT;
    return DVFSPolicy::BALANCED;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == sep) {
            if (i > start) out.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

struct Result {
    std::string policy;
    double time_ms;
    double energy_j;
    double avg_w;
    double edp;
    double ed2p;
};

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Opts o;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (const char* v = flag(a, "--n="))         o.n = std::strtoull(v, nullptr, 10);
        else if (const char* v = flag(a, "--density=")) o.density = std::atof(v);
        else if (const char* v = flag(a, "--reps="))    o.reps = std::atoi(v);
        else if (const char* v = flag(a, "--policies=")) o.policies = v;
        else if (std::strcmp(a, "--help") == 0) {
            printf("Usage: %s [--n=N] [--density=P] [--reps=R] [--policies=list]\n", argv[0]);
            MPI_Finalize();
            return 0;
        }
    }

    printf("HPC Research Project - energy benchmark\n");
    printf("  n=%llu density=%.3g reps=%d\n",
           static_cast<unsigned long long>(o.n), o.density, o.reps);

    const auto info = detect_hardware();
    print_hardware_info(info);
    printf("  energy counters: NVML=%s RAPL=%s\n\n",
           info.has_nvml ? "yes" : "no", info.has_rapl ? "yes" : "no");

    auto A = generate_random_sparse(o.n, o.n, o.density, MatrixFormat::COO);

    SpGEMMDescriptor d;
    d.m = d.n = d.k = o.n;
    d.nnz_a = d.nnz_b = A->num_nonzeros();
    d.num_gpus = info.gpu_names.empty() ? 0 : 1;

    SpGEMMConfig cfg = SpGEMM::recommend_config(d);

    EnergyManager em;
    EnergyConfig ec;
    ec.continuous_sampling = false;
    const bool have_energy = (em.initialize(ec) == static_cast<int>(ErrorCode::SUCCESS));

    if (!have_energy) {
        printf("  NOTE: no energy counters available; reporting runtime only.\n\n");
    }

    printf("%-14s %12s %12s %10s %12s %12s\n",
           "policy", "time[ms]", "energy[J]", "avg[W]", "EDP[J*s]", "ED2P[J*s^2]");
    printf("--------------------------------------------------------------------------\n");

    std::vector<Result> results;
    for (const auto& name : split(o.policies, ',')) {
        if (rank != 0) break;   // only rank 0 drives the benchmark loop
        DVFSConfig dvfs;
        dvfs.policy = parse_policy(name);
        dvfs.performance_weight = 0.5;
        dvfs.energy_weight = 0.5;

        em.apply_dvfs_policy(dvfs);

        auto s = create_spgemm(cfg);
        s->set_matrix_a<>(A.get());
        s->set_matrix_b<>(A.get());

        // Warmup
        for (int i = 0; i < 2; ++i) s->compute();

        em.start_measurement(name.c_str());

        std::vector<double> times;
        BenchmarkConfig bc;
        bc.warmup_iterations = 0;
        bc.measurement_iterations = o.reps;
        bc.min_time_ms = 0;
        run_benchmark([&]() { s->compute(); }, bc, times);

        const EnergyStats es = em.stop_measurement();

        std::sort(times.begin(), times.end());
        const double t_ms = times.empty() ? 0.0 : times[times.size() / 2];
        const double t_s = t_ms / 1000.0;

        Result r;
        r.policy = name;
        r.time_ms = t_ms;
        r.energy_j = es.total_energy_joules;
        r.avg_w = es.avg_power_watts;
        r.edp = es.energy_delay_product();
        r.ed2p = es.energy_delay_squared_product();
        results.push_back(r);

        printf("%-14s %12.3f %12.3f %10.2f %12.4f %12.4f\n",
               r.policy.c_str(), r.time_ms, r.energy_j, r.avg_w, r.edp, r.ed2p);
    }

    printf("--------------------------------------------------------------------------\n");

    if (rank == 0) {
        if (!results.empty()) {
            const auto* best_time = &results[0];
            const auto* best_edp = &results[0];
            for (const auto& r : results) {
                if (r.time_ms < best_time->time_ms) best_time = &r;
                if (r.edp < best_edp->edp) best_edp = &r;
            }
            printf("  fastest      : %s (%.3f ms)\n",
                   best_time->policy.c_str(), best_time->time_ms);
            printf("  lowest EDP   : %s (%.4f J*s)\n",
                   best_edp->policy.c_str(), best_edp->edp);
            if (best_time->policy != best_edp->policy) {
                printf("  note: the energy-optimal policy is NOT the fastest one; "
                       "this is the expected trade-off.\n");
            }
        }

        std::ofstream f("energy_benchmark.csv");
        f << "policy,time_ms,energy_j,avg_power_w,edp,ed2p\n";
        for (const auto& r : results) {
            f << r.policy << ',' << r.time_ms << ',' << r.energy_j << ','
              << r.avg_w << ',' << r.edp << ',' << r.ed2p << '\n';
        }
        printf("\nwrote energy_benchmark.csv\n");
    }

    MPI_Finalize();
    return 0;
}
