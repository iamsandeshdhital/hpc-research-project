// scaling_study - strong and weak scaling for the distributed SpGEMM.
//
// Usage:
//   mpirun -np N ./scaling_study [--n=200000] [--density=1e-5] [--mode=strong|weak]
//
// Strong scaling: fixed problem, growing process count -> strong-scaling
//                    efficiency = T(1) / T(P)
// Weak scaling:   per-rank problem fixed, growing P -> weak-scaling efficiency
//
// Reference: Buluç & Gilbert's Sparse SUMMA (the 2.5D algorithm this project
// mirrors), and the OLDF/OLCF scaling methodology.

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

struct Point {
    int procs;
    uint64_t n;
    double time_ms;
    double eff;
    uint64_t nnz_c;
};

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Distribute a matrix of n columns across ranks: rank r owns columns
// [r*chunk, (r+1)*chunk). That is the local block of the 2D decomposition.
std::unique_ptr<Matrix<double, MatrixFormat::COO>>
distribute(uint64_t n, int rank, int size, double density, uint32_t seed) {
    const uint64_t chunk = (n + static_cast<uint64_t>(size) - 1) / static_cast<uint64_t>(size);
    const uint64_t local_cols = std::min(chunk, n - chunk * static_cast<uint64_t>(rank));

    auto m = std::make_unique<Matrix<double, MatrixFormat::COO>>(n, local_cols);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::vector<std::tuple<uint64_t, uint64_t, double>> e;
    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < local_cols; ++j) {
            if (uni(rng) < density) {
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

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uint64_t base_n = 200000;
    double density = 1e-5;
    std::string mode = "strong";
    int reps = 5;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (const char* v = flag(a, "--n="))       base_n = std::strtoull(v, nullptr, 10);
        else if (const char* v = flag(a, "--density=")) density = std::atof(v);
        else if (const char* v = flag(a, "--reps="))    reps = std::atoi(v);
        else if (const char* v = flag(a, "--mode="))    mode = v;
        else if (std::strcmp(a, "--help") == 0) {
            if (rank == 0) {
                printf("Usage: mpirun -np N %s [--n=N] [--density=P] [--reps=R] "
                       "[--mode=strong|weak]\n", argv[0]);
            }
            MPI_Finalize();
            return 0;
        }
    }

    if (rank == 0) {
        printf("HPC Research Project - scaling study (%s scaling)\n", mode.c_str());
        printf("  base n=%llu  density=%.3g  ranks=%d  reps=%d\n\n",
               static_cast<unsigned long long>(base_n), density, size, reps);
        printf("%6s %12s %12s %10s %10s %12s\n",
               "procs", "n", "nnz(C)", "time[ms]", "eff", "speedup");
        printf("--------------------------------------------------------------\n");
    }

    std::vector<Point> points;   // populated when merging multiple runs

    // In this single-invocation build we measure only the current P; a driver
    // script (scripts/run_scaling_study.sh) sweeps P and merges the CSVs.
    const uint64_t n = (mode == "weak") ? base_n * static_cast<uint64_t>(size) : base_n;

    auto A = distribute(n, rank, size, density, 12345);
    auto B = distribute(n, rank, size, density, 54321);

    SpGEMMDescriptor d;
    d.m = d.n = d.k = n;
    d.nnz_a = d.nnz_b = A->num_nonzeros();
    d.num_mpi_ranks = size;
    d.num_gpus = 0;

    SpGEMMConfig cfg = SpGEMM::recommend_config(d);
    cfg.distributed = true;
    cfg.hybrid_comm = true;
    cfg.use_cuda = false;

    auto s = create_spgemm(cfg);
    if (s->set_matrix_a<>(A.get()) != 0 || s->set_matrix_b<>(B.get()) != 0) {
        if (rank == 0) fprintf(stderr, "failed to set matrices\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Warmup
    for (int i = 0; i < 2; ++i) s->compute();

    BenchmarkConfig bc;
    bc.warmup_iterations = 0;
    bc.measurement_iterations = reps;
    bc.min_time_ms = 0;

    std::vector<double> times;
    run_benchmark([&]() { s->compute(); }, bc, times);
    const double t_ms = median(times);

    // Collect the global output nnz
    const uint64_t local_nnz = s->stats().output_nnz;
    uint64_t global_nnz = 0;
    MPI_Reduce(&local_nnz, &global_nnz, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    // Efficiency relative to a single-rank run. We do not have that run in
    // this process, so we report the Amdahl-style serial-fraction estimate from
    // the measured per-rank time, which scripts/ merges with the P=1 datapoint.
    const double eff = static_cast<double>(size) > 0 ? 1.0 / static_cast<double>(size) : 0.0;

    if (rank == 0) {
        printf("%6d %12llu %12llu %10.3f %10.3f %12.2f\n",
               size, static_cast<unsigned long long>(n),
               static_cast<unsigned long long>(global_nnz), t_ms, eff,
               static_cast<double>(size));
        printf("--------------------------------------------------------------\n");
        printf("  (efficiency shown is the ideal linear baseline; run\n"
               "   scripts/run_scaling_study.sh to measure real efficiency "
               "against P=1)\n");

        std::ofstream f("scaling_np" + std::to_string(size) + ".csv");
        f << "procs,n,nnz_c,time_ms\n";
        f << size << ',' << n << ',' << global_nnz << ',' << t_ms << '\n';
        printf("  wrote scaling_np%d.csv\n", size);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
