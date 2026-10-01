// roofline - locates a kernel on the Roofline model and reports whether it is
// compute- or memory-bound.
//
// Usage:
//   ./roofline [--flops=10000] [--bandwidth=1000] [--n=10000] [--density=1e-4]
//
// The arithmetic intensity that matters for SpGEMM is not flops/byte but
// (multiply + add operations) / (index + value bytes moved), because the
// kernel is dominated by irregular index traffic rather than by FLOPs. This
// tool reports both so the comparison is honest.

#include "hpc/hpc.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

using namespace hpc;

namespace {

const char* flag(const char* s, const char* key) {
    const size_t kl = std::strlen(key);
    return std::strncmp(s, key, kl) == 0 ? s + kl : nullptr;
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

    double flops = 0.0, bandwidth = 0.0;
    uint64_t n = 10000;
    double density = 1e-4;
    bool use_device = false;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (const char* v = flag(a, "--flops="))      flops = std::atof(v);
        else if (const char* v = flag(a, "--bandwidth=")) bandwidth = std::atof(v);
        else if (const char* v = flag(a, "--n="))      n = std::strtoull(v, nullptr, 10);
        else if (const char* v = flag(a, "--density=")) density = std::atof(v);
        else if (std::strcmp(a, "--device") == 0)     use_device = true;
        else if (std::strcmp(a, "--help") == 0) {
            printf("Usage: %s [--flops=F] [--bandwidth=B] [--n=N] [--density=P] [--device]\n",
                   argv[0]);
            MPI_Finalize();
            return 0;
        }
    }

    RooflineModel roof;
    if (use_device || (flops <= 0.0 || bandwidth <= 0.0)) {
        roof = RooflineModel::from_device(0);
        printf("using device-queried roofline:\n");
        printf("  peak FP32   : %.2f GFLOP/s\n", roof.peak_flops);
        printf("  peak memory : %.2f GiB/s\n", roof.peak_bandwidth);
    } else {
        roof = RooflineModel::from_specs(flops, bandwidth);
        printf("using user-supplied roofline:\n");
        printf("  peak FP32   : %.2f GFLOP/s\n", roof.peak_flops);
        printf("  peak memory : %.2f GiB/s\n", roof.peak_bandwidth);
    }

    const double ridge = roof.peak_flops / std::max(1e-12, roof.peak_bandwidth);
    printf("  ridge point : %.3f FLOP/byte\n\n", ridge);

    // Measure the actual kernel.
    auto A = generate_random_sparse(n, n, density, MatrixFormat::COO);

    SpGEMMDescriptor d;
    d.m = d.n = d.k = n;
    d.nnz_a = d.nnz_b = A->num_nonzeros();

    SpGEMMConfig cfg = SpGEMM::recommend_config(d);
    cfg.use_cuda = false;

    auto s = create_spgemm(cfg);
    s->set_matrix_a<>(A.get());
    s->set_matrix_b<>(A.get());
    for (int i = 0; i < 2; ++i) s->compute();

    BenchmarkConfig bc;
    bc.warmup_iterations = 1;
    bc.measurement_iterations = 10;
    bc.min_time_ms = 50;
    std::vector<double> times;
    run_benchmark([&]() { s->compute(); }, bc, times);

    std::sort(times.begin(), times.end());
    const double t_ms = times.empty() ? 0.0 : times[times.size() / 2];
    const auto& st = s->stats();

    // Effective arithmetic intensity as the framework reports it (bytes/flop).
    const double ai_from_stats = st.flops > 0
        ? static_cast<double>(st.memory_bytes) / static_cast<double>(st.flops)
        : 0.0;
    const double flops_per_byte = ai_from_stats > 0 ? 1.0 / ai_from_stats : 0.0;

    const double attainable = roof.predict_performance(flops_per_byte, 1);
    const auto bottleneck = roof.identify_bottleneck(flops_per_byte, 1);
    const double achieved = t_ms > 0 ? st.gflops : 0.0;
    const double efficiency = attainable > 0 ? achieved / attainable : 0.0;

    printf("SpGEMM measurement (n=%llu, density=%.3g)\n",
           static_cast<unsigned long long>(n), density);
    printf("  nnz(A), nnz(B)      : %llu, %llu\n",
           static_cast<unsigned long long>(st.input_nnz_a),
           static_cast<unsigned long long>(st.input_nnz_b));
    printf("  nnz(C)              : %llu\n", static_cast<unsigned long long>(st.output_nnz));
    printf("  runtime             : %.3f ms\n", t_ms);
    printf("  achieved            : %.3f GFLOP/s\n", achieved);
    printf("  memory bandwidth    : %.3f GiB/s\n", st.memory_bandwidth_gb_s);
    printf("  arithmetic intensity: %.6f FLOP/byte\n", flops_per_byte);
    printf("  roofline ceiling    : %.3f GFLOP/s\n", attainable);
    printf("  bottleneck          : %s\n",
           bottleneck == RooflineModel::Bottleneck::COMPUTE ? "compute"
           : bottleneck == RooflineModel::Bottleneck::MEMORY ? "memory"
           : "balanced");
    printf("  roofline efficiency : %.2f%%\n", efficiency * 100.0);

    if (bottleneck == RooflineModel::Bottleneck::MEMORY && efficiency < 0.30) {
        printf("\n  diagnosis: memory-bound and far from the ceiling. For sparse\n"
               "  products the usual causes are (a) index traffic - 64-bit\n"
               "  indices cost 2x a 32-bit index - and (b) uncoalesced gathers\n"
               "  from B's rows. Try 32-bit indices and a block size near 256.\n");
    } else if (bottleneck == RooflineModel::Bottleneck::COMPUTE && efficiency < 0.30) {
        printf("\n  diagnosis: compute-bound but well below the ceiling, which\n"
               "  usually means the semiring is not amenable to the hardware\n"
               "  multiply-add. Check whether a specialised kernel is available.\n");
    } else if (efficiency >= 0.60) {
        printf("\n  the kernel is running close to its theoretical limit; further\n"
               "  gains need algorithmic change, not tuning.\n");
    }

    MPI_Finalize();
    return 0;
}
