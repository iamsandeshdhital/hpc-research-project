// comm_microbench - measures the two hybrid communication paths and finds the
// crossover point, then feeds the result back into the communicator's
// autotuner.
//
// This is the experiment behind the hybrid scheme in McFarland, Bellavita &
// Guidi (ICPE 2025): the faster path depends on message size, so a static
// choice is wrong for at least one end of the range.
//
// Usage:
//   mpirun -np 4 ./comm_microbench [--sizes=64,256,...,16777216] [--reps=200]

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

std::vector<size_t> parse_sizes(const std::string& s) {
    std::vector<size_t> out;
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == ',') {
            if (i > start) out.push_back(std::strtoull(s.substr(start, i - start).c_str(),
                                                       nullptr, 10));
            start = i + 1;
        }
    }
    return out;
}

double time_send_recv(HybridCommunicator& comm, void* buf, size_t bytes,
                      int dest, int source, int tag, CommMode mode) {
    // Warmup
    comm.send(buf, bytes, MPI_BYTE, dest, tag, mode);
    comm.recv(buf, bytes, MPI_BYTE, source, tag, mode);

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; ++i) {
        comm.send(buf, bytes, MPI_BYTE, dest, tag + i, mode);
        comm.recv(buf, bytes, MPI_BYTE, source, tag + i, mode);
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / 10.0;
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string size_list = "64,256,1024,4096,16384,65536,262144,1048576,4194304,16777216";
    int reps = 200;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (const char* v = flag(a, "--sizes=")) size_list = v;
        else if (const char* v = flag(a, "--reps=")) reps = std::atoi(v);
        else if (std::strcmp(a, "--help") == 0) {
            if (rank == 0) printf("Usage: mpirun -np N %s [--sizes=list] [--reps=R]\n", argv[0]);
            MPI_Finalize();
            return 0;
        }
    }
    (void)reps;

    const auto sizes = parse_sizes(size_list);

    if (rank == 0) {
        printf("HPC Research Project - hybrid communication microbenchmark\n");
        print_hardware_info(detect_hardware());
        printf("\n  ranks: %d\n", size);
        if (size < 2) {
            printf("  NOTE: fewer than 2 ranks; measuring self-send overhead only.\n\n");
        }
        printf("%12s %14s %14s %12s %10s\n",
               "bytes", "direct[us]", "staged[us]", "winner", "ratio");
        printf("------------------------------------------------------------\n");
    }

    HybridCommunicator comm(MPI_COMM_WORLD);
    comm.initialize();

    const int next = (rank + 1) % size;
    const int prev = (rank - 1 + size) % size;

    const size_t max_bytes = sizes.empty() ? 4096 : sizes.back();
    std::vector<uint8_t> buf(std::max<size_t>(max_bytes, 1), 0xAB);

    int tag = 10000;
    size_t crossover_direct = 0;   // largest size where direct wins
    size_t crossover_staged = 0;    // smallest size where staging wins

    for (size_t bytes : sizes) {
        const double t_direct = time_send_recv(comm, buf.data(), bytes, next, prev,
                                               tag, CommMode::GPU_DIRECT);
        const double t_staged = time_send_recv(comm, buf.data(), bytes, next, prev,
                                              tag, CommMode::CPU_STAGING);
        tag += 100;

        const bool direct_wins = t_direct <= t_staged;
        const double ratio = t_staged > 0.0 ? t_direct / t_staged : 0.0;

        if (direct_wins) crossover_direct = bytes;
        else if (crossover_staged == 0) crossover_staged = bytes;

        if (rank == 0) {
            printf("%12zu %14.2f %14.2f %12s %10.3f\n",
                   bytes, t_direct, t_staged, direct_wins ? "direct" : "staged", ratio);
        }
    }

    // Broadcast the crossovers so every rank agrees on the new thresholds.
    unsigned long long c[2] = {static_cast<unsigned long long>(crossover_direct),
                               static_cast<unsigned long long>(crossover_staged)};
    MPI_Bcast(c, 2, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("------------------------------------------------------------\n");
        printf("  crossover: direct wins up to %llu B, staging wins from %llu B\n",
               c[0], c[1]);
        printf("  recommend: gpu_direct_threshold=%llu, cpu_staging_threshold=%llu\n",
               c[0], c[1]);
    }

    // Apply the measured thresholds and verify the auto path now tracks them.
    comm.config().gpu_direct_threshold = static_cast<size_t>(c[0]);
    comm.config().cpu_staging_threshold = static_cast<size_t>(c[1] ? c[1] : c[0] + 1);

    if (rank == 0) {
        printf("\n  verifying AUTO selection against the measured thresholds:\n");
        printf("%12s %14s %12s\n", "bytes", "auto[us]", "expected");
        printf("------------------------------------------------------------\n");
    }

    tag = 20000;
    for (size_t bytes : sizes) {
        const double t_auto = time_send_recv(comm, buf.data(), bytes, next, prev,
                                             tag, CommMode::AUTO);
        tag += 100;
        const char* expected = bytes <= c[0] ? "direct" : "staged";
        if (rank == 0) {
            printf("%12zu %14.2f %12s\n", bytes, t_auto, expected);
        }
    }

    // Built-in autotuner (does its own sweep).
    comm.autotune_thresholds(nullptr, 3);
    if (rank == 0) {
        printf("\n  autotuner result: gpu_direct<=%zu B, cpu_staging>=%zu B\n",
               comm.config().gpu_direct_threshold,
               comm.config().cpu_staging_threshold);

        std::ofstream f("comm_microbench.csv");
        f << "bytes,direct_us,staged_us,ratio,winner\n";
        tag = 30000;
        for (size_t bytes : sizes) {
            const double d = time_send_recv(comm, buf.data(), bytes, next, prev,
                                            tag, CommMode::GPU_DIRECT);
            const double s = time_send_recv(comm, buf.data(), bytes, next, prev,
                                            tag, CommMode::CPU_STAGING);
            tag += 100;
            f << bytes << ',' << d << ',' << s << ',' << (s > 0 ? d / s : 0.0) << ','
              << (d <= s ? "direct" : "staged") << '\n';
        }
        printf("  wrote comm_microbench.csv\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
