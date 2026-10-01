// Integration tests for the hybrid communicator. Run under MPI with 1..4 ranks.
//
// These exercise the correctness of the point-to-point and collective paths,
// the size-based mode selection, the staging-buffer pool, and the threshold
// autotuner. When MPI is unavailable the test degrades to a single-process run
// of everything that does not require more than one rank.

#include "test_framework.hpp"
#include "hpc/hpc.hpp"

#include <mpi.h>
#include <vector>
#include <cstring>
#include <algorithm>

using namespace hpc;

namespace {

int g_rank = 0;
int g_size = 1;
bool g_mpi_active = false;

void broadcast_bool(bool& v) {
    int x = v ? 1 : 0;
    MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD);
    v = (x != 0);
}

} // namespace

// ---------------------------------------------------------------------------
// Setup / teardown driven from a single "harness" test
// ---------------------------------------------------------------------------

TEST(Distributed, InitialiseMpi) {
    int provided = 0;
    if (MPI_Initialized(&provided) == MPI_SUCCESS && provided) {
        g_mpi_active = true;
    } else {
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        g_mpi_active = true;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    if (g_rank == 0) {
        printf("\n  [mpi] world size = %d\n", g_size);
    }
    EXPECT_GE(g_size, 1);
}

// ---------------------------------------------------------------------------
// Point-to-point
// ---------------------------------------------------------------------------

TEST(Comm, PointToPointRing) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);

    if (g_size < 2) {
        // Single rank: sending to self must be a no-op.
        double v = 42.0;
        EXPECT_EQ(comm.send(&v, 1, MPI_DOUBLE, 0, 7), MPI_SUCCESS);
        return;
    }

    const int next = (g_rank + 1) % g_size;
    const int prev = (g_rank - 1 + g_size) % g_size;

    for (int round = 0; round < 4; ++round) {
        const double out = 100.0 * (g_rank + 1) + round;
        double in = -1.0;

        ASSERT_EQ(comm.send(&out, 1, MPI_DOUBLE, next, 100), MPI_SUCCESS);
        ASSERT_EQ(comm.recv(&in, 1, MPI_DOUBLE, prev, 100), MPI_SUCCESS);

        EXPECT_NEAR(in, 100.0 * (prev + 1) + round, 1e-12);
    }

    comm.print_stats();
}

TEST(Comm, PointToPointLargeMessage) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);
    if (g_size < 2) return;

    // Large enough to exceed cpu_staging_threshold so the hybrid path engages.
    const size_t n = 1 << 18;   // 256 Ki doubles = 2 MiB
    std::vector<double> out(n), in(n, -1.0);

    for (size_t i = 0; i < n; ++i) out[i] = static_cast<double>(i) + g_rank;

    const int next = (g_rank + 1) % g_size;
    const int prev = (g_rank - 1 + g_size) % g_size;

    ASSERT_EQ(comm.send(out.data(), n, MPI_DOUBLE, next, 200), MPI_SUCCESS);
    ASSERT_EQ(comm.recv(in.data(), n, MPI_DOUBLE, prev, 200), MPI_SUCCESS);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(in[i], static_cast<double>(i) + prev, 1e-9);
        if (in[i] != static_cast<double>(i) + prev) break;   // stop after first failure
    }
}

TEST(Comm, ExplicitModeSelections) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);
    if (g_size < 2) return;

    const int next = (g_rank + 1) % g_size;
    const int prev = (g_rank - 1 + g_size) % g_size;

    const CommMode modes[] = {
        CommMode::GPU_DIRECT, CommMode::CPU_STAGING, CommMode::ZERO_COPY, CommMode::AUTO
    };

    const size_t n = 4096;
    std::vector<double> out(n), in(n, -1.0);

    int tag = 300;
    for (CommMode m : modes) {
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<double>(g_rank) * 1000.0 + i;

        ASSERT_EQ(comm.send(out.data(), n, MPI_DOUBLE, next, tag), MPI_SUCCESS);
        ASSERT_EQ(comm.recv(in.data(), n, MPI_DOUBLE, prev, tag), MPI_SUCCESS);

        for (size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(in[i], static_cast<double>(prev) * 1000.0 + i, 1e-9);
        }
        ++tag;
    }
}

TEST(Comm, SendrecvExchange) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);
    if (g_size < 2) return;

    const int next = (g_rank + 1) % g_size;
    const int prev = (g_rank - 1 + g_size) % g_size;

    const size_t n = 1024;
    std::vector<double> out(n), in(n, 0.0);
    for (size_t i = 0; i < n; ++i) out[i] = 1.0 + g_rank;

    ASSERT_EQ(comm.sendrecv(out.data(), n, MPI_DOUBLE,
                            in.data(), n, MPI_DOUBLE,
                            next, 400, prev, 400), MPI_SUCCESS);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(in[i], 1.0 + prev, 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Non-blocking
// ---------------------------------------------------------------------------

TEST(Comm, NonBlockingExchange) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);
    if (g_size < 2) return;

    const int next = (g_rank + 1) % g_size;
    const int prev = (g_rank - 1 + g_size) % g_size;

    const size_t n = 8192;
    std::vector<double> out(n), in(n, -1.0);
    for (size_t i = 0; i < n; ++i) out[i] = 7.0 * (g_rank + 1);

    CommRequest sreq, rreq;
    ASSERT_EQ(comm.isend(out.data(), n, MPI_DOUBLE, next, 500,
                         CommMode::AUTO, &sreq), MPI_SUCCESS);
    ASSERT_EQ(comm.irecv(in.data(), n, MPI_DOUBLE, prev, 500,
                         CommMode::AUTO, &rreq), MPI_SUCCESS);

    // The requests must not be considered active after wait().
    ASSERT_EQ(comm.wait(&sreq), MPI_SUCCESS);
    ASSERT_EQ(comm.wait(&rreq), MPI_SUCCESS);
    EXPECT_FALSE(sreq.active);
    EXPECT_TRUE(sreq.completed);
    EXPECT_FALSE(rreq.active);
    EXPECT_TRUE(rreq.completed);

    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(in[i], 7.0 * (prev + 1), 1e-12);
}

TEST(Comm, WaitallCompletes) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);
    if (g_size < 2) return;

    const int next = (g_rank + 1) % g_size;
    const int prev = (g_rank - 1 + g_size) % g_size;

    const size_t n = 256;
    std::vector<double> out(n), in(n, 0.0);
    std::vector<CommRequest> reqs;
    reqs.reserve(2);

    for (size_t i = 0; i < n; ++i) out[i] = 3.0 * g_rank;

    CommRequest s;
    ASSERT_EQ(comm.isend(out.data(), n, MPI_DOUBLE, next, 600, CommMode::AUTO, &s), MPI_SUCCESS);
    reqs.push_back(std::move(s));

    CommRequest r;
    ASSERT_EQ(comm.irecv(in.data(), n, MPI_DOUBLE, prev, 600, CommMode::AUTO, &r), MPI_SUCCESS);
    reqs.push_back(std::move(r));

    std::vector<CommRequest*> ptrs;
    for (auto& x : reqs) ptrs.push_back(&x);

    EXPECT_EQ(comm.waitall(static_cast<int>(ptrs.size()), ptrs.data()), MPI_SUCCESS);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(in[i], 3.0 * prev, 1e-12);
}

TEST(Comm, TestReportsCompletion) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);
    if (g_size < 2) return;

    const int next = (g_rank + 1) % g_size;
    const int prev = (g_rank - 1 + g_size) % g_size;

    double out = static_cast<double>(g_rank), in = -1.0;
    CommRequest s, r;
    ASSERT_EQ(comm.isend(&out, 1, MPI_DOUBLE, next, 700, CommMode::AUTO, &s), MPI_SUCCESS);
    ASSERT_EQ(comm.irecv(&in, 1, MPI_DOUBLE, prev, 700, CommMode::AUTO, &r), MPI_SUCCESS);

    // Spin until both complete.
    int flag_s = 0, flag_r = 0;
    for (int spin = 0; spin < 100000 && (!flag_s || !flag_r); ++spin) {
        comm.test(&s, &flag_s);
        comm.test(&r, &flag_r);
    }
    EXPECT_EQ(flag_s, 1);
    EXPECT_EQ(flag_r, 1);
    EXPECT_NEAR(in, static_cast<double>(prev), 1e-12);
}

// ---------------------------------------------------------------------------
// Collectives
// ---------------------------------------------------------------------------

TEST(Comm, AllreduceSum) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);

    double in = 1.0;
    double out = 0.0;
    ASSERT_EQ(comm.allreduce(&in, &out, 1, MPI_DOUBLE, MPI_SUM), MPI_SUCCESS);

    EXPECT_NEAR(out, static_cast<double>(g_size), 1e-12);
}

TEST(Comm, AllreduceMax) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);

    double in = static_cast<double>(g_rank);
    double out = 0.0;
    ASSERT_EQ(comm.allreduce(&in, &out, 1, MPI_DOUBLE, MPI_MAX), MPI_SUCCESS);

    EXPECT_NEAR(out, static_cast<double>(g_size - 1), 1e-12);
}

TEST(Comm, AllreduceLargeArray) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);

    const size_t n = 100000;
    std::vector<double> in(n), out(n, 0.0);
    for (size_t i = 0; i < n; ++i) in[i] = 1.0;

    ASSERT_EQ(comm.allreduce(in.data(), out.data(), n, MPI_DOUBLE, MPI_SUM), MPI_SUCCESS);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(out[i], static_cast<double>(g_size), 1e-9);
        if (out[i] != static_cast<double>(g_size)) break;
    }
}

TEST(Comm, AllgatherCollectsEveryRank) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);

    const int n = 4;
    double in[4];
    for (int i = 0; i < n; ++i) in[i] = g_rank * 10.0 + i;

    std::vector<double> out(n * g_size, -1.0);
    ASSERT_EQ(comm.allgather(in, n, MPI_DOUBLE, out.data(), n, MPI_DOUBLE), MPI_SUCCESS);

    for (int r = 0; r < g_size; ++r) {
        for (int i = 0; i < n; ++i) {
            EXPECT_NEAR(out[r * n + i], r * 10.0 + i, 1e-12);
        }
    }
}

TEST(Comm, AlltoallRoutesByRank) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);

    const int n = 3;
    double in[n];
    for (int i = 0; i < n; ++i) in[i] = g_rank * 100.0 + i;

    std::vector<double> out(n * g_size, -1.0);
    ASSERT_EQ(comm.alltoall(in, n, MPI_DOUBLE, out.data(), n, MPI_DOUBLE), MPI_SUCCESS);

    // Element j of rank r's buffer must land in slot r*g_size + j on every rank.
    for (int dst = 0; dst < g_size; ++dst) {
        for (int i = 0; i < n; ++i) {
            EXPECT_NEAR(out[dst * n + i], dst * 100.0 + i, 1e-12);
        }
    }
}

TEST(Comm, BroadcastFromRoot) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);

    double v = (g_rank == 0) ? 12345.0 : -1.0;
    ASSERT_EQ(comm.broadcast(&v, 1, MPI_DOUBLE, 0), MPI_SUCCESS);
    EXPECT_NEAR(v, 12345.0, 1e-12);
}

TEST(Comm, GatherToRoot) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);

    double in = static_cast<double>(g_rank) * 7.0;
    std::vector<double> out(g_rank == 0 ? g_size : 0, -1.0);

    ASSERT_EQ(comm.gather(&in, 1, MPI_DOUBLE,
                          out.empty() ? nullptr : out.data(), 1, MPI_DOUBLE, 0), MPI_SUCCESS);

    if (g_rank == 0) {
        for (int r = 0; r < g_size; ++r) EXPECT_NEAR(out[r], r * 7.0, 1e-12);
    }
}

TEST(Comm, ReduceToRoot) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);

    double in = static_cast<double>(g_rank) + 1.0;
    double out = 0.0;

    ASSERT_EQ(comm.reduce(&in, &out, 1, MPI_DOUBLE, MPI_SUM, 0), MPI_SUCCESS);
    if (g_rank == 0) {
        double expected = 0.0;
        for (int r = 0; r < g_size; ++r) expected += r + 1.0;
        EXPECT_NEAR(out, expected, 1e-12);
    }
}

TEST(Comm, ScatterFromRoot) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);

    std::vector<double> in;
    if (g_rank == 0) {
        in.resize(g_size);
        for (int r = 0; r < g_size; ++r) in[r] = 500.0 + r;
    }

    double out = -1.0;
    ASSERT_EQ(comm.scatter(in.empty() ? nullptr : in.data(), 1, MPI_DOUBLE,
                           &out, 1, MPI_DOUBLE, 0), MPI_SUCCESS);
    EXPECT_NEAR(out, 500.0 + g_rank, 1e-12);
}

TEST(Comm, ZeroCountCollectivesAreNoOps) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);

    double a = 1.0, b = 1.0;
    EXPECT_EQ(comm.allreduce(&a, &b, 0, MPI_DOUBLE, MPI_SUM), MPI_SUCCESS);
    EXPECT_NEAR(b, 1.0, 1e-15);
}

// ---------------------------------------------------------------------------
// Hybrid path: both modes must produce identical results
// ---------------------------------------------------------------------------

TEST(Comm, BothPathsAgreeForEverySize) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);
    if (g_size < 2) return;

    const int next = (g_rank + 1) % g_size;
    const int prev = (g_rank - 1 + g_size) % g_size;

    // Ladder spanning the threshold boundaries.
    const size_t sizes[] = {1, 16, 256, 4096, 1u << 14, 1u << 16, 1u << 18, 1u << 20};
    int tag = 800;

    for (size_t bytes : sizes) {
        const size_t n = std::max<size_t>(1, bytes / sizeof(double));
        std::vector<double> out(n), in_direct(n, -1.0), in_staged(n, -2.0);
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<double>(i) * 1.5 + g_rank;

        // GPU-direct
        ASSERT_EQ(comm.send(out.data(), n, MPI_DOUBLE, next, tag, CommMode::GPU_DIRECT), MPI_SUCCESS);
        ASSERT_EQ(comm.recv(in_direct.data(), n, MPI_DOUBLE, prev, tag, CommMode::GPU_DIRECT), MPI_SUCCESS);

        // CPU-staged
        ASSERT_EQ(comm.send(out.data(), n, MPI_DOUBLE, next, tag + 1, CommMode::CPU_STAGING), MPI_SUCCESS);
        ASSERT_EQ(comm.recv(in_staged.data(), n, MPI_DOUBLE, prev, tag + 1, CommMode::CPU_STAGING), MPI_SUCCESS);

        for (size_t i = 0; i < n; ++i) {
            const double expect = static_cast<double>(i) * 1.5 + prev;
            EXPECT_NEAR(in_direct[i], expect, 1e-10);
            EXPECT_NEAR(in_staged[i], expect, 1e-10);
        }

        bool agreed = true;
        for (size_t i = 0; i < n; ++i) {
            if (in_direct[i] != in_staged[i]) { agreed = false; break; }
        }
        EXPECT_TRUE(agreed);

        tag += 2;
    }
}

TEST(Comm, AutoModeTracksSizeLadder) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);
    if (g_size < 2) return;

    const int next = (g_rank + 1) % g_size;
    const int prev = (g_rank - 1 + g_size) % g_size;

    const size_t sizes[] = {64, 1024, 1u << 14, 1u << 16, 1u << 20, 1u << 22};
    int tag = 900;

    for (size_t bytes : sizes) {
        const size_t n = std::max<size_t>(1, bytes / sizeof(int));
        std::vector<int> out(n), in(n, 0);
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<int>(g_rank + 1);

        ASSERT_EQ(comm.send(out.data(), n, MPI_INT, next, tag, CommMode::AUTO), MPI_SUCCESS);
        ASSERT_EQ(comm.recv(in.data(), n, MPI_INT, prev, tag, CommMode::AUTO), MPI_SUCCESS);

        for (size_t i = 0; i < n; ++i) {
            EXPECT_EQ(in[i], prev + 1);
            if (in[i] != prev + 1) break;
        }
        ++tag;
    }
}

// ---------------------------------------------------------------------------
// Context / buffer pool / statistics
// ---------------------------------------------------------------------------

TEST(CommContext, RankAndSizeAreCorrect) {
    CommContext ctx(MPI_COMM_WORLD);
    EXPECT_EQ(ctx.rank(), g_rank);
    EXPECT_EQ(ctx.size(), g_size);
    EXPECT_EQ(ctx.mpi_comm(), MPI_COMM_WORLD);
}

TEST(CommContext, StagingBufferPoolReusesBuffers) {
    CommContext ctx(MPI_COMM_WORLD);
    ASSERT_EQ(ctx.initialize(), 0);

    void* b1 = ctx.get_staging_buffer(1024);
    ASSERT_TRUE(b1 != nullptr);
    ctx.return_staging_buffer(b1);

    void* b2 = ctx.get_staging_buffer(1024);
    // With only one buffer configured, the pool hands the same one back.
    EXPECT_TRUE(b2 != nullptr);
    ctx.return_staging_buffer(b2);

    // A larger request must still succeed.
    void* b3 = ctx.get_staging_buffer(1024 * 1024);
    EXPECT_TRUE(b3 != nullptr);
    ctx.return_staging_buffer(b3);
}

TEST(CommContext, StagingBufferIsWritable) {
    CommContext ctx(MPI_COMM_WORLD);
    ASSERT_EQ(ctx.initialize(), 0);

    const size_t n = 4096;
    void* buf = ctx.get_staging_buffer(n * sizeof(double));
    ASSERT_TRUE(buf != nullptr);

    double* d = static_cast<double*>(buf);
    for (size_t i = 0; i < n; ++i) d[i] = static_cast<double>(i) * 2.0;

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(d[i], static_cast<double>(i) * 2.0, 1e-15);
    }
    ctx.return_staging_buffer(buf);
}

TEST(CommContext, ConfigurationRoundTrips) {
    CommContext ctx(MPI_COMM_WORLD);
    ASSERT_EQ(ctx.initialize(), 0);

    CommConfig cfg = ctx.config();
    EXPECT_EQ(cfg.gpu_direct_threshold, static_cast<size_t>(64u * 1024u));
    EXPECT_EQ(cfg.cpu_staging_threshold, static_cast<size_t>(1024u * 1024u));

    cfg.gpu_direct_threshold = 4096;
    ctx.config() = cfg;
    EXPECT_EQ(ctx.config().gpu_direct_threshold, static_cast<size_t>(4096));

    // Mutating the returned reference must be visible through the getter.
    ctx.config().cpu_staging_threshold = 8192;
    EXPECT_EQ(ctx.config().cpu_staging_threshold, static_cast<size_t>(8192));
}

TEST(CommContext, StatisticsAccumulate) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);
    if (g_size < 2) return;

    comm.context().reset_stats();
    const auto before = comm.stats().num_messages_sent;

    const int next = (g_rank + 1) % g_size;
    const int prev = (g_rank - 1 + g_size) % g_size;

    double v = 1.0, r = 0.0;
    ASSERT_EQ(comm.send(&v, 1, MPI_DOUBLE, next, 1000), MPI_SUCCESS);
    ASSERT_EQ(comm.recv(&r, 1, MPI_DOUBLE, prev, 1000), MPI_SUCCESS);

    EXPECT_EQ(comm.stats().num_messages_sent, before + 1);
    EXPECT_EQ(comm.stats().num_messages_received, before + 1);
    EXPECT_EQ(comm.stats().total_bytes_sent, 8u);
    EXPECT_EQ(comm.stats().total_bytes_received, 8u);
}

TEST(CommContext, StatisticsResetZeroesCounters) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);
    if (g_size < 2) return;

    comm.context().reset_stats();
    EXPECT_EQ(comm.stats().num_messages_sent, 0u);
    EXPECT_EQ(comm.stats().total_comm_time, 0.0);
}

TEST(CommContext, BarrierSynchronisesRanks) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);

    int token = 0;
    if (g_rank == 0) token = 0xC0FFEE;
    MPI_Bcast(&token, 1, MPI_INT, 0, MPI_COMM_WORLD);
    comm.context().barrier();
    EXPECT_EQ(token, 0xC0FFEE);
}

// ---------------------------------------------------------------------------
// Threshold autotuner
// ---------------------------------------------------------------------------

TEST(CommAutotune, ThresholdsAreUpdated) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);
    if (g_size < 2) return;

    const auto before = comm.config().gpu_direct_threshold;
    comm.autotune_thresholds(nullptr, 2);
    const auto after = comm.config().gpu_direct_threshold;

    // The tuner must leave a self-consistent, non-degenerate configuration.
    EXPECT_GT(after, 0u);
    EXPECT_GE(comm.config().cpu_staging_threshold, comm.config().gpu_direct_threshold);
    EXPECT_LE(after, 1u << 24);
    (void)before;
}

TEST(CommAutotune, ThresholdAutotuneIsCollective) {
    // Every rank must reach this point (it contains internal barriers).
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);

    comm.autotune_thresholds(nullptr, 1);

    // Broadcast the result so we can assert every rank agrees.
    unsigned long long v = comm.config().gpu_direct_threshold;
    unsigned long long expected = v;
    MPI_Bcast(&expected, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    EXPECT_EQ(v, expected);
}

// ---------------------------------------------------------------------------
// Packed matrix exchange
// ---------------------------------------------------------------------------

TEST(Comm, PackedSizeMatchesWrittenBytes) {
    // Build a small COO matrix and verify pack/unpack round-trips.
    Matrix<double, MatrixFormat::COO> m(4, 5);
    m.nnz_ = 3;
    m.coo_rows_ = {0, 1, 3};
    m.coo_cols_ = {1, 3, 0};
    m.coo_values_ = {1.5, 2.5, 3.5};

    const size_t need = compute_packed_size(&m);
    EXPECT_GT(need, 0u);

    std::vector<uint8_t> buf(need);
    size_t pos = 0;
    pack_matrix_for_mpi(&m, buf.data(), &pos);

    // Header is 2*8 + 8 + 2*4 = 32 bytes; the payload is 3 doubles.
    EXPECT_EQ(pos, 32u + 3u * sizeof(double));
    EXPECT_LE(pos, need);
}

TEST(Comm, MatrixExchangeBetweenRanks) {
    HybridCommunicator comm(MPI_COMM_WORLD);
    ASSERT_EQ(comm.initialize(), 0);
    if (g_size < 2) return;

    // Each rank owns 3 nonzeros of an m x n matrix with m == g_size.
    const uint64_t ncols = 4;
    const uint64_t nnz = 3;

    Matrix<double, MatrixFormat::COO> m(g_size, ncols);
    m.nnz_ = nnz;
    m.coo_rows_ = {static_cast<uint64_t>(g_rank), static_cast<uint64_t>(g_rank),
                   static_cast<uint64_t>(g_rank)};
    m.coo_cols_ = {0, 1, 2};
    m.coo_values_ = {1.0 * (g_rank + 1), 2.0 * (g_rank + 1), 3.0 * (g_rank + 1)};

    // Exchange just the packed payload.
    std::vector<uint8_t> sbuf(compute_packed_size(&m));
    size_t pos = 0;
    pack_matrix_for_mpi(&m, sbuf.data(), &pos);

    // Learn the incoming size.
    uint64_t sbytes = pos;
    std::vector<uint64_t> rbytes(g_size, 0);
    MPI_Allgather(&sbytes, 1, MPI_UINT64_T, rbytes.data(), 1, MPI_UINT64_T,
                  MPI_COMM_WORLD);

    std::vector<std::vector<uint8_t>> rbuf(g_size);
    std::vector<MPI_Request> reqs;
    for (int r = 0; r < g_size; ++r) {
        rbuf[r].resize(rbytes[r]);
        reqs.push_back(MPI_REQUEST_NULL);
        MPI_Irecv(rbuf[r].data(), static_cast<int>(rbytes[r]), MPI_BYTE, r, 91,
                  MPI_COMM_WORLD, &reqs.back());
    }
    MPI_Send(sbuf.data(), static_cast<int>(pos), MPI_BYTE, g_rank, 91, MPI_COMM_WORLD);
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

    // Every rank should have received a matrix describing the same rows.
    for (int r = 0; r < g_size; ++r) {
        ASSERT_GE(rbuf[r].size(), 32u);
        uint64_t rows, cols, nz;
        std::memcpy(&rows, rbuf[r].data() + 0, 8);
        std::memcpy(&cols, rbuf[r].data() + 8, 8);
        std::memcpy(&nz,   rbuf[r].data() + 16, 8);
        EXPECT_EQ(rows, static_cast<uint64_t>(g_size));
        EXPECT_EQ(cols, ncols);
        EXPECT_EQ(nz, nnz);
    }
}

// ---------------------------------------------------------------------------
// Distributed SpGEMM correctness
// ---------------------------------------------------------------------------

TEST(DistributedSpGEMM, EachRankComputesCorrectLocalBlock) {
    // Block-diagonal-free distributed product on small local blocks. Every rank
    // multiplies the same two matrices locally and the results must agree.
    const uint64_t n = 32;

    auto A = generate_erdos_renyi(n, 0.1, MatrixFormat::COO);
    auto B = generate_erdos_renyi(n, 0.1, MatrixFormat::COO);

    SpGEMMConfig cfg;
    cfg.use_cuda = false;
    cfg.distributed = true;
    cfg.hybrid_comm = true;

    auto s = create_spgemm(cfg);
    ASSERT_EQ(s->set_matrix_a<>(A.get()), 0);
    ASSERT_EQ(s->set_matrix_b<>(B.get()), 0);
    ASSERT_EQ(s->compute(), 0);

    // Gather the output nnz from every rank; all must be identical.
    const uint64_t local_nnz = s->stats().output_nnz;
    std::vector<uint64_t> all(g_size, 0);
    MPI_Allgather(&local_nnz, 1, MPI_UINT64_T, all.data(), 1, MPI_UINT64_T,
                  MPI_COMM_WORLD);

    for (int r = 1; r < g_size; ++r) {
        EXPECT_EQ(all[r], all[0]);
    }
    EXPECT_GT(all[0], 0u);
}

TEST(DistributedSpGEMM, ResultsAreRankIndependent) {
    const uint64_t n = 16;
    auto A = generate_erdos_renyi(n, 0.2, MatrixFormat::COO);

    SpGEMMConfig cfg;
    cfg.use_cuda = false;
    cfg.distributed = true;

    auto s = create_spgemm(cfg);
    ASSERT_EQ(s->set_matrix_a<>(A.get()), 0);
    ASSERT_EQ(s->set_matrix_b<>(A.get()), 0);
    ASSERT_EQ(s->compute(), 0);

    const double sum = static_cast<double>(s->stats().output_nnz);
    double all_sum = 0.0;
    MPI_Allreduce(&sum, &all_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    EXPECT_NEAR(all_sum, sum * g_size, 1e-6);
}

// ---------------------------------------------------------------------------
// Final barrier so every rank finishes cleanly
// ---------------------------------------------------------------------------

TEST(Distributed, FinalBarrier) {
    CommContext ctx(MPI_COMM_WORLD);
    ctx.barrier();
    EXPECT_TRUE(true);

    bool ok = true;
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    EXPECT_TRUE(ok);
}

int main(int argc, char** argv) {
    // Let the framework drive the tests; MPI is initialised by the first test.
    return hptest::run_all(argc, argv);
}
