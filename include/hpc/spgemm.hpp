#pragma once

#include "config.hpp"
#include "matrix.hpp"
#include "semiring.hpp"
#include <cstdint>
#include <vector>
#include <memory>
#include <functional>

HPC_NAMESPACE_BEGIN

// SpGEMM algorithm types
enum class SpGEMMAlgorithm : uint8_t {
    AUTO = 0,
    HASHMAP = 1,          // Hashmap-based (Gustavson)
    HEAP = 2,             // Heap-based
    MERGE_PATH = 3,       // Merge path
    GALATIC = 4,          // GALATIC-style GPU
    CUSPARSE = 5,         // cuSPARSE
    ROCSPARSE = 6,        // rocSPARSE
    MKL = 7,              // Intel MKL
    CUSTOM = 8
};

// SpGEMM phase types
enum class SpGEMMPhase : uint8_t {
    SYMBOLIC = 0,         // Symbolic phase (structure only)
    NUMERIC = 1,          // Numeric phase (values)
    FUSED = 2             // Fused symbolic + numeric
};

// SpGEMM configuration
struct HPC_API SpGEMMConfig {
    SpGEMMAlgorithm algorithm = SpGEMMAlgorithm::AUTO;
    SpGEMMPhase phase = SpGEMMPhase::FUSED;

    // Parallelism
    bool use_openmp = true;
    int openmp_threads = 0;  // 0 = auto
    bool use_cuda = true;
    int cuda_device = 0;
    int cuda_blocks = 0;     // 0 = auto
    int cuda_threads_per_block = 256;

    // Memory management
    size_t hashmap_load_factor = 2;
    size_t initial_hashmap_size = 1024;
    bool use_managed_memory = false;
    bool use_async_copy = true;

    // Optimization
    bool sort_output = true;
    bool remove_zeros = true;
    double zero_threshold = 1e-12;
    bool transpose_b = false;

    // Distributed
    bool distributed = false;
    bool hybrid_comm = true;
    size_t comm_threshold = 64 * 1024;

    // Profiling
    bool enable_profiling = true;
    bool detailed_timing = false;
};

// SpGEMM statistics
struct HPC_API SpGEMMStats {
    double symbolic_time_ms = 0.0;
    double numeric_time_ms = 0.0;
    double total_time_ms = 0.0;
    double gpu_kernel_time_ms = 0.0;
    double cpu_time_ms = 0.0;
    double comm_time_ms = 0.0;

    uint64_t input_nnz_a = 0;
    uint64_t input_nnz_b = 0;
    uint64_t output_nnz = 0;
    uint64_t flops = 0;
    uint64_t memory_bytes = 0;

    double gflops = 0.0;
    double memory_bandwidth_gb_s = 0.0;
    double arithmetic_intensity = 0.0;

    uint64_t hashmap_collisions = 0;
    uint64_t heap_operations = 0;
    uint32_t max_hashmap_size = 0;

    void reset() {
        *this = SpGEMMStats{};
    }

    void compute_derived() {
        if (total_time_ms > 0) {
            gflops = (flops / 1e9) / (total_time_ms / 1000.0);
            memory_bandwidth_gb_s = (memory_bytes / 1e9) / (total_time_ms / 1000.0);
        }
        if (flops > 0) {
            arithmetic_intensity = static_cast<double>(memory_bytes) / static_cast<double>(flops);
        }
    }
};

// SpGEMM descriptor for autotuning
struct HPC_API SpGEMMDescriptor {
    uint64_t m = 0, n = 0, k = 0;
    uint64_t nnz_a = 0, nnz_b = 0;
    double density_a = 0.0, density_b = 0.0;
    MatrixFormat format_a = MatrixFormat::CSR;
    MatrixFormat format_b = MatrixFormat::CSR;
    MatrixFormat format_c = MatrixFormat::CSR;
    SemiringID semiring = SemiringType::PLUS_TIMES_FLOAT;
    bool trans_a = false, trans_b = false;
    int num_gpus = 1;
    int num_mpi_ranks = 1;
};

// Forward declarations
template<typename T, typename SemiringT>
class SpGEMMKernel;

// Main SpGEMM interface
class HPC_API SpGEMM {
public:
    SpGEMM();
    ~SpGEMM();

    // Initialize with configuration
    int initialize(const SpGEMMConfig& config = SpGEMMConfig{});

    // Set matrices
    template<typename T>
    int set_matrix_a(const MatrixBase* matrix);

    template<typename T>
    int set_matrix_b(const MatrixBase* matrix);

    template<typename T>
    int set_matrix_c(MatrixBase* matrix);

    // Set semiring
    int set_semiring(SemiringID semiring_id);

    // Compute SpGEMM
    int compute();

    // Compute with explicit phase control
    int compute_symbolic();
    int compute_numeric();

    // Get statistics
    const SpGEMMStats& stats() const { return stats_; }
    SpGEMMStats& stats() { return stats_; }
    void reset_stats() { stats_.reset(); }

    // Autotuning
    int autotune(const SpGEMMDescriptor& desc, int iterations = 5);

    // Get recommended configuration
    static SpGEMMConfig recommend_config(const SpGEMMDescriptor& desc);

    // Kernel selection
    static SpGEMMAlgorithm select_algorithm(const SpGEMMDescriptor& desc);

    // Memory estimation
    static size_t estimate_memory(const SpGEMMDescriptor& desc);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    SpGEMMConfig config_;
    SpGEMMStats stats_;
    SemiringID semiring_id_ = SemiringType::PLUS_TIMES_FLOAT;
};

// GPU Kernels for SpGEMM
namespace kernel {

// Kernel configuration
struct KernelConfig {
    int block_size = 256;
    int grid_size = 0;
    size_t shared_memory = 0;
    bool use_tensor_cores = false;
    int warp_per_row = 1;
};

// Symbolic phase kernel (structure computation)
template<typename IndexT>
HPC_GLOBAL void spgemm_symbolic_kernel(
    const IndexT* __restrict__ A_row_ptr,
    const IndexT* __restrict__ A_col_idx,
    const IndexT* __restrict__ B_row_ptr,
    const IndexT* __restrict__ B_col_idx,
    IndexT* __restrict__ C_row_ptr,
    IndexT* __restrict__ C_col_idx,
    IndexT m, IndexT n, IndexT k,
    IndexT* workspace,
    size_t workspace_size
);

// Numeric phase kernel (value computation)
template<typename IndexT, typename ValueT, typename SemiringT>
HPC_GLOBAL void spgemm_numeric_kernel(
    const IndexT* __restrict__ A_row_ptr,
    const IndexT* __restrict__ A_col_idx,
    const ValueT* __restrict__ A_values,
    const IndexT* __restrict__ B_row_ptr,
    const IndexT* __restrict__ B_col_idx,
    const ValueT* __restrict__ B_values,
    IndexT* __restrict__ C_row_ptr,
    IndexT* __restrict__ C_col_idx,
    ValueT* __restrict__ C_values,
    IndexT m, IndexT n, IndexT k,
    SemiringT semiring,
    ValueT* workspace,
    size_t workspace_size
);

// GALATIC-style 4-stage kernel
template<typename IndexT, typename ValueT, typename SemiringT>
HPC_GLOBAL void galatic_spgemm_stage1_expand(
    const IndexT* __restrict__ A_row_ptr,
    const IndexT* __restrict__ A_col_idx,
    const ValueT* __restrict__ A_values,
    const IndexT* __restrict__ B_row_ptr,
    const IndexT* __restrict__ B_col_idx,
    const ValueT* __restrict__ B_values,
    IndexT* __restrict__ C_row_ptr,
    IndexT* __restrict__ C_col_idx,
    ValueT* __restrict__ C_values,
    IndexT* workspace,
    IndexT m, IndexT n, IndexT k,
    SemiringT semiring
);

template<typename IndexT, typename ValueT>
HPC_GLOBAL void galatic_spgemm_stage2_sort(
    IndexT* __restrict__ C_col_idx,
    ValueT* __restrict__ C_values,
    IndexT* workspace,
    IndexT nnz
);

template<typename IndexT, typename ValueT, typename SemiringT>
HPC_GLOBAL void galatic_spgemm_stage3_combine(
    IndexT* __restrict__ C_col_idx,
    ValueT* __restrict__ C_values,
    IndexT* __restrict__ C_row_ptr,
    IndexT nnz,
    SemiringT semiring
);

template<typename IndexT, typename ValueT>
HPC_GLOBAL void galatic_spgemm_stage4_compress(
    IndexT* __restrict__ C_col_idx,
    ValueT* __restrict__ C_values,
    IndexT* __restrict__ C_row_ptr,
    IndexT m, IndexT nnz
);

// Hashmap-based kernel (Gustavson)
template<typename IndexT, typename ValueT, typename SemiringT>
HPC_GLOBAL void hashmap_spgemm_kernel(
    const IndexT* __restrict__ A_row_ptr,
    const IndexT* __restrict__ A_col_idx,
    const ValueT* __restrict__ A_values,
    const IndexT* __restrict__ B_row_ptr,
    const IndexT* __restrict__ B_col_idx,
    const ValueT* __restrict__ B_values,
    IndexT* __restrict__ C_row_ptr,
    IndexT* __restrict__ C_col_idx,
    ValueT* __restrict__ C_values,
    IndexT m, IndexT n, IndexT k,
    SemiringT semiring,
    IndexT* hashmap_keys,
    ValueT* hashmap_values,
    IndexT hashmap_size
);

// Merge-path kernel
template<typename IndexT, typename ValueT, typename SemiringT>
HPC_GLOBAL void merge_path_spgemm_kernel(
    const IndexT* __restrict__ A_row_ptr,
    const IndexT* __restrict__ A_col_idx,
    const ValueT* __restrict__ A_values,
    const IndexT* __restrict__ B_row_ptr,
    const IndexT* __restrict__ B_col_idx,
    const ValueT* __restrict__ B_values,
    IndexT* __restrict__ C_row_ptr,
    IndexT* __restrict__ C_col_idx,
    ValueT* __restrict__ C_values,
    IndexT m, IndexT n, IndexT k,
    SemiringT semiring
);

// Heap-based kernel
template<typename IndexT, typename ValueT, typename SemiringT>
HPC_GLOBAL void heap_spgemm_kernel(
    const IndexT* __restrict__ A_row_ptr,
    const IndexT* __restrict__ A_col_idx,
    const ValueT* __restrict__ A_values,
    const IndexT* __restrict__ B_row_ptr,
    const IndexT* __restrict__ B_col_idx,
    const ValueT* __restrict__ B_values,
    IndexT* __restrict__ C_row_ptr,
    IndexT* __restrict__ C_col_idx,
    ValueT* __restrict__ C_values,
    IndexT m, IndexT n, IndexT k,
    SemiringT semiring
);

} // namespace kernel

// CPU implementations
namespace cpu {

// Sequential Gustavson
template<typename IndexT, typename ValueT, typename SemiringT>
void gustavson_spgemm(
    const IndexT* A_row_ptr, const IndexT* A_col_idx, const ValueT* A_values,
    const IndexT* B_row_ptr, const IndexT* B_col_idx, const ValueT* B_values,
    IndexT* C_row_ptr, IndexT* C_col_idx, ValueT* C_values,
    IndexT m, IndexT n, IndexT k, SemiringT semiring
);

// Parallel Gustavson with OpenMP
template<typename IndexT, typename ValueT, typename SemiringT>
void parallel_gustavson_spgemm(
    const IndexT* A_row_ptr, const IndexT* A_col_idx, const ValueT* A_values,
    const IndexT* B_row_ptr, const IndexT* B_col_idx, const ValueT* B_values,
    IndexT* C_row_ptr, IndexT* C_col_idx, ValueT* C_values,
    IndexT m, IndexT n, IndexT k, SemiringT semiring, int num_threads
);

// Heap-based
template<typename IndexT, typename ValueT, typename SemiringT>
void heap_spgemm(
    const IndexT* A_row_ptr, const IndexT* A_col_idx, const ValueT* A_values,
    const IndexT* B_row_ptr, const IndexT* B_col_idx, const ValueT* B_values,
    IndexT* C_row_ptr, IndexT* C_col_idx, ValueT* C_values,
    IndexT m, IndexT n, IndexT k, SemiringT semiring
);

// Merge-path
template<typename IndexT, typename ValueT, typename SemiringT>
void merge_path_spgemm(
    const IndexT* A_row_ptr, const IndexT* A_col_idx, const ValueT* A_values,
    const IndexT* B_row_ptr, const IndexT* B_col_idx, const ValueT* B_values,
    IndexT* C_row_ptr, IndexT* C_col_idx, ValueT* C_values,
    IndexT m, IndexT n, IndexT k, SemiringT semiring
);

} // namespace cpu

// Distributed SpGEMM (2.5D Sparse SUMMA)
namespace distributed {

struct DistributedConfig {
    int grid_rows = 1;
    int grid_cols = 1;
    int grid_layers = 1;  // For 2.5D
    int rank = 0;
    int num_ranks = 1;
    MPI_Comm comm = MPI_COMM_WORLD;
    bool hybrid_comm = true;
};

template<typename IndexT, typename ValueT, typename SemiringT>
class DistributedSpGEMM {
public:
    DistributedSpGEMM(const DistributedConfig& config);
    ~DistributedSpGEMM();

    int compute(
        const MatrixBase* A_local,
        const MatrixBase* B_local,
        MatrixBase* C_local,
        SemiringID semiring
    );

    // 2.5D Sparse SUMMA steps
    int broadcast_A_row(const MatrixBase* A, int root_col);
    int broadcast_B_col(const MatrixBase* B, int root_row);
    int local_multiply(const MatrixBase* A, const MatrixBase* B, MatrixBase* C, SemiringID semiring);
    int reduce_scatter_C(MatrixBase* C);

    const SpGEMMStats& stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace distributed

// Factory functions
HPC_API std::unique_ptr<SpGEMM> create_spgemm(const SpGEMMConfig& config = SpGEMMConfig{});

// Benchmarking
struct HPC_API BenchmarkResult {
    std::string name;
    SpGEMMStats stats;
    SpGEMMConfig config;
    SpGEMMDescriptor descriptor;
    bool success = true;
    std::string error_message;
};

HPC_API std::vector<BenchmarkResult> run_spgemm_benchmark(
    const std::vector<SpGEMMDescriptor>& descriptors,
    const std::vector<SpGEMMConfig>& configs,
    int warmup_runs = 3,
    int benchmark_runs = 10
);

HPC_NAMESPACE_END