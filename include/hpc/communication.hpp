#pragma once

#include "config.hpp"
#include "matrix.hpp"
#include <cstdint>
#include <vector>
#include <functional>
#include <memory>
#include <mpi.h>

HPC_NAMESPACE_BEGIN

// Communication mode types
enum class CommMode : uint8_t {
    AUTO = 0,           // Automatic selection based on message size
    GPU_DIRECT = 1,     // Direct GPU-GPU communication (CUDA-aware MPI)
    CPU_STAGING = 2,    // Stage through CPU memory
    ZERO_COPY = 3,      // Zero-copy with pinned memory
    NCCL = 4,           // NCCL-based communication
    NVSHMEM = 5         // NVSHMEM-based communication
};

// Communication direction
enum class CommDirection : uint8_t {
    SEND = 0,
    RECV = 1,
    SENDRECV = 2,
    ALLREDUCE = 3,
    ALLGATHER = 4,
    ALLTOALL = 5,
    BROADCAST = 6,
    REDUCE = 7,
    REDUCE_SCATTER = 8,
    SCATTER = 9,
    GATHER = 10
};

// Communication buffer descriptor
struct HPC_API CommBuffer {
    void* ptr = nullptr;
    size_t size = 0;
    bool on_device = false;
    int device_id = -1;
    MPI_Datatype datatype = MPI_BYTE;
    bool is_pinned = false;

    HPC_HOST_DEVICE CommBuffer() = default;
    HPC_HOST_DEVICE CommBuffer(void* p, size_t s, bool dev, int dev_id)
        : ptr(p), size(s), on_device(dev), device_id(dev_id) {}
};

// Communication request for async operations
struct HPC_API CommRequest {
    MPI_Request mpi_request = MPI_REQUEST_NULL;
    CommBuffer send_buffer;
    CommBuffer recv_buffer;
    CommMode mode = CommMode::AUTO;
    CommDirection direction = CommDirection::SEND;
    int tag = 0;
    int peer_rank = -1;
    bool active = false;
    bool completed = false;

    HPC_HOST CommRequest() = default;
    HPC_HOST CommRequest(const CommRequest&) = delete;
    HPC_HOST CommRequest& operator=(const CommRequest&) = delete;
    HPC_HOST CommRequest(CommRequest&& other) noexcept = default;
    HPC_HOST CommRequest& operator=(CommRequest&& other) noexcept = default;
};

// Hybrid communication statistics
struct HPC_API CommStats {
    uint64_t total_bytes_sent = 0;
    uint64_t total_bytes_received = 0;
    uint64_t num_messages_sent = 0;
    uint64_t num_messages_received = 0;
    double total_comm_time = 0.0;
    double gpu_direct_time = 0.0;
    double cpu_staging_time = 0.0;
    uint64_t gpu_direct_count = 0;
    uint64_t cpu_staging_count = 0;
    uint64_t auto_switch_count = 0;

    void reset() {
        *this = CommStats{};
    }
};

// Communication configuration
struct HPC_API CommConfig {
    // Hybrid communication thresholds (in bytes)
    size_t gpu_direct_threshold = 64 * 1024;      // Below this: GPU direct
    size_t cpu_staging_threshold = 1024 * 1024;   // Above this: CPU staging
    size_t zero_copy_threshold = 4 * 1024;        // Below this: zero-copy

    // Buffer management
    size_t staging_buffer_size = 16 * 1024 * 1024;  // 16 MB default
    int num_staging_buffers = 4;
    bool use_pinned_memory = true;

    // Performance tuning
    bool enable_overlap = true;              // Overlap comm with compute
    bool enable_compression = false;         // Enable data compression
    int compression_level = 1;               // Compression level (1-9)
    bool enable_persistence = false;         // Persistent communication

    // NCCL/NVSHMEM settings
    bool use_nccl_for_collectives = false;
    bool use_nvshmem = false;
    int nccl_algo = 0;                       // NCCL algorithm selection

    // Profiling
    bool enable_profiling = true;
    bool log_communication = false;
};

// Forward declaration
class HybridCommunicator;

// Communication context for a communicator
class HPC_API CommContext {
public:
    CommContext(MPI_Comm comm = MPI_COMM_WORLD);
    ~CommContext();

    // Initialize hybrid communication
    int initialize(const CommConfig& config = CommConfig{});

    // Get configuration
    const CommConfig& config() const { return config_; }
    CommConfig& config() { return config_; }

    // Get statistics
    const CommStats& stats() const { return stats_; }
    CommStats& stats() { return stats_; }
    void reset_stats() { stats_.reset(); }

    // Buffer management
    void* allocate_staging_buffer(size_t size);
    void deallocate_staging_buffer(void* ptr);
    void* get_staging_buffer(size_t size);
    void return_staging_buffer(void* ptr);

    // MPI communicator
    MPI_Comm mpi_comm() const { return comm_; }
    int rank() const { return rank_; }
    int size() const { return size_; }

    // Device management
    int num_gpus() const { return num_gpus_; }
    int local_gpu_id() const { return local_gpu_id_; }
    void set_gpu_id(int id) { local_gpu_id_ = id; }

    // Synchronization
    void barrier() const;
    void synchronize_device() const;

private:
    MPI_Comm comm_;
    int rank_;
    int size_;
    CommConfig config_;
    CommStats stats_;
    int num_gpus_ = 0;
    int local_gpu_id_ = 0;

    // Staging buffers
    std::vector<void*> staging_buffers_;
    std::vector<bool> staging_buffer_in_use_;
    std::vector<void*> pinned_buffers_;
};

// Hybrid communicator - main interface
class HPC_API HybridCommunicator {
public:
    // Constructor
    HybridCommunicator(MPI_Comm comm = MPI_COMM_WORLD);
    ~HybridCommunicator();

    // Initialize
    int initialize(const CommConfig& config = CommConfig{});

    // Point-to-point communication
    int send(const void* buf, size_t count, MPI_Datatype datatype, int dest, int tag, CommMode mode = CommMode::AUTO);
    int recv(void* buf, size_t count, MPI_Datatype datatype, int source, int tag, CommMode mode = CommMode::AUTO);
    int sendrecv(const void* sendbuf, size_t sendcount, MPI_Datatype sendtype,
                 void* recvbuf, size_t recvcount, MPI_Datatype recvtype,
                 int dest, int sendtag, int source, int recvtag,
                 CommMode mode = CommMode::AUTO, MPI_Status* status = nullptr);

    // Non-blocking communication
    int isend(const void* buf, size_t count, MPI_Datatype datatype, int dest, int tag,
              CommMode mode, CommRequest* request);
    int irecv(void* buf, size_t count, MPI_Datatype datatype, int source, int tag,
              CommMode mode, CommRequest* request);
    int wait(CommRequest* request, MPI_Status* status = nullptr);
    int test(CommRequest* request, int* flag, MPI_Status* status = nullptr);
    int waitall(int count, CommRequest* requests[], MPI_Status* statuses = nullptr);

    // Collective communication
    int allreduce(const void* sendbuf, void* recvbuf, size_t count, MPI_Datatype datatype,
                  MPI_Op op, CommMode mode = CommMode::AUTO);
    int allgather(const void* sendbuf, size_t sendcount, MPI_Datatype sendtype,
                  void* recvbuf, size_t recvcount, MPI_Datatype recvtype,
                  CommMode mode = CommMode::AUTO);
    int alltoall(const void* sendbuf, size_t sendcount, MPI_Datatype sendtype,
                 void* recvbuf, size_t recvcount, MPI_Datatype recvtype,
                 CommMode mode = CommMode::AUTO);
    int broadcast(void* buf, size_t count, MPI_Datatype datatype, int root,
                  CommMode mode = CommMode::AUTO);
    int reduce(const void* sendbuf, void* recvbuf, size_t count, MPI_Datatype datatype,
               MPI_Op op, int root, CommMode mode = CommMode::AUTO);
    int reduce_scatter(const void* sendbuf, void* recvbuf, size_t recvcount,
                       MPI_Datatype datatype, MPI_Op op,
                       CommMode mode = CommMode::AUTO);
    int scatter(const void* sendbuf, size_t sendcount, MPI_Datatype sendtype,
                void* recvbuf, size_t recvcount, MPI_Datatype recvtype,
                int root, CommMode mode = CommMode::AUTO);
    int gather(const void* sendbuf, size_t sendcount, MPI_Datatype sendtype,
               void* recvbuf, size_t recvcount, MPI_Datatype recvtype,
               int root, CommMode mode = CommMode::AUTO);

    // Specialized sparse matrix communication
    int alltoallv_sparse(const MatrixBase* send_matrix,
                         const int* sendcounts, const int* sdispls,
                         const int* recvcounts, const int* rdispls,
                         MatrixBase* recv_matrix,
                         CommMode mode = CommMode::AUTO);

    // 2.5D SpGEMM communication (Sparse SUMMA)
    int sparse_summa_broadcast_row(const MatrixBase* matrix, int root_col, int row_comm);
    int sparse_summa_broadcast_col(const MatrixBase* matrix, int root_row, int col_comm);
    int sparse_summa_scatter_row(const MatrixBase* matrix, int row_comm);
    int sparse_summa_scatter_col(const MatrixBase* matrix, int col_comm);

    // Utility
    const CommContext& context() const { return *context_; }
    CommContext& context() { return *context_; }
    const CommStats& stats() const { return context_->stats(); }
    const CommConfig& config() const { return context_->config(); }
    CommConfig& config() { return context_->config(); }

    // Autotuning
    void autotune_thresholds(const MatrixBase* sample_matrix, int iterations = 10);
    void print_stats() const;

private:
    std::unique_ptr<CommContext> context_;

    // Internal helper to select communication mode
    CommMode select_mode(size_t message_size, CommDirection dir) const;

    // Internal implementations for different modes
    int send_gpu_direct(const void* buf, size_t count, MPI_Datatype datatype, int dest, int tag);
    int send_cpu_staging(const void* buf, size_t count, MPI_Datatype datatype, int dest, int tag);
    int send_zero_copy(const void* buf, size_t count, MPI_Datatype datatype, int dest, int tag);

    int recv_gpu_direct(void* buf, size_t count, MPI_Datatype datatype, int source, int tag);
    int recv_cpu_staging(void* buf, size_t count, MPI_Datatype datatype, int source, int tag);
    int recv_zero_copy(void* buf, size_t count, MPI_Datatype datatype, int source, int tag);

    // Collective implementations
    int allreduce_gpu_direct(const void* sendbuf, void* recvbuf, size_t count,
                             MPI_Datatype datatype, MPI_Op op);
    int allreduce_cpu_staging(const void* sendbuf, void* recvbuf, size_t count,
                              MPI_Datatype datatype, MPI_Op op);
};

// Factory function
HPC_API std::unique_ptr<HybridCommunicator> create_communicator(MPI_Comm comm = MPI_COMM_WORLD);

// CUDA-aware MPI detection
HPC_API bool is_cuda_aware_mpi_available();
HPC_API bool is_nccl_available();
HPC_API bool is_nvshmem_available();

// Communication utilities
HPC_API void pack_matrix_for_mpi(const MatrixBase* matrix, void* buffer, size_t* position);
HPC_API void unpack_matrix_from_mpi(MatrixBase* matrix, const void* buffer, size_t* position);
HPC_API size_t compute_packed_size(const MatrixBase* matrix);

HPC_NAMESPACE_END