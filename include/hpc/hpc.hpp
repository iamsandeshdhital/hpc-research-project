#pragma once

// HPC Research Project - Main Header
// Based on research from Cornell University, Ohio State University, ETH Zurich, MIT, and ORNL
// Author: Sandesh Dhital (iamsandeshdhital)
// Email: mesandeshdhital@gmail.com

#include "config.hpp"
#include "semiring.hpp"
#include "matrix.hpp"
#include "communication.hpp"
#include "energy.hpp"
#include "spgemm.hpp"
#include "autotuner.hpp"

#include <exception>
#include <functional>
#include <string>

HPC_NAMESPACE_BEGIN

// Version information
struct VersionInfo {
    int major = HPC_PROJECT_VERSION_MAJOR;
    int minor = HPC_PROJECT_VERSION_MINOR;
    int patch = HPC_PROJECT_VERSION_PATCH;
    const char* version_string = HPC_PROJECT_VERSION_STRING;
    const char* git_commit = "unknown";
    const char* build_date = __DATE__ " " __TIME__;
    const char* compiler = "unknown";
    const char* cuda_version = "unknown";
    const char* mpi_version = "unknown";
};

HPC_API VersionInfo get_version_info();
HPC_API void print_version_info();

// Initialize library
HPC_API int hpc_init(int* argc = nullptr, char*** argv = nullptr);
HPC_API int hpc_finalize();

// Global configuration
struct GlobalConfig {
    int default_device = 0;
    int default_mpi_threads = 1;
    bool enable_cuda = true;
    bool enable_openmp = true;
    bool enable_energy_monitoring = true;
    bool enable_autotuning = true;
    std::string log_level = "info";
    std::string cache_dir = "./.hpc_cache";
};

HPC_API GlobalConfig& global_config();
HPC_API void set_global_config(const GlobalConfig& config);

// Logging
enum class LogLevel : int {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    FATAL = 5
};

HPC_API void set_log_level(LogLevel level);
HPC_API LogLevel get_log_level();
HPC_API void set_log_callback(std::function<void(LogLevel, const char*, const char*)> callback);

// Error handling
class HPC_API HPCError : public std::exception {
public:
    HPCError(int code, const std::string& message, const char* file, int line);
    const char* what() const noexcept override;
    int code() const { return code_; }
    const std::string& message() const { return message_; }
    const char* file() const { return file_; }
    int line() const { return line_; }

private:
    int code_;
    std::string message_;
    const char* file_;
    int line_;
    std::string what_;
};

#define HPC_THROW(code, msg) throw hpc::HPCError(code, msg, __FILE__, __LINE__)
#define HPC_CHECK(call) \
    do { \
        int _err = call; \
        if (_err != 0) { \
            HPC_THROW(_err, "Function call failed: " #call); \
        } \
    } while(0)

// Error codes
enum class ErrorCode : int {
    SUCCESS = 0,
    INVALID_ARGUMENT = -1,
    OUT_OF_MEMORY = -2,
    NOT_INITIALIZED = -3,
    ALREADY_INITIALIZED = -4,
    UNSUPPORTED_OPERATION = -5,
    CUDA_ERROR = -6,
    MPI_ERROR = -7,
    FILE_ERROR = -8,
    FORMAT_ERROR = -9,
    CONVERGENCE_ERROR = -10,
    NUMERICAL_ERROR = -11,
    PERMISSION_ERROR = -12,
    TIMEOUT_ERROR = -13,
    INTERNAL_ERROR = -100
};

// Utility functions
HPC_API std::string format_bytes(size_t bytes);
HPC_API std::string format_time(double seconds);
HPC_API std::string format_flops(double flops);
HPC_API std::string format_bandwidth(double bytes_per_sec);

// Hardware detection
struct HardwareInfo {
    std::string cpu_model;
    int cpu_cores = 0;
    int cpu_threads = 0;
    size_t cpu_cache_l1 = 0;
    size_t cpu_cache_l2 = 0;
    size_t cpu_cache_l3 = 0;
    size_t system_memory = 0;
    std::vector<std::string> gpu_names;
    std::vector<int> gpu_compute_capability;
    std::vector<size_t> gpu_memory;
    bool has_nvml = false;
    bool has_rocm = false;
    bool has_rapl = false;
    std::string mpi_vendor;
    std::string mpi_version;
    std::string interconnect;
};

HPC_API HardwareInfo detect_hardware();
HPC_API void print_hardware_info(const HardwareInfo& info);

// Benchmarking utilities
struct BenchmarkConfig {
    int warmup_iterations = 5;
    int measurement_iterations = 20;
    int min_time_ms = 1000;
    bool verbose = true;
    std::string output_format = "json";
    std::string output_file = "";
};

HPC_API void run_benchmark(
    const std::function<void()>& kernel,
    const BenchmarkConfig& config,
    std::vector<double>& times
);

template<typename Func>
double benchmark_kernel(Func&& kernel, int iterations = 100) {
    // Warmup
    for (int i = 0; i < 10; ++i) {
        kernel();
    }

    // Synchronize
#if HPC_HAVE_CUDA
    cudaDeviceSynchronize();
#endif

    // Measure
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        kernel();
    }
#if HPC_HAVE_CUDA
    cudaDeviceSynchronize();
#endif
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    return duration.count() / 1e9 / iterations;
}

HPC_NAMESPACE_END