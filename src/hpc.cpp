#include "hpc/hpc.hpp"
#include <cstdio>
#include <cstring>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <functional>
#include <thread>

#if HPC_PLATFORM_WINDOWS
#include <windows.h>
#endif

#if __has_include(<cpuid.h>)
#include <cpuid.h>
#endif

#if HPC_PLATFORM_MACOS
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#if __has_include(<nvml.h>)
#include <nvml.h>
#define HPC_HAVE_NVML_HEADER 1
#else
#define HPC_HAVE_NVML_HEADER 0
#endif

HPC_NAMESPACE_BEGIN

namespace {

bool g_initialized = false;
GlobalConfig g_global_config;
LogLevel g_log_level = LogLevel::INFO;
std::function<void(LogLevel, const char*, const char*)> g_log_callback = nullptr;

const char* log_level_name(LogLevel l) {
    switch (l) {
        case LogLevel::TRACE: return "TRACE";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO";
        case LogLevel::WARN:  return "WARN";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
    }
    return "?";
}

const char* compiler_name() {
#if HPC_COMPILER_GCC
    return "GCC " __VERSION__;
#elif HPC_COMPILER_CLANG
    return "Clang " __VERSION__;
#elif HPC_COMPILER_MSVC
    return "MSVC";
#elif HPC_COMPILER_INTEL
    return "Intel";
#else
    return "unknown";
#endif
}

} // namespace

// ---------------------------------------------------------------------------
// HPCError
// ---------------------------------------------------------------------------

HPCError::HPCError(int code, const std::string& message, const char* file, int line)
    : code_(code), message_(message), file_(file), line_(line) {
    what_ = "HPCError[" + std::to_string(code) + "]: " + message_ +
            " (" + file_ + ":" + std::to_string(line_) + ")";
}

const char* HPCError::what() const noexcept { return what_.c_str(); }

// ---------------------------------------------------------------------------
// Version / init
// ---------------------------------------------------------------------------

VersionInfo get_version_info() {
    VersionInfo v;
    v.compiler = compiler_name();
#if HPC_HAVE_CUDA
    v.cuda_version = "CUDA " + std::to_string(CUDART_VERSION / 1000) + "." +
                     std::to_string((CUDART_VERSION % 1000) / 10);
#else
    v.cuda_version = "not built";
#endif
#if HPC_HAVE_MPI
    // MPI_Get_library_version is safe before MPI_Init, but guard anyway so a
    // library user who has not initialised MPI gets a clean string.
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        char buf[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
        int len = 0;
        MPI_Get_library_version(buf, &len);
        v.mpi_version = buf;
    } else {
        v.mpi_version = "built but not initialised";
    }
#else
    v.mpi_version = "not built";
#endif
    return v;
}

void print_version_info() {
    VersionInfo v = get_version_info();
    printf("HPC Research Project v%s\n", HPC_PROJECT_VERSION_STRING);
    printf("  compiler : %s\n", v.compiler);
    printf("  cuda     : %s\n", v.cuda_version);
    printf("  mpi      : %.48s\n", v.mpi_version);
    printf("  openmp   : %s\n", HPC_HAVE_OPENMP ? "yes" : "no");
}

int hpc_init(int* argc, char*** argv) {
    if (g_initialized) return static_cast<int>(ErrorCode::ALREADY_INITIALIZED);

#if HPC_HAVE_MPI
    int provided = 0;
    MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_SERIAL) {
        fprintf(stderr, "[hpc] warning: MPI_THREAD_MULTIPLE not available "
                        "(got level %d)\n", provided);
    }
#else
    (void)argc; (void)argv;
#endif

#if HPC_HAVE_CUDA
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0) {
        cudaSetDevice(g_global_config.default_device);
        cudaFree(nullptr); // force context creation so errors surface early
    }
#endif

    g_initialized = true;
    return static_cast<int>(ErrorCode::SUCCESS);
}

int hpc_finalize() {
    if (!g_initialized) return static_cast<int>(ErrorCode::NOT_INITIALIZED);
    g_initialized = false;

#if HPC_HAVE_CUDA
    cudaDeviceReset();
#endif
#if HPC_HAVE_MPI
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) MPI_Finalize();
#else
    return static_cast<int>(ErrorCode::SUCCESS);
#endif
    return static_cast<int>(ErrorCode::SUCCESS);
}

GlobalConfig& global_config() { return g_global_config; }
void set_global_config(const GlobalConfig& config) { g_global_config = config; }

void set_log_level(LogLevel level) { g_log_level = level; }
LogLevel get_log_level() { return g_log_level; }
void set_log_callback(std::function<void(LogLevel, const char*, const char*)> callback) {
    g_log_callback = std::move(callback);
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB"};
    double v = static_cast<double>(bytes);
    int u = 0;
    while (v >= 1024.0 && u < 5) { v /= 1024.0; ++u; }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << v << ' ' << units[u];
    return ss.str();
}

std::string format_time(double seconds) {
    std::ostringstream ss;
    if (seconds < 1e-6)       ss << std::setprecision(2) << seconds * 1e9 << " ns";
    else if (seconds < 1e-3)  ss << std::setprecision(2) << seconds * 1e6 << " us";
    else if (seconds < 1.0)   ss << std::setprecision(2) << seconds * 1e3 << " ms";
    else                      ss << std::fixed << std::setprecision(3) << seconds << " s";
    return ss.str();
}

std::string format_flops(double flops) {
    const char* units[] = {"FLOP/s", "KFLOP/s", "MFLOP/s", "GFLOP/s", "TFLOP/s", "PFLOP/s"};
    double v = flops;
    int u = 0;
    while (v >= 1000.0 && u < 5) { v /= 1000.0; ++u; }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << v << ' ' << units[u];
    return ss.str();
}

std::string format_bandwidth(double bytes_per_sec) {
    const char* units[] = {"B/s", "KiB/s", "MiB/s", "GiB/s", "TiB/s"};
    double v = bytes_per_sec;
    int u = 0;
    while (v >= 1024.0 && u < 4) { v /= 1024.0; ++u; }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << v << ' ' << units[u];
    return ss.str();
}

// ---------------------------------------------------------------------------
// Hardware detection
// ---------------------------------------------------------------------------

HardwareInfo detect_hardware() {
    HardwareInfo info;

    // CPU model
#if defined(_WIN32)
    char buf[256];
    DWORD size = sizeof(buf);
    HKEY key;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      0, KEY_READ, &key) == ERROR_SUCCESS) {
        DWORD type;
        if (RegQueryValueExA(key, "ProcessorNameString", nullptr, &type,
                            (LPBYTE)buf, &size) == ERROR_SUCCESS) {
            info.cpu_model = buf;
        }
        RegCloseKey(key);
    }
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    info.cpu_cores = si.dwNumberOfProcessors;
    info.cpu_threads = si.dwNumberOfProcessors;
#elif defined(__linux__)
    if (std::ifstream f("/proc/cpuinfo")) {
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("model name", 0) == 0) {
                const auto pos = line.find(':');
                if (pos != std::string::npos) {
                    info.cpu_model = line.substr(pos + 2);
                }
                break;
            }
        }
    }
    info.cpu_cores = static_cast<int>(std::thread::hardware_concurrency());
    info.cpu_threads = info.cpu_cores;

    if (std::ifstream f("/proc/meminfo")) {
        std::string key;
        size_t kb = 0;
        std::string unit;
        while (f >> key >> kb >> unit) {
            if (key == "MemTotal:") { info.system_memory = kb * 1024; break; }
        }
    }
#elif defined(__APPLE__)
    size_t sz = 0;
    sysctlbyname("machdep.cpu.brand_string", nullptr, &sz, nullptr, 0);
    std::string brand(sz, '\0');
    sysctlbyname("machdep.cpu.brand_string", brand.data(), &sz, nullptr, 0);
    info.cpu_model = brand.c_str();
    info.cpu_threads = static_cast<int>(std::thread::hardware_concurrency());
#endif

    // GPUs
#if HPC_HAVE_CUDA
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) == cudaSuccess) {
        for (int i = 0; i < ndev; ++i) {
            cudaDeviceProp p{};
            if (cudaGetDeviceProperties(&p, i) == cudaSuccess) {
                info.gpu_names.emplace_back(p.name);
                info.gpu_compute_capability.push_back(p.major * 10 + p.minor);
                info.gpu_memory.push_back(p.totalGlobalMem);
            }
        }
    }
#endif

#if HPC_HAVE_NVML_HEADER
    if (nvmlInit() == NVML_SUCCESS) { info.has_nvml = true; nvmlShutdown(); }
#endif
#if HPC_PLATFORM_LINUX && HPC_ARCH_X86_64
    if (std::ifstream("/sys/class/powercap/intel-rapl/energy_uj")) info.has_rapl = true;
#endif

#if HPC_HAVE_MPI
    int mpi_ready = 0;
    MPI_Initialized(&mpi_ready);
    if (mpi_ready) {
        char mv[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
        int mlen = 0;
        MPI_Get_library_version(mv, &mlen);
        std::string s(mv, static_cast<size_t>(mlen));
        if (s.find("Open MPI") != std::string::npos) info.mpi_vendor = "Open MPI";
        else if (s.find("MPICH") != std::string::npos) info.mpi_vendor = "MPICH";
        else if (s.find("Intel") != std::string::npos) info.mpi_vendor = "Intel MPI";
        else info.mpi_vendor = "Unknown";
        info.mpi_version = s.substr(0, s.find('\n'));
    } else {
        info.mpi_vendor = "not initialised";
        info.mpi_version = "MPI built in but not initialised";
    }
#endif

    return info;
}

void print_hardware_info(const HardwareInfo& info) {
    printf("Hardware:\n");
    printf("  CPU        : %s (%d cores / %d threads)\n",
           info.cpu_model.c_str(), info.cpu_cores, info.cpu_threads);
    if (info.system_memory) printf("  Memory     : %s\n", format_bytes(info.system_memory).c_str());
    for (size_t i = 0; i < info.gpu_names.size(); ++i) {
        printf("  GPU[%zu]     : %s (sm_%d, %s)\n", i, info.gpu_names[i].c_str(),
               info.gpu_compute_capability[i],
               format_bytes(info.gpu_memory[i]).c_str());
    }
    if (!info.mpi_vendor.empty())
        printf("  MPI        : %s\n", info.mpi_version.c_str());
    printf("  Energy     : NVML=%s RAPL=%s ROCm=%s\n",
           info.has_nvml ? "yes" : "no",
           info.has_rapl ? "yes" : "no",
           info.has_rocm ? "yes" : "no");
}

// ---------------------------------------------------------------------------
// Benchmarking
// ---------------------------------------------------------------------------

void run_benchmark(const std::function<void()>& kernel, const BenchmarkConfig& config,
                   std::vector<double>& times) {
    times.clear();
    if (!kernel) return;

    // Warmup
    for (int i = 0; i < config.warmup_iterations; ++i) kernel();

#if HPC_HAVE_CUDA
    cudaDeviceSynchronize();
#endif

    const auto start = std::chrono::high_resolution_clock::now();

    int it = 0;
    while (it < config.measurement_iterations) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        kernel();
#if HPC_HAVE_CUDA
        cudaDeviceSynchronize();
#endif
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
        ++it;

        // Stop early once we have enough samples for statistical stability.
        if (it >= 5 && it < config.measurement_iterations) {
            double sum = 0.0, mn = times[0], mx = times[0];
            for (double t : times) {
                sum += t; mn = std::min(mn, t); mx = std::max(mx, t);
            }
            const double mean = sum / times.size();
            if (mean > 0 && (mx - mn) / mean < 0.02) break;
            if (mean >= config.min_time_ms) break;
        }
    }

    (void)start;
}

HPC_NAMESPACE_END