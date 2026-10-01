#pragma once

// HPC Research Project Configuration
// Based on research from Cornell University, Ohio State University, ETH Zurich, MIT, and ORNL

#define HPC_PROJECT_VERSION_MAJOR 1
#define HPC_PROJECT_VERSION_MINOR 0
#define HPC_PROJECT_VERSION_PATCH 0
#define HPC_PROJECT_VERSION_STRING "1.0.0"

// Standard library headers pulled in by nearly every translation unit.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <climits>
#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>

// Feature detection
#if defined(__CUDACC__) || defined(__NVCC__)
#define HPC_HAVE_CUDA 1
#else
#define HPC_HAVE_CUDA 0
#endif

#if defined(_OPENMP)
#define HPC_HAVE_OPENMP 1
#else
#define HPC_HAVE_OPENMP 0
#endif

#if defined(MPI_VERSION)
#define HPC_HAVE_MPI 1
#else
#define HPC_HAVE_MPI 0
#endif

// Platform detection
#if defined(_WIN32) || defined(_WIN64)
#define HPC_PLATFORM_WINDOWS 1
#define HPC_PLATFORM_LINUX 0
#define HPC_PLATFORM_MACOS 0
#elif defined(__linux__)
#define HPC_PLATFORM_WINDOWS 0
#define HPC_PLATFORM_LINUX 1
#define HPC_PLATFORM_MACOS 0
#elif defined(__APPLE__)
#define HPC_PLATFORM_WINDOWS 0
#define HPC_PLATFORM_LINUX 0
#define HPC_PLATFORM_MACOS 1
#else
#define HPC_PLATFORM_WINDOWS 0
#define HPC_PLATFORM_LINUX 0
#define HPC_PLATFORM_MACOS 0
#endif

// Architecture detection
#if defined(__x86_64__) || defined(_M_X64)
#define HPC_ARCH_X86_64 1
#define HPC_ARCH_ARM64 0
#define HPC_ARCH_PPC64LE 0
#elif defined(__aarch64__)
#define HPC_ARCH_X86_64 0
#define HPC_ARCH_ARM64 1
#define HPC_ARCH_PPC64LE 0
#elif defined(__powerpc64le__)
#define HPC_ARCH_X86_64 0
#define HPC_ARCH_ARM64 0
#define HPC_ARCH_PPC64LE 1
#else
#define HPC_ARCH_X86_64 0
#define HPC_ARCH_ARM64 0
#define HPC_ARCH_PPC64LE 0
#endif

// Compiler detection
#if defined(__GNUC__) && !defined(__clang__)
#define HPC_COMPILER_GCC 1
#define HPC_COMPILER_CLANG 0
#define HPC_COMPILER_MSVC 0
#define HPC_COMPILER_INTEL 0
#elif defined(__clang__)
#define HPC_COMPILER_GCC 0
#define HPC_COMPILER_CLANG 1
#define HPC_COMPILER_MSVC 0
#define HPC_COMPILER_INTEL 0
#elif defined(_MSC_VER)
#define HPC_COMPILER_GCC 0
#define HPC_COMPILER_CLANG 0
#define HPC_COMPILER_MSVC 1
#define HPC_COMPILER_INTEL 0
#elif defined(__INTEL_COMPILER)
#define HPC_COMPILER_GCC 0
#define HPC_COMPILER_CLANG 0
#define HPC_COMPILER_MSVC 0
#define HPC_COMPILER_INTEL 1
#else
#define HPC_COMPILER_GCC 0
#define HPC_COMPILER_CLANG 0
#define HPC_COMPILER_MSVC 0
#define HPC_COMPILER_INTEL 0
#endif

// Export macros
#if HPC_PLATFORM_WINDOWS
#ifdef HPC_BUILD_SHARED_LIBS
#ifdef hpc_EXPORTS
#define HPC_API __declspec(dllexport)
#else
#define HPC_API __declspec(dllimport)
#endif
#else
#define HPC_API
#endif
#else
#define HPC_API __attribute__((visibility("default")))
#endif

// Alignment macros
#define HPC_ALIGN(n) alignas(n)
#define HPC_CACHE_LINE_SIZE 64
#define HPC_ALIGN_CACHE HPC_ALIGN(HPC_CACHE_LINE_SIZE)

// Likely/unlikely hints
#if HPC_COMPILER_GCC || HPC_COMPILER_CLANG || HPC_COMPILER_INTEL
#define HPC_LIKELY(x) __builtin_expect(!!(x), 1)
#define HPC_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define HPC_LIKELY(x) (x)
#define HPC_UNLIKELY(x) (x)
#endif

// Force inline
#if HPC_COMPILER_MSVC
#define HPC_FORCE_INLINE __forceinline
#else
#define HPC_FORCE_INLINE __attribute__((always_inline)) inline
#endif

// No inline
#if HPC_COMPILER_MSVC
#define HPC_NOINLINE __declspec(noinline)
#else
#define HPC_NOINLINE __attribute__((noinline))
#endif

// Restrict keyword
#if HPC_COMPILER_MSVC
#define HPC_RESTRICT __restrict
#else
#define HPC_RESTRICT __restrict__
#endif

// Deprecated macro
#if HPC_COMPILER_GCC || HPC_COMPILER_CLANG || HPC_COMPILER_INTEL
#define HPC_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif HPC_COMPILER_MSVC
#define HPC_DEPRECATED(msg) __declspec(deprecated(msg))
#else
#define HPC_DEPRECATED(msg)
#endif

// Namespace macros
#define HPC_NAMESPACE_BEGIN namespace hpc {
#define HPC_NAMESPACE_END }

// CUDA-specific macros
#if HPC_HAVE_CUDA
#define HPC_HOST_DEVICE __host__ __device__
#define HPC_DEVICE __device__
#define HPC_GLOBAL __global__
#define HPC_HOST __host__
#define HPC_SHARED __shared__
#define HPC_CONSTANT __constant__
#else
#define HPC_HOST_DEVICE
#define HPC_DEVICE
#define HPC_GLOBAL
#define HPC_HOST
#define HPC_SHARED
#define HPC_CONSTANT
#endif

// MPI types
#if HPC_HAVE_MPI
#include <mpi.h>
#define HPC_MPI_COMM_WORLD MPI_COMM_WORLD
#define HPC_MPI_INT MPI_INT
#define HPC_MPI_FLOAT MPI_FLOAT
#define HPC_MPI_DOUBLE MPI_DOUBLE
#define HPC_MPI_LONG_LONG MPI_LONG_LONG
#define HPC_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#define HPC_MPI_BYTE MPI_BYTE
#define HPC_MPI_PACKED MPI_PACKED
#endif

// Suppress unused-parameter warnings for intentionally ignored arguments.
#define HPC_UNUSED(x) ((void)(x))

// Debug macros
#ifdef NDEBUG
#define HPC_ASSERT(condition) ((void)0)
#define HPC_DEBUG_PRINT(fmt, ...) ((void)0)
#else
#include <cassert>
#include <cstdio>
#define HPC_ASSERT(condition) assert(condition)
#define HPC_DEBUG_PRINT(fmt, ...) fprintf(stderr, "[DEBUG] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#endif

// Error handling
#define HPC_CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[CUDA ERROR] %s:%d: %s failed: %s\n", \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
            std::abort(); \
        } \
    } while(0)

#define HPC_CHECK_MPI(call) \
    do { \
        int err = call; \
        if (err != MPI_SUCCESS) { \
            char err_str[MPI_MAX_ERROR_STRING]; \
            int len; \
            MPI_Error_string(err, err_str, &len); \
            fprintf(stderr, "[MPI ERROR] %s:%d: %s failed: %s\n", \
                    __FILE__, __LINE__, #call, err_str); \
            std::abort(); \
        } \
    } while(0)

// Timing utilities
#if HPC_HAVE_CUDA
#define HPC_CUDA_TIMER_START(event) cudaEventCreate(&event); cudaEventRecord(event)
#define HPC_CUDA_TIMER_STOP(event) cudaEventRecord(event); cudaEventSynchronize(event)
#define HPC_CUDA_TIMER_ELAPSED(start, end, ms) cudaEventElapsedTime(&ms, start, end)
#endif

// Constants
namespace hpc {
    constexpr double PI = 3.14159265358979323846;
    constexpr size_t KIB = 1024;
    constexpr size_t MIB = 1024 * KIB;
    constexpr size_t GIB = 1024 * MIB;
    constexpr int INVALID_DEVICE = -1;
    constexpr int MAX_DEVICES = 32;
    constexpr int MAX_SEMIRING_NAME = 64;
    constexpr int MAX_MATRIX_NAME = 256;
} // namespace hpc