cmake_minimum_required(VERSION 3.20)
project(HPCResearchProject
    VERSION 1.0.0
    LANGUAGES C CXX CUDA
    DESCRIPTION "Hybrid MPI+OpenMP+CUDA SpGEMM with Energy-Aware Optimization"
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build types
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(ENABLE_MPI "Enable MPI support" ON)
option(ENABLE_CUDA "Enable CUDA support" ON)
option(ENABLE_OPENMP "Enable OpenMP support" ON)
option(ENABLE_ROCM "Enable ROCm/HIP support" OFF)
option(ENABLE_ENERGY_MONITORING "Enable energy monitoring (NVML/ROCm-SMI)" ON)
option(ENABLE_AUTOTUNER "Enable ML-based autotuner" ON)
option(ENABLE_SANITIZERS "Enable address/thread sanitizers" OFF)
option(ENABLE_TESTING "Enable testing" ON)
option(ENABLE_BENCHMARKS "Enable benchmarks" ON)
option(ENABLE_EXAMPLES "Enable examples" ON)
option(ENABLE_DOCS "Enable documentation generation" ON)
option(ENABLE_PYTHON_BINDINGS "Enable Python bindings" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wconversion -Wshadow")
    if(ENABLE_SANITIZERS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined -fno-omit-frame-pointer")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fsanitize=address,undefined")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address,undefined")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX")
endif()

# Find MPI
if(ENABLE_MPI)
    find_package(MPI REQUIRED)
    if(MPI_CXX_FOUND)
        message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
    endif()
endif()

# Find CUDA
if(ENABLE_CUDA)
    find_package(CUDAToolkit 12.0 REQUIRED)
    message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")

    # Target architectures to emit SASS/PTX for (Ampere .. Hopper).
    set(CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "CUDA architectures")
endif()

# Find OpenMP
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
endif()

# Find Python (optional; bindings are only built when pybind11 is present)
if(ENABLE_PYTHON_BINDINGS)
    find_package(Python3 3.10 COMPONENTS Interpreter Development)
    find_package(pybind11 CONFIG QUIET)
    if(Python3_FOUND AND pybind11_FOUND)
        message(STATUS "Python ${Python3_VERSION} + pybind11 found: bindings enabled")
        set(HAVE_PYTHON_BINDINGS TRUE)
    else()
        message(STATUS "pybind11 not found: Python bindings disabled")
        set(ENABLE_PYTHON_BINDINGS OFF)
    endif()
endif()

# Find NVML for energy monitoring
if(ENABLE_ENERGY_MONITORING AND ENABLE_CUDA)
    find_library(NVML_LIBRARY nvidia-ml)
    if(NVML_LIBRARY)
        message(STATUS "NVML found: ${NVML_LIBRARY}")
        set(HAVE_NVML 1)
    else()
        message(WARNING "NVML not found, energy monitoring will be limited")
        set(HAVE_NVML 0)
    endif()
endif()

# Find ROCm-SMI for AMD GPUs
if(ENABLE_ENERGY_MONITORING AND ENABLE_ROCM)
    find_library(ROCM_SMI_LIBRARY rocm_smi64)
    if(ROCM_SMI_LIBRARY)
        message(STATUS "ROCm-SMI found: ${ROCM_SMI_LIBRARY}")
        set(HAVE_ROCM_SMI 1)
    else()
        message(WARNING "ROCm-SMI not found")
        set(HAVE_ROCM_SMI 0)
    endif()
endif()

# Find hwloc for topology awareness (optional)
find_package(HWLOC QUIET)
if(HWLOC_FOUND)
    message(STATUS "hwloc found: ${HWLOC_VERSION}")
    set(HAVE_HWLOC 1)
else()
    message(STATUS "hwloc not found, using fallback topology detection")
    set(HAVE_HWLOC 0)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add subdirectories
add_subdirectory(src)

# Testing
if(ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Examples
if(ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()

# Documentation
if(ENABLE_DOCS)
    find_package(Doxygen QUIET)
    if(DOXYGEN_FOUND)
        add_subdirectory(docs)
    else()
        message(WARNING "Doxygen not found, skipping documentation generation")
    endif()
endif()

# Installation
install(DIRECTORY include/ DESTINATION include)

set(HPC_INSTALL_TARGETS
    hpc_semiring hpc_matrix hpc_communication
    hpc_energy hpc_spgemm hpc_autotuner hpc
)
if(ENABLE_CUDA)
    list(APPEND HPC_INSTALL_TARGETS hpc_cuda)
endif()

install(TARGETS ${HPC_INSTALL_TARGETS}
        EXPORT HPCResearchProjectTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

install(EXPORT HPCResearchProjectTargets
        FILE HPCResearchProjectTargets.cmake
        NAMESPACE HPC::
        DESTINATION lib/cmake/HPCResearchProject)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/HPCResearchProjectConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/HPCResearchProjectConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/HPCResearchProjectConfig.cmake
    INSTALL_DESTINATION lib/cmake/HPCResearchProject
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/HPCResearchProjectConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/HPCResearchProjectConfigVersion.cmake
    DESTINATION lib/cmake/HPCResearchProject
)

# CPack configuration
set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_DESCRIPTION}")
set(CPACK_PACKAGE_VENDOR "Sandesh Dhital")
set(CPACK_PACKAGE_CONTACT "mesandeshdhital@gmail.com")
set(CPACK_GENERATOR "TGZ;ZIP")
include(CPack)