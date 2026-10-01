# Core libraries

# ---- semiring ----
add_library(hpc_semiring semiring.cpp)
target_include_directories(hpc_semiring PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
if(ENABLE_MPI)
    target_link_libraries(hpc_semiring PUBLIC MPI::MPI_CXX)
endif()

# ---- matrix ----
add_library(hpc_matrix matrix.cpp)
target_link_libraries(hpc_matrix PUBLIC hpc_semiring)
if(ENABLE_OPENMP)
    target_link_libraries(hpc_matrix PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---- communication ----
add_library(hpc_communication communication.cpp)
target_link_libraries(hpc_communication PUBLIC hpc_matrix)
if(ENABLE_MPI)
    target_link_libraries(hpc_communication PUBLIC MPI::MPI_CXX)
endif()
if(ENABLE_CUDA)
    target_link_libraries(hpc_communication PUBLIC CUDA::cudart)
endif()
if(HAVE_NVML)
    target_link_libraries(hpc_communication PUBLIC ${NVML_LIBRARY})
endif()

# ---- energy ----
add_library(hpc_energy energy.cpp)
target_link_libraries(hpc_energy PUBLIC hpc_matrix)
find_package(Threads REQUIRED)
target_link_libraries(hpc_energy PUBLIC Threads::Threads)
if(HAVE_NVML)
    target_link_libraries(hpc_energy PUBLIC ${NVML_LIBRARY})
    target_compile_definitions(hpc_energy PRIVATE HPC_HAVE_NVML_LIB=1)
endif()
if(HAVE_ROCM_SMI)
    target_link_libraries(hpc_energy PUBLIC ${ROCM_SMI_LIBRARY})
    target_compile_definitions(hpc_energy PRIVATE HPC_HAVE_ROCM_SMI=1)
endif()

# ---- spgemm ----
add_library(hpc_spgemm spgemm.cpp)
target_link_libraries(hpc_spgemm PUBLIC hpc_matrix hpc_communication hpc_energy)
if(ENABLE_OPENMP)
    target_link_libraries(hpc_spgemm PUBLIC OpenMP::OpenMP_CXX)
endif()
if(ENABLE_CUDA)
    target_link_libraries(hpc_spgemm PUBLIC CUDA::cudart)
endif()

# ---- autotuner ----
add_library(hpc_autotuner autotuner.cpp)
target_link_libraries(hpc_autotuner PUBLIC hpc_spgemm hpc_energy)

# ---- top-level umbrella library ----
add_library(hpc hpc.cpp)
target_link_libraries(hpc PUBLIC
    hpc_semiring hpc_matrix hpc_communication hpc_energy hpc_spgemm hpc_autotuner
)

# ---- convenience static "all" library ----
add_library(hpc_all INTERFACE)
target_link_libraries(hpc_all INTERFACE hpc)

# Set the version / SOVERSION on all shared libraries
set_target_properties(hpc_semiring hpc_matrix hpc_communication
                      hpc_energy hpc_spgemm hpc_autotuner hpc
    PROPERTIES
        VERSION   ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
)

# CUDA sources (compiled only when CUDA is enabled)
if(ENABLE_CUDA)
    add_library(hpc_cuda
        kernels/gpu_spgemm_kernels.cu
        kernels/communication_kernels.cu
    )
    target_link_libraries(hpc_cuda PUBLIC hpc_spgemm CUDA::cudart)
    set_target_properties(hpc_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
    # The public umbrella library also exposes the CUDA kernels.
    target_link_libraries(hpc PUBLIC hpc_cuda)
endif()
