# Benchmark suite for the HPC Research Project

set(BENCH_COMMON_LIBS hpc hpc_semiring hpc_matrix hpc_communication
                     hpc_energy hpc_spgemm hpc_autotuner)
if(ENABLE_MPI)
    list(APPEND BENCH_COMMON_LIBS MPI::MPI_CXX)
endif()
if(ENABLE_OPENMP)
    list(APPEND BENCH_COMMON_LIBS OpenMP::OpenMP_CXX)
endif()
if(ENABLE_CUDA)
    list(APPEND BENCH_COMMON_LIBS CUDA::cudart)
endif()
find_package(Threads REQUIRED)
list(APPEND BENCH_COMMON_LIBS Threads::Threads)

# ---- spgemm benchmark ----
add_executable(spgemm_benchmark spgemm_benchmark.cpp)
target_link_libraries(spgemm_benchmark PRIVATE ${BENCH_COMMON_LIBS})

# ---- energy benchmark ----
add_executable(energy_benchmark energy_benchmark.cpp)
target_link_libraries(energy_benchmark PRIVATE ${BENCH_COMMON_LIBS})

# ---- scaling study ----
add_executable(scaling_study scaling_study.cpp)
target_link_libraries(scaling_study PRIVATE ${BENCH_COMMON_LIBS})

# ---- hybrid communication microbenchmark ----
add_executable(comm_microbench comm_microbench.cpp)
target_link_libraries(comm_microbench PRIVATE ${BENCH_COMMON_LIBS})

# ---- roofline model ----
add_executable(roofline roofline.cpp)
target_link_libraries(roofline PRIVATE ${BENCH_COMMON_LIBS})

message(STATUS "Benchmarks: spgemm_benchmark, energy_benchmark, scaling_study, "
               "comm_microbench, roofline")
