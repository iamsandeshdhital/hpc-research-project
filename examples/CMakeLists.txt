# Example applications for the HPC Research Project

set(EX_COMMON_LIBS hpc hpc_semiring hpc_matrix hpc_communication
                   hpc_energy hpc_spgemm hpc_autotuner)
if(ENABLE_MPI)
    list(APPEND EX_COMMON_LIBS MPI::MPI_CXX)
endif()
if(ENABLE_OPENMP)
    list(APPEND EX_COMMON_LIBS OpenMP::OpenMP_CXX)
endif()
if(ENABLE_CUDA)
    list(APPEND EX_COMMON_LIBS CUDA::cudart)
endif()
find_package(Threads REQUIRED)
list(APPEND EX_COMMON_LIBS Threads::Threads)

# --- shortest path via min-plus SpGEMM ---
add_executable(example_shortest_path shortest_path.cpp)
target_link_libraries(example_shortest_path PRIVATE ${EX_COMMON_LIBS})

# --- transitive closure / reachability via boolean SpGEMM ---
add_executable(example_reachability reachability.cpp)
target_link_libraries(example_reachability PRIVATE ${EX_COMMON_LIBS})

# --- PageRank via SpGEMM power iteration ---
add_executable(example_pagerank pagerank.cpp)
target_link_libraries(example_pagerank PRIVATE ${EX_COMMON_LIBS})

# --- energy-aware autotuning ---
add_executable(example_energy_autotune energy_autotune.cpp)
target_link_libraries(example_energy_autotune PRIVATE ${EX_COMMON_LIBS})

message(STATUS "Examples: shortest_path, reachability, pagerank, energy_autotune")
