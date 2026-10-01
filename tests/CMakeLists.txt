# Test suite for the HPC Research Project
#
# The suite uses the self-contained harness in tests/test_framework.hpp rather
# than GoogleTest, so that `ctest` works on bare login nodes and inside minimal
# containers where no package manager is available.

set(TEST_LIBS hpc hpc_semiring hpc_matrix hpc_communication hpc_energy
              hpc_spgemm hpc_autotuner)

if(ENABLE_MPI)
    list(APPEND TEST_LIBS MPI::MPI_CXX)
endif()
if(ENABLE_OPENMP)
    list(APPEND TEST_LIBS OpenMP::OpenMP_CXX)
endif()
if(ENABLE_CUDA)
    list(APPEND TEST_LIBS CUDA::cudart)
endif()

find_package(Threads REQUIRED)
list(APPEND TEST_LIBS Threads::Threads)

# ---- unit tests ----
add_executable(test_semiring test_semiring.cpp)
target_link_libraries(test_semiring PRIVATE ${TEST_LIBS})
add_test(NAME unit_semiring COMMAND test_semiring)
set_tests_properties(unit_semiring PROPERTIES TIMEOUT 120)

add_executable(test_matrix test_matrix.cpp)
target_link_libraries(test_matrix PRIVATE ${TEST_LIBS})
add_test(NAME unit_matrix COMMAND test_matrix)
set_tests_properties(unit_matrix PROPERTIES TIMEOUT 120)

add_executable(test_spgemm test_spgemm.cpp)
target_link_libraries(test_spgemm PRIVATE ${TEST_LIBS})
add_test(NAME unit_spgemm COMMAND test_spgemm)
set_tests_properties(unit_spgemm PROPERTIES TIMEOUT 600)

add_executable(test_energy test_energy.cpp)
target_link_libraries(test_energy PRIVATE ${TEST_LIBS})
add_test(NAME unit_energy COMMAND test_energy)
set_tests_properties(unit_energy PROPERTIES TIMEOUT 120)

add_executable(test_autotuner test_autotuner.cpp)
target_link_libraries(test_autotuner PRIVATE ${TEST_LIBS})
add_test(NAME unit_autotuner COMMAND test_autotuner)
set_tests_properties(unit_autotuner PROPERTIES TIMEOUT 300)

# ---- integration tests (require MPI) ----
if(ENABLE_MPI)
    add_executable(test_distributed test_distributed.cpp)
    target_link_libraries(test_distributed PRIVATE ${TEST_LIBS})

    # Correctness under MPI with 1..4 ranks
    foreach(N 1 2 3 4)
        add_test(NAME integration_distributed_np${N}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${N}
                         ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_distributed>
                         ${MPIEXEC_POSTFLAGS})
        set_tests_properties(integration_distributed_np${N}
            PROPERTIES TIMEOUT 600 RUN_SERIAL FALSE)
    endforeach()
endif()

# ---- regression tests ----
add_executable(test_regression test_regression.cpp)
target_link_libraries(test_regression PRIVATE ${TEST_LIBS})
add_test(NAME regression_performance COMMAND test_regression)
set_tests_properties(regression_performance PROPERTIES TIMEOUT 900)

# ---- property-based / fuzz tests ----
add_executable(test_property test_property.cpp)
target_link_libraries(test_property PRIVATE ${TEST_LIBS})
add_test(NAME property_spgemm COMMAND test_property)
set_tests_properties(property_spgemm PROPERTIES TIMEOUT 900)

# ---- example smoke test ----
add_executable(test_examples test_examples.cpp)
target_link_libraries(test_examples PRIVATE ${TEST_LIBS})
add_test(NAME smoke_examples COMMAND test_examples)
set_tests_properties(smoke_examples PROPERTIES TIMEOUT 300)

# ---- suite summary ----
message(STATUS "Tests: unit(semiring, matrix, spgemm, energy, autotuner), "
               "integration(distributed), regression, property, smoke(examples)")
