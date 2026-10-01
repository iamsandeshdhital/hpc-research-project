# Contributing to HPC Research Project

Thank you for considering a contribution. This document explains how to get a
change merged without friction.

## Ground rules

1. **Correctness before speed.** A fast kernel that returns wrong answers is
   worse than a slow correct one. Every change to a kernel needs a test that
   fails without it.
2. **Show the measurement.** If you claim a speedup, include the numbers from
   `benchmarks/` and say what machine they came from.
3. **Match the surrounding style.** 4-space indent, `m_`-free member names,
   `snake_case` functions, types in `hpc::`, trailing `_t` not required here.

## Getting set up

```bash
git clone https://github.com/iamsandeshdhital/hpc-research-project.git
cd hpc-research-project
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CUDA and MPI are optional. To enable them:

```bash
cmake -S . -B build -DENABLE_CUDA=ON -DENABLE_MPI=ON
```

## Before opening a pull request

Run all three of these and include the output:

```bash
# 1. Correctness
ctest --test-dir build --output-on-failure

# 2. Sanitizers (catches the memory bugs that unit tests miss)
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON
cmake --build build-asan -j
ctest --test-dir build-asan --output-on-failure

# 3. Performance regression
ctest --test-dir build -R regression --output-on-failure
```

If a regression threshold trips and you believe the new behaviour is correct,
update the baseline rather than loosening the threshold, and explain why in the
PR description.

## Adding a kernel

A new SpGEMM algorithm should:

- live in `include/hpc/spgemm.hpp` under `namespace hpc::cpu` or
  `namespace hpc::kernel`;
- be added to the `INSTANTIATE_CPU_KERNELS` list in `src/spgemm.cpp` for
  `int32_t` and `int64_t` index types;
- have a case in `SpGEMM::select_algorithm()` with a comment saying *why* it
  wins in that regime;
- be covered by `tests/test_spgemm.cpp::KernelAgreement`, which asserts that
  every kernel matches the dense reference.

## Style notes

- Prefer `std::size_t` over `unsigned`. The codebase uses `uint64_t` for matrix
  dimensions and index arrays on purpose (matrices can exceed 2^31 nonzeros).
- Index arrays are `int32_t` or `int64_t` depending on magnitude; value arrays
  are `double` unless a kernel genuinely needs otherwise.
- Do not allocate inside the inner loop of a parallel region. Per-thread
  scratch is the single biggest OpenMP scaling bug in this kind of code.
- Any loop over rows must use `schedule(dynamic)` or a cost model. Static
  scheduling on a sparse product is dominated by the largest row.

## Reporting bugs

Include:

- the exact command line and the full output;
- `hpc::print_version_info()` and `hpc::print_hardware_info()` output;
- the matrix you used (`--matrix=...` or the generator parameters);
- whether the bug reproduces with `--gpus=0 --threads=1` (i.e. without
  parallelism), which isolates algorithmic from concurrency faults.

## License

By contributing you agree that your contribution is licensed under the MIT
License, matching the project.
