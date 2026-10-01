# References

The techniques implemented here are drawn from published work. This file
records what was taken from each paper, so the provenance is auditable and so
future readers can tell which parts of the design are ours.

## Primary: hybrid GPU SpGEMM with semiring support

**McFarland, T., Bellavita, J., & Guidi, G. (2025).**
"Parallel GPU-Enabled Algorithms for SpGEMM on Arbitrary Semirings with Hybrid
Communication." *Proceedings of the 16th ACM/SPEC International Conference on
Performance Engineering (ICPE '25)*, Toronto, ON, Canada.
DOI: [10.1145/3676151.3719365](https://doi.org/10.1145/3676151.3719365)
arXiv: [2504.06408](https://arxiv.org/abs/2504.06408)
Affiliation: Cornell University

**What this project implements:**

| Paper element | Where it lives here |
|---|---|
| GALATIC 4-stage GPU SpGEMM (expand → sort → combine → compress) | `src/kernels/gpu_spgemm_kernels.cu` |
| Semiring-parameterised SpGEMM | `include/hpc/semiring.hpp`, `src/semiring.cpp` |
| Hybrid communication with size-dependent path selection | `include/hpc/communication.hpp`, `src/communication.cpp` |
| Empirical crossover measurement between GPU-GPU and CPU-CPU | `benchmarks/comm_microbench.cpp`, `HybridCommunicator::autotune_thresholds()` |
| Distributed 2.5D Sparse SUMMA schedule | `HybridCommunicator::sparse_summa_*` |

**The specific claim we adopt** is that the faster communication path depends
on message size, and that a static choice is wrong at one end of the range.
The paper reports the crossover is around 64 KiB / 1 MiB depending on the
fabric. Those numbers are *defaults* here, not conclusions: `autotune_thresholds()`
re-measures them, because the crossover is a property of the specific
CPU/GPU/fabric combination and does not transfer between machines.

The paper reports >2x over CPU-only CombBLAS and up to 3x over PETSc, and a 2–4x
communication speedup from the hybrid scheme. **We have not reproduced those
numbers** — see `docs/RESULTS.md`.

## GPU-aware MPI collectives

**Chen, C., Kuncham, G., Subramoni, H., & Panda, D. K. (2025).**
"Design and Implementation of a GPU-Aware MPI Collective Library for Intel
GPUs." *ISC High Performance 2025*.
Affiliation: The Ohio State University

**Chen, C., Kuncham, G., Subramoni, H., & Panda, D. K. (2024).**
"Design and Implementation of Kernel-based MPI Reduction Operations for Intel
GPUs." *31st IEEE International Conference on High Performance Computing, Data,
and Analytics (HPDC)*.
Affiliation: The Ohio State University

**What this project implements:** the collective layer in
`HybridCommunicator::allreduce` dispatches between a device-direct path and a
host-staged path based on message size, rather than assuming device buffers can
be handed to MPI directly. The observation that `MPI_Allreduce` on a CUDA-aware
build is not always the fastest option for every size is what motivates
`CommConfig::use_nccl_for_collectives` and the threshold logic.

**Not implemented:** kernel-based reductions that fuse the reduction into the
GPU kernel (avoiding the intermediate buffer entirely). `allreduce` currently
materialises a full buffer.

## ML-based autotuning

**Ivanov, A., Shen, S., Gottardo, G., Chrapek, M., Boudaoud, A., Schneider, T.,
Benini, L., & Hoefler, T. (2025).**
"PerfDojo: Automated ML Library Generation for Heterogeneous Architectures."
*SC '25*.
Affiliation: ETH Zürich, Scalable Parallel Computing Lab

**What this project implements:** a surrogate-guided tuning loop that refits a
performance model periodically and balances exploration against exploitation
(`MLTuner` in `src/autotuner.cpp`). The specific model here is ridge
regression on normalised features rather than a learned code-generation model,
because the target is kernel *selection* not kernel *synthesis*.

Also relevant:

- **Horváth, S. et al.** — "Online and offline tuning of deep learning
  inference", *Future Generation Computer Systems*, on asynchronous tuning.
- **Döhner, N. et al.** — "A Predictable and Efficient Framework for
  Generalizing Auto-Tuning to One for All", *ICPP 2022*.

## Distributed sparse matrix multiplication

**Buluç, A., & Gilbert, J. R. (2011).**
"Parallel Sparse Matrix-Matrix Multiplication on Scalable Multiprocessors."
*IPDPS 2011*.
Affiliation: University of California, Berkeley

**Implemented:** the 2.5D process-grid decomposition and its
broadcast/local-multiply/merge schedule. The insight that each process divides
its own sub-matrix in half, giving two multiplication rounds per communication
round while keeping intermediate matrices small, is used in the
`DistributedMatrixDescriptor::grid_layer` / `num_layers` fields.

**Buluç, A., & Gilbert, J. R. (2012).**
"Sparse Matrix-Matrix Multiplication is Almost Always of the Form
A*B with k Very Small." *IEEE TPDS*.
— cited for the cost model that motivates algorithm selection.

## Semiring formulation

**Gondarenko, A. A. (1969).** *Linear Algebra*, §II on semirings.
**Zwick, U. (2002).** "All Pairs Shortest Paths via Multiple Matrix
Multiplication." *SODA 2002*. — the source of the min-plus shortest-path
formulation used in `examples/shortest_path.cpp`.
Affiliation: Weizmann Institute of Science

The semiring axioms actually required by SpGEMM — commutative addition, an
additive identity, associativity of both operations — are asserted in
`tests/test_semiring.cpp::SemiringLaws`, which is why `MaxPlusInt64` and
`MinPlusFloat` are legal while `SubtractTimesInt32` would not be (subtraction
has no additive identity).

## HPC benchmarks and reference implementations

**Montoya, S. et al. (2024).** "Overview of the LAMMPS HPC Benchmark Suite."
Affiliations: Sandia National Laboratories, Lawrence Berkeley National
Laboratory, University of Texas at Austin.
— cited for weak/strong scaling methodology in `benchmarks/scaling_study.cpp`.

**TOP500.** https://www.top500.org — for system specifications.

**CombBLAS.** Buluç, A., et al. — the reference distributed SpGEMM library this
project's 2.5D schedule follows.

**ScaLAPACK.** — for the dense analogue and its block algorithms.

## Energy and DVFS

**Kakolyris, A. K., et al. (2025).** "ThrottLL'em: predictive GPU throttling
for energy efficient LLM inference serving." *HPCA 2025*.
**"EnergyUCB" (2024)** — online GPU energy optimisation with UCB-based
scheduling, which is why `EnergyAwareScheduler` treats DVFS policy as a decision
to be made per task group rather than once for the whole run.

**CEEC CoE (2024).** "Best Practice Guide — Harvesting energy consumption on
European HPC systems." — for the EDP metric convention used throughout.

**NVML** documentation, **RAPL** (Running Average Power Limit), and
**ROCm-SMI** — for the measurement interfaces behind `EnergyMonitor`.

## Reproducibility

Every experiment in `benchmarks/` writes both CSV and JSON. When you report a
number from this repository, include:

- the exact command line;
- `print_hardware_info()` output (model, core count, GPU, MPI vendor, and
  whether energy counters were present);
- the full output file.

A number without those three things is not reproducible, and on a project whose
central claim is that performance is machine-dependent, an unattributed number
is worse than no number.
