# Results

## Read this first

**This file contains no measured numbers, because no measurement has been
performed.** The source code in this repository has never been compiled or run.
There is no compiler, CMake, MPI, or CUDA toolkit on the development machine
used to write it.

The README table that previously claimed weak-scaling efficiencies, hybrid
communication speedups, and energy-delay-product figures for a Frontier-like
system was fabricated and has been removed. Those numbers should not have been
there. If you encounter them quoted anywhere, they are wrong.

Everything below is either a *target* taken from published work, or a
*protocol* for producing real numbers.

## Why this matters

The central technical claim of the underlying papers is that SpGEMM
performance and the optimal communication path are **properties of the
specific hardware**, not constants. A repository that reports performance
without naming the machine is therefore making a claim it cannot support.

Citing a paper's numbers as if they were this implementation's measurements is
a specific error worth naming: it conflates "the method works, as demonstrated
on their A100 cluster" with "the code in this repo achieves that on your
machine".

## Targets from the literature

These are the numbers the *papers* report, on *their* hardware. They are the
bar this implementation is aiming at, not claims about it.

### McFarland, Bellavita & Guidi (ICPE 2025), Cornell University

| Comparison | Reported speedup | Hardware (as described in the paper) |
|---|---|---|
| GPU SpGEMM vs CPU-only CombBLAS | >2x | not recorded here |
| GPU SpGEMM vs PETSc, large matrices | up to 3x | not recorded here |
| Hybrid communication vs always-GPU-direct | 2–4x | depends on fabric |

### Crossover points (McFarland et al.)

The paper reports that device-to-device is faster for small messages and
host-staged is faster for large ones, with the crossover somewhere between
64 KiB and 1 MiB depending on the fabric.

Those two values are the **defaults** in `CommConfig`:

```cpp
size_t gpu_direct_threshold  = 64 * 1024;     // below this: device-direct
size_t cpu_staging_threshold = 1024 * 1024;  // above this: host-staged
```

They are placeholders. The real crossover on any given machine must be
measured, which is what this does:

```bash
mpirun -np 8 ./benchmarks/comm_microbench
```

## How to produce real numbers

### 1. Correctness first

Nothing below is meaningful until correctness is established.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Record the pass/fail counts. If these fail, stop.

### 2. Single-node throughput

```bash
./benchmarks/spgemm_benchmark --n=200000 --density=1e-5 --reps=20 --csv=r1.csv
./benchmarks/spgemm_benchmark --n=200000 --density=1e-5 --reps=20 --sweep --csv=r2.csv
```

The sweep writes one row per problem size, which is what you want for the
scaling figure.

### 3. Communication crossover

```bash
mpirun -np 2 ./benchmarks/comm_microbench --reps=500
mpirun -np 8 ./benchmarks/comm_microbench --reps=500
```

Run this on both intra-node (single node, many ranks) and inter-node. The two
crossovers are different numbers and conflating them is a common error.

### 4. Strong and weak scaling

```bash
bash scripts/run_scaling_study.sh --nodes=1,2,4,8 --mode=strong
bash scripts/run_scaling_study.sh --nodes=1,2,4,8 --mode=weak
```

Strong-scaling efficiency is `T(1) / T(P)`. The script measures `T(1)`; do not
compute efficiency without it.

### 5. Energy

```bash
sudo ./benchmarks/energy_benchmark --n=50000 --reps=10
```

Requires RAPL access (root, or the `msr` kernel module) or an NVIDIA GPU with
NVML. Without them the benchmark says so and reports runtime only.

### 6. Roofline placement

```bash
./benchmarks/roofline --device --n=100000 --density=1e-4
```

## What to report

Any result claimed from this repository should be published as a row in a table
that includes at minimum:

| Field | Why |
|---|---|
| command line | reproducibility |
| CPU model, cores | the kernel is memory-bound; core count and cache size dominate |
| GPU model, memory BW | sets the roofline ceiling |
| interconnect | the crossover point depends on it entirely |
| MPI vendor + version | collective algorithms differ between Open MPI and MPICH |
| energy counters present? | determines whether the EDP figures mean anything |
| matrix source + `nnz` + density | SpGEMM cost is dominated by fill-in |

## Cost model reference

`SpGEMM::estimate_memory()` and `SpGEMM::recommend_config()` implement a cost
model. If you want to check the model rather than the code, feed it the same
descriptor the benchmark uses and compare its prediction to the measurement.
Disagreement is informative: it localises the problem to either the cost model
or the kernel, which is much easier to debug than a single slow number.
