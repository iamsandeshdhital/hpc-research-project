# Architecture

## The problem this project solves

Sparse General Matrix Multiplication (SpGEMM) — computing `C = A * B` where
`A` and `B` are too sparse for dense multiplication — sits underneath a lot of
HPC work: genomics pipelines, graph analytics, PageRank, shortest paths,
exponentiation of sparse operators in quantum-chemistry codes.

Three facts make it hard to do fast:

1. **The work is irregular.** Each row of `A` touches a different set of rows
   of `B`, so control flow and memory access are data-dependent. Peak FLOPS
   is not the relevant limit; index traffic and cache behaviour are.
2. **The output is not known in advance.** `nnz(C)` must be computed before
   `C` can be allocated, which forces a symbolic pass or a conservative bound.
3. **The arithmetic is not always arithmetic.** `min`/`+` gives all-pairs
   shortest paths, `or`/`and` gives transitive closure. A framework that
   hardcodes `multiply` and `add` has to be rewritten for each.

## Layered design

```
┌──────────────────────────────────────────────────────────────────┐
│ Applications                                                    │
│   shortest_path · reachability · pagerank · benchmarks          │
└──────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│ SpGEMM facade  (src/spgemm.cpp)                                 │
│   algorithm selection · cost model · phase control · statistics  │
└──────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│ Kernels                                                        │
│   CPU: Gustavson · heap · MergePath (OpenMP or serial)          │
│   GPU: GALATIC 4-stage (src/kernels/*.cu)                       │
└──────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│ Semiring abstraction  (include/hpc/semiring.hpp)                │
│   (add, multiply, identities, equality) parameterised per kernel │
└──────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│ Systems layer                                                   │
│   HybridCommunicator (MPI, GPU-direct vs host-staged)           │
│   EnergyManager (NVML / ROCm-SMI / RAPL, DVFS)                  │
│   AutoTuner (surrogate-guided search, runtime or EDP objective) │
└──────────────────────────────────────────────────────────────────┘
```

The dependency direction is strictly downward. The semiring layer knows
nothing about matrices; the systems layer knows nothing about SpGEMM.

## Design decisions and their justification

### Semirings are a template parameter, not a virtual call

A semiring supplies `add`, `multiply`, `identity_add`, `identity_multiply`,
`equals`, `copy`, `destroy`. Templating on it lets the compiler inline
`add` into the innermost loop of the kernel. A `std::function`-based design
was tried first and cost roughly 40% on the Gustavson kernel because the call
cannot be devirtualised.

The trade-off is that each semiring instantiates its own copy of each kernel,
which matters for binary size. The instantiation list in `src/spgemm.cpp` is
explicit rather than generated, so it is clear what actually gets compiled.

### Symbolic and numeric are separate phases

`C_row_ptr[m]` is needed before any value can be written, so `compute()` runs
`compute_symbolic()` first. Both phases are separately callable because:

- for iterative algorithms (PageRank) the structure of `C` is fixed across
  iterations, so the symbolic pass is amortised over many numeric passes;
- separating them roughly doubles peak memory, which is the dominant cost for
  very large problems, so callers sometimes prefer a fused single pass.

`SpGEMMConfig::phase` selects between `SYMBOLIC`, `NUMERIC`, and `FUSED`.

### OpenMP parallelism is row-parallel, not nested

Every CPU kernel parallelises over rows of `C`. Rows are independent: each row
writes only to `C_row_ptr[i+1]`, and the accumulation workspace is per-thread.
This gives a false-sharing hazard on the row-pointer array, which is why the
symbolic phase counts row lengths in a separate pass and only then fills
values. The counting pass has no writes at all, so it scales cleanly.

`schedule(dynamic)` is used because row cost varies by orders of magnitude in
real graphs.

### The hybrid communication threshold is measured, not assumed

`HybridCommunicator` can route a message three ways:

| Mode | Path | Best when |
|------|------|-----------|
| `GPU_DIRECT` | device → NIC → device | small messages; the DMA engine's fixed cost dominates |
| `CPU_STAGING` | device → pinned host → NIC → host → device | large messages; the PCIe copy is cheaper than a NIC transfer |
| `ZERO_COPY` | device → pinned host, zero intermediate | tiny messages that fit the staging pool |

The crossover is a property of the fabric, the CPU, and the GPU, so it is
measured per machine:

```bash
mpirun -np 8 ./benchmarks/comm_microbench      # prints the crossover
```

`HybridCommunicator::autotune_thresholds()` does the same sweep at runtime and
writes the result into `CommConfig`. The built-in defaults (64 KiB direct,
1 MiB staged) are placeholders for a generic cluster; on real hardware the
measured values are usually different.

### Energy is a first-class tuning dimension

Most autotuners minimise time. That is the wrong objective when power is
capped, carbon-priced, or simply the scarcest resource in the allocation.
`TuningObjective` therefore includes `MINIMIZE_EDP` and `MINIMIZE_ED2P`, and
`AutoTuner::tune_spgemm_energy()` supplies both runtime and energy to the
search.

The consequence is visible in `examples/energy_autotune`: the tuner's choice is
neither the fastest nor the lowest-energy configuration. That is not a bug —
it is the objective being honoured.

### The surrogate model is a ridge regression, deliberately

`PerformancePredictor` fits a linear model on normalised configuration
features via the normal equations. A gradient-boosted model would fit the
hardware's non-linearities better, but it needs far more samples than an
online tuner collects within its iteration budget, and it makes the
uncertainty estimate (needed for exploration/exploitation balance) awkward.
Ridge regression on ~50 points is enough to tell the search which direction
looks promising, which is all it is used for.

## Sparse matrix formats

| Format | Layout | Used for |
|--------|--------|----------|
| COO | `(row, col, value)` triplets | input, I/O, the GALATIC expand stage |
| CSR | row pointers + column indices + values | the CPU kernels' native format |
| CSC | column pointers + row indices + values | CombBLAS compatibility |
| DCSC | CSC with empty columns removed | highly sparse inputs, CombBLAS |

The CPU kernels take CSR directly because row-wise iteration is what the inner
loop needs. COO appears at the boundaries: reading files, distributing blocks
across ranks, and the GPU expand stage, which produces COO naturally and then
sorts into CSR.

## Numerical considerations

**Index width.** Index arrays are templated on `IndexT`. 64-bit indices double
index traffic, which for sparse products is often the bottleneck. Below 2^31
nonzeros, 32-bit indices are a large win. The cost model in
`SpGEMM::estimate_memory()` reflects this.

**Zero elimination.** `remove_zeros` drops entries whose accumulated value
falls below `zero_threshold` (default 1e-12). When anything is dropped, the row
pointers are rebuilt by matching compacted columns against the retained
symbolic structure — recomputing them from the compacted array alone would be
wrong, because a dropped entry silently shortens every row after it.

**Accumulator initialisation.** Each row's dense workspace is reset using the
semiring's `identity_add`, not by memset to zero. For `min-plus` that identity
is `+inf`, and zeroing the workspace would silently produce wrong distances.

## Known limitations

- The GPU GALATIC pipeline requires a device-to-host sort (via Thrust) between
  stages 1 and 3. For very large `nnz(C)` this dominates, and a fused
  sort-and-combine would be the obvious next step.
- `alltoallv_sparse` assumes a `(row, col, value)` packed layout for the
  exchanged blocks. Other layouts are not handled.
- The distributed path implements the 2.5D grid decomposition's communication
  schedule but does not yet vary `grid_layers` at runtime; the cost model
  predicts it, the implementation does not yet exploit it.
- No Python bindings are built by default; `ENABLE_PYTHON_BINDINGS=ON` requires
  pybind11, which is found optionally.
