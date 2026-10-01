# HPC Research Project: Hybrid MPI+OpenMP+CUDA SpGEMM with Energy-Aware Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/iamsandeshdhital/hpc-research-project/workflows/CI/badge.svg)](https://github.com/iamsandeshdhital/hpc-research-project/actions)
[![Documentation](https://img.shields.io/badge/docs-Doxygen-8da0cb.svg)](docs/ARCHITECTURE.md)

## Overview

This project implements a **high-performance distributed Sparse General Matrix-Matrix Multiplication (SpGEMM)** framework with **hybrid communication optimization** and **energy-aware GPU scheduling**, based on cutting-edge research from leading institutions:

- **Cornell University** - "Parallel GPU-Enabled Algorithms for SpGEMM on Arbitrary Semirings with Hybrid Communication" (ICPE 2025) [arXiv:2504.06408]
- **Ohio State University** - "Design and Implementation of GPU-Aware MPI Collective Library for Intel/AMD GPUs" (ISC HPC 2025)
- **ETH Zurich / Torsten Hoefler Group** - "PerfDojo: Automated ML Library Generation for Heterogeneous Architectures" (SC 2025)
- **MIT Shin Group** - "GPU implementation of second-order linear and nonlinear programming solvers" (2025)
- **ORNL/OLCF** - Frontier/Exascale HPC best practices (2024-2025)

## Status

**This code has never been compiled or run.** It was written on a machine with
no compiler, CMake, MPI, or CUDA toolkit available, so no measurement in this
repository has been performed and none is claimed. See
[docs/RESULTS.md](docs/RESULTS.md) for why that matters and for the protocol to
produce real numbers.

What this means concretely:

- The tests are written but **not verified to pass**.
- The benchmark harnesses are written but **have not been run**.
- The thresholds in `CommConfig` (64 KiB / 1 MiB) are placeholders quoted from
  the literature, not measurements of your hardware.

If you build this and find it broken, that is expected â€” please open an issue.
Treat it as a well-documented research scaffold, not a validated library.

## Key Features

### ðŸš€ Hybrid Parallel Programming Models
- **MPI** - Distributed memory communication across nodes
- **OpenMP** - Shared-memory parallelism within nodes
- **CUDA** - GPU acceleration for compute-intensive kernels
- **Hybrid Communication** - Dynamic switching between GPU-GPU and CPU-CPU paths based on message size

### âš¡ Energy-Aware Optimization
- Dynamic GPU frequency/voltage scaling (DVFS)
- Power-aware task scheduling
- Carbon-efficient computation tracking
- Real-time energy profiling with NVML/ROCm-SMI

### ðŸ“Š Comprehensive Benchmarking Suite
- Weak/strong scaling analysis
- Communication overhead profiling
- Energy-to-solution metrics
- Roofline model analysis
- Regression testing framework

### ðŸ”¬ Research-Grade Implementation
- Semiring abstraction for arbitrary algebraic structures
- Multiple sparse matrix formats (CSR, CSC, DCSC, COO)
- 2.5D distributed SpGEMM algorithm (Sparse SUMMA)
- Automated performance tuning with ML-based autotuning

## Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                    Application Layer                             â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  Graph Analytics  â”‚  Genomics  â”‚  ML Training  â”‚  Scientific   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                              â”‚
                              â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                   SpGEMM API Layer                               â”‚
â”‚  Semiring Abstraction  â”‚  Matrix Formats  â”‚  Algorithm Selectionâ”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                              â”‚
                              â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                 Hybrid Runtime Layer                             â”‚
â”‚  MPI (Inter-node)  â”‚  OpenMP (Intra-node)  â”‚  CUDA (GPU)        â”‚
â”‚  Hybrid Comm       â”‚  Energy Manager       â”‚  Autotuner         â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                              â”‚
                              â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                      Hardware Layer                              â”‚
â”‚  Multi-GPU Nodes  â”‚  InfiniBand/Slingshot  â”‚  CPU Sockets       â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

## Quick Start

### Prerequisites

```bash
# System requirements
- CMake 3.20+
- CUDA 12.0+ / ROCm 6.0+
- MPI (OpenMPI 4.1+ / MPICH 4.0+)
- OpenMP 5.0+
- Python 3.10+ (for benchmarking/analysis)
```

### Building

```bash
git clone https://github.com/iamsandeshdhital/hpc-research-project.git
cd hpc-research-project
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_CUDA=ON \
      -DENABLE_MPI=ON \
      -DENABLE_OPENMP=ON \
      -DENABLE_ENERGY_MONITORING=ON \
      ..
make -j$(nproc)
```

### Running Benchmarks

```bash
# Single-node GPU benchmark
./benchmarks/spgemm_benchmark --matrix=./data/matrices/road_usa.mtx --gpus=1

# Multi-node distributed benchmark (requires MPI)
mpirun -np 4 ./benchmarks/spgemm_benchmark --matrix=./data/matrices/road_usa.mtx --gpus=4

# Energy-aware benchmark
./benchmarks/energy_benchmark --matrix=./data/matrices/road_usa.mtx --policy=energy_optimal

# Scaling study
./scripts/run_scaling_study.sh --nodes=1,2,4,8 --matrix=synthetic --size=1M
```

## Project Structure

```
hpc-research-project/
â”œâ”€â”€ src/                    # Core implementation
â”‚   â”œâ”€â”€ spgemm/            # SpGEMM kernels (CUDA, OpenMP, Hybrid)
â”‚   â”œâ”€â”€ communication/     # Hybrid MPI communication layer
â”‚   â”œâ”€â”€ energy/            # Energy monitoring & optimization
â”‚   â”œâ”€â”€ autotuner/         # ML-based performance autotuning
â”‚   â”œâ”€â”€ matrix/            # Sparse matrix formats & I/O
â”‚   â””â”€â”€ semiring/          # Semiring abstraction
â”œâ”€â”€ include/               # Public headers
â”œâ”€â”€ tests/                 # Unit & integration tests
â”œâ”€â”€ benchmarks/            # Benchmarking suite
â”œâ”€â”€ examples/              # Example applications
â”œâ”€â”€ docs/                  # Documentation
â”œâ”€â”€ scripts/               # Build/test/deployment scripts
â”œâ”€â”€ ci/                    # CI/CD configuration
â”œâ”€â”€ config/                # Configuration files
â””â”€â”€ data/                  # Sample matrices (git-lfs)
```

## Research Papers Implemented

| Paper | Institution | Year | Implementation |
|-------|-------------|------|----------------|
| Parallel GPU-Enabled Algorithms for SpGEMM on Arbitrary Semirings with Hybrid Communication | Cornell University | 2025 | âœ… Core SpGEMM + Hybrid Comm |
| Design and Implementation of GPU-Aware MPI Collective Library for Intel GPUs | Ohio State University | 2025 | âœ… GPU-Aware MPI |
| PerfDojo: Automated ML Library Generation for Heterogeneous Architectures | ETH Zurich | 2025 | âœ… Autotuner |
| GPU implementation of second-order linear and nonlinear programming solvers | MIT | 2025 | ðŸ”„ Planned |
| Energy-efficient GPU SM allocation | Multiple | 2025 | âœ… Energy Manager |

## Performance Results

**None.** No measurement has been performed â€” see the Status section above and
[docs/RESULTS.md](docs/RESULTS.md). The numbers reported in the source papers
are recorded in [docs/REFERENCES.md](docs/REFERENCES.md) with the caveat that
they were obtained on different hardware by different authors.

The scripts below produce real numbers on your machine:

```bash
mpirun -np 8 ./benchmarks/comm_microbench    # measures the crossover point
./benchmarks/spgemm_benchmark --sweep        # throughput vs problem size
bash scripts/run_scaling_study.sh --nodes=1,2,4,8
sudo ./benchmarks/energy_benchmark           # needs RAPL or NVML
./benchmarks/roofline --device               # is the kernel memory-bound?
```

## Testing

```bash
# Run all tests
ctest --output-on-failure

# Run specific test suites
ctest -R "unit" --output-on-failure
ctest -R "integration" --output-on-failure
ctest -R "regression" --output-on-failure

# Run with sanitizers
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON ..
make && ctest
```

## Citation

There is no archived release of this software, so there is nothing to cite as a
DOI. If you build on the *ideas*, cite the underlying papers below - they are
the actual contribution being reused.

### Referenced Papers

```bibtex
@inproceedings{mcfarland2025parallel,
  title={Parallel GPU-Enabled Algorithms for SpGEMM on Arbitrary Semirings with Hybrid Communication},
  author={McFarland, Thomas and Bellavita, Julian and Guidi, Giulia},
  booktitle={Proceedings of the 16th ACM/SPEC International Conference on Performance Engineering},
  pages={1--12},
  year={2025},
  doi={10.1145/3676151.3719365}
}

@inproceedings{chen2025gpu,
  title={Design and Implementation of a GPU-Aware MPI Collective Library for Intel GPUs},
  author={Chen, C. and Kuncham, G. and Subramoni, H. and Panda, D.K.},
  booktitle={ISC High Performance 2025},
  year={2025}
}

@inproceedings{hoefler2025perfdojo,
  title={PerfDojo: Automated ML Library Generation for Heterogeneous Architectures},
  author={Ivanov, Andrei and Shen, Siyuan and Gottardo, Gioele and others},
  booktitle={SC 2025},
  year={2025}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This work is based on revolutionary research from:
- **Cornell University** (McFarland, Bellavita, Guidi)
- **Ohio State University** (Chen, Kuncham, Subramoni, Panda)
- **ETH Zurich** (Hoefler Group)
- **MIT** (Shin Group)
- **Oak Ridge National Laboratory** (OLCF)
- **NVIDIA** (Selene/SaturnV teams)
- **AMD** (Instinct team)
- **Intel** (Ponte Vecchio/Xe teams)

## Contact

**Sandesh Dhital** - [mesandeshdhital@gmail.com](mailto:mesandeshdhital@gmail.com)

Project Link: [https://github.com/iamsandeshdhital/hpc-research-project](https://github.com/iamsandeshdhital/hpc-research-project)
