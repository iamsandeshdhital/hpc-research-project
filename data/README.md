# Matrix data

Matrices are **not** checked into this repository. Two reasons:

1. Binary/large matrix files bloat the git history irreversibly, and
2. every benchmark in `benchmarks/` can generate its own matrices
   reproducibly from a seed, which is a better provenance record than a
   downloaded file of unknown origin.

## Generating matrices

```cpp
// Synthetic, deterministic given the seed (42 is hardcoded in the generator)
auto er  = generate_erdos_renyi(n, p,   MatrixFormat::COO);
auto rmt = generate_rmat(scale, edge_factor, MatrixFormat::COO);
auto g2  = generate_grid_2d(n, MatrixFormat::COO);
auto g3  = generate_grid_3d(n, MatrixFormat::COO);
auto rnd = generate_random_sparse(rows, cols, density, MatrixFormat::COO);
```

RMAT uses the Graph500 default parameters (a=0.57, b=0.19, c=0.19, d=0.05),
so `generate_rmat(k, 16)` produces a matrix comparable to Graph500's SCALE 11
edge factor at scale k.

## Downloading SuiteSparse matrices

If you want real-world matrices, the standard collection is SuiteSparse
(https://sparse.tamu.edu). The useful ones for SpGEMM benchmarking:

| Matrix | Shape | Why it is interesting |
|---|---|---|
| `web1` | 256 570 x 256 570 | web crawl, highly skewed degrees |
| `web4` | 52 331 x 52 331 | older web crawl, more regular |
| `wikipedia` | 2 597 731 x 2 597 731 | real graph, tests 32-bit index limits |
| `GHS_psdef` | 246 827 x 246 827 | scientific, block structure |
| `dielectric2` | dielectric matrix | 3-D FEM, regular fill-in |
| `road_usa` | 1 339 x 1 339 | small, dense enough to eyeball by hand |

```bash
mkdir -p data
cd data
BASE=https://sparse.tamu.edu/MM
for m in web1 GHS_psdef; do
    curl -O ${BASE}/${m}.tar.gz
    tar xzf ${m}.tar.gz
done
```

Then read them with:

```cpp
MatrixBase* m = nullptr;
if (read_matrix_market("data/web1/web1.mtx", &m)) { /* ... */ }
```

## Using an external matrix in a benchmark

`spgemm_benchmark` currently generates its own matrices (so that the results
are reproducible without shipping data files). To benchmark a real matrix,
extend it to accept `--matrix=path.mtx` and pass the loaded matrix through to
`SpGEMM::set_matrix_a<double>()`.

## Reproducibility record

When you report a result, record which of these you used:

- generator name and all parameters, **or**
- matrix name from SuiteSparse with its size and `nnz`, **or**
- an RMAT scale and edge factor.

"Random sparse" on its own is not a sufficient description: two matrices with
the same `n` and `nnz` can have very different fill-in behaviour under
multiplication, because the *degree distribution* is what determines how
products collide.
