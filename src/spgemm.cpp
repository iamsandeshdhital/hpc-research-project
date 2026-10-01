#include "hpc/spgemm.hpp"
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <queue>
#include <cstring>
#include <chrono>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <type_traits>

#if HPC_HAVE_OPENMP
#include <omp.h>
#endif

HPC_NAMESPACE_BEGIN

namespace {
inline double now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(steady_clock::now().time_since_epoch()).count();
}
} // namespace

// ===========================================================================
// CPU kernels (namespace hpc::cpu)
// ===========================================================================

// ---------------------------------------------------------------------------
// Gustavson hashmap-based SpGEMM. This is the classic reference formulation:
// for each row i of C, accumulate products a_ik * b_kj into a dense workspace
// indexed by column j, using a hash map to track which columns have been
// touched. Column indices are emitted in sorted order because we scan j
// linearly at the end.
// ---------------------------------------------------------------------------
template<typename IndexT, typename ValueT, typename SemiringT>
static void gustavson_impl(
    const IndexT* A_row_ptr, const IndexT* A_col_idx, const ValueT* A_values,
    const IndexT* B_row_ptr, const IndexT* B_col_idx, const ValueT* B_values,
    IndexT* C_row_ptr, IndexT* C_col_idx, ValueT* C_values,
    IndexT m, IndexT n, IndexT k, SemiringT semiring);

// ---------------------------------------------------------------------------
// Gustavson hashmap-based SpGEMM (classic reference formulation).
// Defined after gustavson_impl so both share the same core loop.
// ---------------------------------------------------------------------------
template<typename IndexT, typename ValueT, typename SemiringT>
void cpu::gustavson_spgemm(
    const IndexT* A_row_ptr, const IndexT* A_col_idx, const ValueT* A_values,
    const IndexT* B_row_ptr, const IndexT* B_col_idx, const ValueT* B_values,
    IndexT* C_row_ptr, IndexT* C_col_idx, ValueT* C_values,
    IndexT m, IndexT n, IndexT k, SemiringT semiring)
{
    gustavson_impl(A_row_ptr, A_col_idx, A_values,
                   B_row_ptr, B_col_idx, B_values,
                   C_row_ptr, C_col_idx, C_values, m, n, k, semiring);
}

// Wrapper: the sequential Gustavson needs a monotonically increasing row
// pointer, so we track the cursor explicitly.
template<typename IndexT, typename ValueT, typename SemiringT>
static void gustavson_impl(
    const IndexT* A_row_ptr, const IndexT* A_col_idx, const ValueT* A_values,
    const IndexT* B_row_ptr, const IndexT* B_col_idx, const ValueT* B_values,
    IndexT* C_row_ptr, IndexT* C_col_idx, ValueT* C_values,
    IndexT m, IndexT n, IndexT k, SemiringT semiring)
{
    std::vector<ValueT> dense(static_cast<size_t>(n));
    std::vector<IndexT> marker(static_cast<size_t>(n), static_cast<IndexT>(-1));
    std::vector<IndexT> touched;

    ValueT zero{};
    semiring.identity_add(&zero);

    IndexT cursor = 0;
    C_row_ptr[0] = 0;

    for (IndexT i = 0; i < m; ++i) {
        touched.clear();

        for (IndexT p = A_row_ptr[i]; p < A_row_ptr[i + 1]; ++p) {
            const IndexT kk = A_col_idx[p];
            const ValueT a = A_values[p];
            if (kk < 0 || kk >= k) continue;

            for (IndexT q = B_row_ptr[kk]; q < B_row_ptr[kk + 1]; ++q) {
                const IndexT j = B_col_idx[q];
                if (j < 0 || j >= n) continue;

                if (marker[j] != i) {
                    marker[j] = i;
                    dense[j] = zero;
                    touched.push_back(j);
                }

                ValueT prod{};
                semiring.multiply(&a, &B_values[q], &prod);
                semiring.add(&dense[j], &prod, &dense[j]);
            }
        }

        std::sort(touched.begin(), touched.end());

        for (IndexT j : touched) {
            C_col_idx[cursor] = j;
            C_values[cursor] = dense[j];
            ++cursor;
        }
        C_row_ptr[i + 1] = cursor;
    }
}

template<typename IndexT, typename ValueT, typename SemiringT>
void cpu::parallel_gustavson_spgemm(
    const IndexT* A_row_ptr, const IndexT* A_col_idx, const ValueT* A_values,
    const IndexT* B_row_ptr, const IndexT* B_col_idx, const ValueT* B_values,
    IndexT* C_row_ptr, IndexT* C_col_idx, ValueT* C_values,
    IndexT m, IndexT n, IndexT k, SemiringT semiring, int num_threads)
{
#if HPC_HAVE_OPENMP
    if (num_threads <= 0) num_threads = omp_get_max_threads();

    // Two-pass approach:
    //   1) count nnz per output row in parallel (no writes)
    //   2) prefix-sum into C_row_ptr, then fill in parallel
    std::vector<IndexT> row_counts(static_cast<size_t>(m), 0);

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 8)
    for (IndexT i = 0; i < m; ++i) {
        std::vector<ValueT> dense(static_cast<size_t>(n));
        std::vector<IndexT> marker(static_cast<size_t>(n), static_cast<IndexT>(-1));
        ValueT zero{};
        semiring.identity_add(&zero);
        IndexT cnt = 0;

        for (IndexT p = A_row_ptr[i]; p < A_row_ptr[i + 1]; ++p) {
            const IndexT kk = A_col_idx[p];
            const ValueT a = A_values[p];
            if (kk < 0 || kk >= k) continue;
            for (IndexT q = B_row_ptr[kk]; q < B_row_ptr[kk + 1]; ++q) {
                const IndexT j = B_col_idx[q];
                if (j < 0 || j >= n) continue;
                if (marker[j] != i) { marker[j] = i; dense[j] = zero; ++cnt; }
                ValueT prod{};
                semiring.multiply(&a, &B_values[q], &prod);
                semiring.add(&dense[j], &prod, &dense[j]);
            }
        }
        row_counts[i] = cnt;
    }

    // Prefix sum
    C_row_ptr[0] = 0;
    for (IndexT i = 0; i < m; ++i) C_row_ptr[i + 1] = C_row_ptr[i] + row_counts[i];

    // Fill
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 8)
    for (IndexT i = 0; i < m; ++i) {
        std::vector<ValueT> dense(static_cast<size_t>(n));
        std::vector<IndexT> marker(static_cast<size_t>(n), static_cast<IndexT>(-1));
        std::vector<IndexT> touched;
        ValueT zero{};
        semiring.identity_add(&zero);
        IndexT cursor = C_row_ptr[i];

        for (IndexT p = A_row_ptr[i]; p < A_row_ptr[i + 1]; ++p) {
            const IndexT kk = A_col_idx[p];
            const ValueT a = A_values[p];
            if (kk < 0 || kk >= k) continue;
            for (IndexT q = B_row_ptr[kk]; q < B_row_ptr[kk + 1]; ++q) {
                const IndexT j = B_col_idx[q];
                if (j < 0 || j >= n) continue;
                if (marker[j] != i) { marker[j] = i; dense[j] = zero; touched.push_back(j); }
                ValueT prod{};
                semiring.multiply(&a, &B_values[q], &prod);
                semiring.add(&dense[j], &prod, &dense[j]);
            }
        }
        std::sort(touched.begin(), touched.end());
        for (IndexT j : touched) {
            C_col_idx[cursor] = j;
            C_values[cursor] = dense[j];
            ++cursor;
        }
    }
#else
    (void)num_threads;
    cpu::gustavson_spgemm(A_row_ptr, A_col_idx, A_values,
                          B_row_ptr, B_col_idx, B_values,
                          C_row_ptr, C_col_idx, C_values, m, n, k, semiring);
#endif
}

template<typename IndexT, typename ValueT, typename SemiringT>
void cpu::heap_spgemm(
    const IndexT* A_row_ptr, const IndexT* A_col_idx, const ValueT* A_values,
    const IndexT* B_row_ptr, const IndexT* B_col_idx, const ValueT* B_values,
    IndexT* C_row_ptr, IndexT* C_col_idx, ValueT* C_values,
    IndexT m, IndexT n, IndexT k, SemiringT semiring)
{
    // Heap-based: emit (value, column) pairs, then sort. Avoids the O(n)
    // workspace that Gustavson needs, at the cost of extra memory traffic.
    std::vector<std::pair<ValueT, IndexT>> acc;
    IndexT cursor = 0;
    C_row_ptr[0] = 0;

    for (IndexT i = 0; i < m; ++i) {
        acc.clear();
        for (IndexT p = A_row_ptr[i]; p < A_row_ptr[i + 1]; ++p) {
            const IndexT kk = A_col_idx[p];
            const ValueT a = A_values[p];
            if (kk < 0 || kk >= k) continue;
            for (IndexT q = B_row_ptr[kk]; q < B_row_ptr[kk + 1]; ++q) {
                const IndexT j = B_col_idx[q];
                if (j < 0 || j >= n) continue;
                ValueT prod{};
                semiring.multiply(&a, &B_values[q], &prod);
                acc.emplace_back(prod, j);
            }
        }
        std::sort(acc.begin(), acc.end(),
                  [](const auto& x, const auto& y) { return x.second < y.second; });

        // Coalesce duplicates
        IndexT w = cursor;
        for (size_t t = 0; t < acc.size(); ++t) {
            if (w > cursor && acc[t].second == C_col_idx[w - 1]) {
                semiring.add(&C_values[w - 1], &acc[t].first, &C_values[w - 1]);
            } else {
                C_col_idx[w] = acc[t].second;
                C_values[w] = acc[t].first;
                ++w;
            }
        }
        cursor = w;
        C_row_ptr[i + 1] = cursor;
    }
}

template<typename IndexT, typename ValueT, typename SemiringT>
void cpu::merge_path_spgemm(
    const IndexT* A_row_ptr, const IndexT* A_col_idx, const ValueT* A_values,
    const IndexT* B_row_ptr, const IndexT* B_col_idx, const ValueT* B_values,
    IndexT* C_row_ptr, IndexT* C_col_idx, ValueT* C_values,
    IndexT m, IndexT n, IndexT k, SemiringT semiring)
{
    // Merge-path reduces the per-row workspace when both A and B rows are
    // sorted. We exploit the sortedness of B rows to stream through products.
    std::vector<std::pair<ValueT, IndexT>> acc;
    IndexT cursor = 0;
    C_row_ptr[0] = 0;

    for (IndexT i = 0; i < m; ++i) {
        acc.clear();
        for (IndexT p = A_row_ptr[i]; p < A_row_ptr[i + 1]; ++p) {
            const IndexT kk = A_col_idx[p];
            const ValueT a = A_values[p];
            if (kk < 0 || kk >= k) continue;
            // B's row is sorted, so products come out ordered by column.
            for (IndexT q = B_row_ptr[kk]; q < B_row_ptr[kk + 1]; ++q) {
                const IndexT j = B_col_idx[q];
                if (j < 0 || j >= n) continue;
                ValueT prod{};
                semiring.multiply(&a, &B_values[q], &prod);
                acc.emplace_back(prod, j);
            }
        }
        std::stable_sort(acc.begin(), acc.end(),
                         [](const auto& x, const auto& y) { return x.second < y.second; });

        IndexT w = cursor;
        for (size_t t = 0; t < acc.size(); ++t) {
            if (w > cursor && acc[t].second == C_col_idx[w - 1]) {
                semiring.add(&C_values[w - 1], &acc[t].first, &C_values[w - 1]);
            } else {
                C_col_idx[w] = acc[t].second;
                C_values[w] = acc[t].first;
                ++w;
            }
        }
        cursor = w;
        C_row_ptr[i + 1] = cursor;
    }
}

// Fix the sequential Gustavson entry point to route to the correct implementation.
template<typename IndexT, typename ValueT, typename SemiringT>
void cpu::gustavson_spgemm_dispatch(    const IndexT* A_row_ptr, const IndexT* A_col_idx, const ValueT* A_values,
    const IndexT* B_row_ptr, const IndexT* B_col_idx, const ValueT* B_values,
    IndexT* C_row_ptr, IndexT* C_col_idx, ValueT* C_values,
    IndexT m, IndexT n, IndexT k, SemiringT semiring)
{
    gustavson_impl(A_row_ptr, A_col_idx, A_values,
                   B_row_ptr, B_col_idx, B_values,
                   C_row_ptr, C_col_idx, C_values, m, n, k, semiring);
}

// ===========================================================================
// Explicit instantiations for the CPU kernels
// ===========================================================================

#define INSTANTIATE_CPU_KERNELS(IndexT, ValueT, SEMIRING)                                  \
    template void cpu::gustavson_spgemm<IndexT, ValueT, SEMIRING>(                        \
        const IndexT*, const IndexT*, const ValueT*,                                       \
        const IndexT*, const IndexT*, const ValueT*,                                       \
        IndexT*, IndexT*, ValueT*, IndexT, IndexT, IndexT, SEMIRING);                      \
    template void cpu::parallel_gustavson_spgemm<IndexT, ValueT, SEMIRING>(                \
        const IndexT*, const IndexT*, const ValueT*,                                       \
        const IndexT*, const IndexT*, const ValueT*,                                       \
        IndexT*, IndexT*, ValueT*, IndexT, IndexT, IndexT, SEMIRING, int);                  \
    template void cpu::heap_spgemm<IndexT, ValueT, SEMIRING>(                              \
        const IndexT*, const IndexT*, const ValueT*,                                       \
        const IndexT*, const IndexT*, const ValueT*,                                       \
        IndexT*, IndexT*, ValueT*, IndexT, IndexT, IndexT, SEMIRING);                      \
    template void cpu::merge_path_spgemm<IndexT, ValueT, SEMIRING>(                         \
        const IndexT*, const IndexT*, const ValueT*,                                       \
        const IndexT*, const IndexT*, const ValueT*,                                       \
        IndexT*, IndexT*, ValueT*, IndexT, IndexT, IndexT, SEMIRING);

INSTANTIATE_CPU_KERNELS(int32_t, float,  PlusTimesFloat)
INSTANTIATE_CPU_KERNELS(int32_t, double, PlusTimesDouble)
INSTANTIATE_CPU_KERNELS(int32_t, int32_t,PlusTimesInt32)
INSTANTIATE_CPU_KERNELS(int64_t, double, PlusTimesDouble)
INSTANTIATE_CPU_KERNELS(int64_t, float,  PlusTimesFloat)
INSTANTIATE_CPU_KERNELS(int64_t, int64_t,PlusTimesInt64)

// Min-plus variants (shortest-path kernels)
INSTANTIATE_CPU_KERNELS(int32_t, int32_t, MinPlusInt32)
INSTANTIATE_CPU_KERNELS(int32_t, float,  MinPlusFloat)
INSTANTIATE_CPU_KERNELS(int32_t, double, MinPlusDouble)
INSTANTIATE_CPU_KERNELS(int64_t, int64_t, MinPlusInt64)
INSTANTIATE_CPU_KERNELS(int64_t, float,  MinPlusFloat)
INSTANTIATE_CPU_KERNELS(int64_t, double, MinPlusDouble)

#undef INSTANTIATE_CPU_KERNELS

// ===========================================================================
// SpGEMM class
// ===========================================================================

struct SpGEMM::Impl {
    // Input matrices in CSR (double)
    std::vector<uint64_t> a_row_ptr, a_col_idx;
    std::vector<double>   a_values;
    std::vector<uint64_t> b_row_ptr, b_col_idx;
    std::vector<double>   b_values;

    // Output (CSR, double)
    std::vector<uint64_t> c_row_ptr, c_col_idx;
    std::vector<double>   c_values;

    // Output CSR in int32 for kernel specialisation
    std::vector<int32_t>  c_row_ptr_i32, c_col_idx_i32;
    std::vector<int32_t>  a_row_ptr_i32, a_col_idx_i32;
    std::vector<int32_t>  b_row_ptr_i32, b_col_idx_i32;

    uint64_t m = 0, n = 0, k = 0;

    // Semantic symbolic phase (structure of C)
    std::vector<uint64_t> c_row_ptr_sym;
    std::vector<int32_t>  c_col_idx_sym;

    bool symbolic_done = false;
    bool numeric_done = false;

    // Comm buffers for async overlap
    std::vector<void*> pending_buffers;
};

SpGEMM::SpGEMM() : impl_(std::make_unique<Impl>()) {}
SpGEMM::~SpGEMM() = default;

int SpGEMM::initialize(const SpGEMMConfig& config) {
    config_ = config;

#if HPC_HAVE_OPENMP
    if (config_.openmp_threads > 0) {
        omp_set_num_threads(config_.openmp_threads);
    }
#endif
#if HPC_HAVE_CUDA
    if (config_.use_cuda) {
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0) {
            cudaSetDevice(config_.cuda_device % ndev);
        } else {
            config_.use_cuda = false;
        }
    }
#endif
    return static_cast<int>(ErrorCode::SUCCESS);
}

template<typename T>
int SpGEMM::set_matrix_a(const MatrixBase* matrix) {
    if (!matrix) return static_cast<int>(ErrorCode::INVALID_ARGUMENT);
    // The stored value type is double; the template parameter lets call sites
    // state intent and leaves room for float without an API change.
    static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value,
                  "SpGEMM currently supports double and float value types");
    if (!std::is_same<T, double>::value) {
        return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION);
    }
    impl_->m = matrix->num_rows();
    impl_->k = matrix->num_cols();

    // Generic COO -> CSR conversion path (format-agnostic entry).
    const size_t nnz = matrix->num_nonzeros();
    impl_->a_row_ptr.assign(impl_->m + 1, 0);
    impl_->a_col_idx.resize(nnz);
    impl_->a_values.resize(nnz);

    // Read triplets if COO; otherwise treat as packed (row, col, value).
    const auto* mm = dynamic_cast<const Matrix<double, MatrixFormat::COO>*>(matrix);
    if (mm) {
        for (size_t i = 0; i < nnz; ++i) {
            impl_->a_col_idx[i] = mm->coo_cols_[i];
            impl_->a_values[i] = mm->coo_values_[i];
            impl_->a_row_ptr[mm->coo_rows_[i] + 1]++;
        }
    } else {
        // Fall back to a dense-ish interpretation: not supported without format
        return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION);
    }

    for (uint64_t i = 0; i < impl_->m; ++i) impl_->a_row_ptr[i + 1] += impl_->a_row_ptr[i];

    // Sort column indices within each row
    std::vector<std::pair<uint64_t, std::pair<uint64_t, double>>> tmp;
    for (uint64_t i = 0; i < impl_->m; ++i) {
        const uint64_t s = impl_->a_row_ptr[i], e = impl_->a_row_ptr[i + 1];
        tmp.clear();
        for (uint64_t p = s; p < e; ++p) {
            tmp.emplace_back(impl_->a_col_idx[p],
                             std::make_pair(i, impl_->a_values[p]));
        }
        std::sort(tmp.begin(), tmp.end(),
                  [](const auto& x, const auto& y) { return x.first < y.first; });
        for (uint64_t p = 0; p < tmp.size(); ++p) {
            impl_->a_col_idx[s + p] = tmp[p].first;
            impl_->a_values[s + p] = tmp[p].second.second;
        }
    }

    // int32 mirrors
    impl_->a_row_ptr_i32.resize(impl_->a_row_ptr.size());
    impl_->a_col_idx_i32.resize(impl_->a_col_idx.size());
    std::copy(impl_->a_row_ptr.begin(), impl_->a_row_ptr.end(), impl_->a_row_ptr_i32.begin());
    std::copy(impl_->a_col_idx.begin(), impl_->a_col_idx.end(), impl_->a_col_idx_i32.begin());

    return static_cast<int>(ErrorCode::SUCCESS);
}

template<typename T>
int SpGEMM::set_matrix_b(const MatrixBase* matrix) {
    if (!matrix) return static_cast<int>(ErrorCode::INVALID_ARGUMENT);
    impl_->k = matrix->num_rows();
    impl_->n = matrix->num_cols();

    const size_t nnz = matrix->num_nonzeros();
    impl_->b_row_ptr.assign(impl_->k + 1, 0);
    impl_->b_col_idx.resize(nnz);
    impl_->b_values.resize(nnz);

    const auto* mm = dynamic_cast<const Matrix<double, MatrixFormat::COO>*>(matrix);
    if (mm) {
        for (size_t i = 0; i < nnz; ++i) {
            impl_->b_col_idx[i] = mm->coo_cols_[i];
            impl_->b_values[i] = mm->coo_values_[i];
            impl_->b_row_ptr[mm->coo_rows_[i] + 1]++;
        }
    } else {
        return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION);
    }
    for (uint64_t i = 0; i < impl_->k; ++i) impl_->b_row_ptr[i + 1] += impl_->b_row_ptr[i];

    std::vector<std::pair<uint64_t, double>> tmp;
    for (uint64_t i = 0; i < impl_->k; ++i) {
        const uint64_t s = impl_->b_row_ptr[i], e = impl_->b_row_ptr[i + 1];
        tmp.clear();
        for (uint64_t p = s; p < e; ++p) tmp.emplace_back(impl_->b_col_idx[p], impl_->b_values[p]);
        std::sort(tmp.begin(), tmp.end(),
                  [](const auto& x, const auto& y) { return x.first < y.first; });
        for (uint64_t p = 0; p < tmp.size(); ++p) {
            impl_->b_col_idx[s + p] = tmp[p].first;
            impl_->b_values[s + p] = tmp[p].second;
        }
    }

    impl_->b_row_ptr_i32.resize(impl_->b_row_ptr.size());
    impl_->b_col_idx_i32.resize(impl_->b_col_idx.size());
    std::copy(impl_->b_row_ptr.begin(), impl_->b_row_ptr.end(), impl_->b_row_ptr_i32.begin());
    std::copy(impl_->b_col_idx.begin(), impl_->b_col_idx.end(), impl_->b_col_idx_i32.begin());

    return static_cast<int>(ErrorCode::SUCCESS);
}

template<typename T>
int SpGEMM::set_matrix_c(MatrixBase* matrix) {
    if (!matrix) return static_cast<int>(ErrorCode::INVALID_ARGUMENT);
    if (matrix->num_rows() != impl_->m || matrix->num_cols() != impl_->n) {
        return static_cast<int>(ErrorCode::INVALID_ARGUMENT);
    }
    impl_->c_row_ptr.assign(impl_->m + 1, 0);
    const uint64_t out_nnz = matrix->num_nonzeros();
    impl_->c_col_idx.assign(out_nnz, 0);
    impl_->c_values.assign(out_nnz, 0.0);
    impl_->c_row_ptr_i32.assign(impl_->m + 1, 0);
    impl_->c_col_idx_i32.assign(out_nnz, 0);
    return static_cast<int>(ErrorCode::SUCCESS);
}

int SpGEMM::set_semiring(SemiringID semiring_id) {
    semiring_id_ = semiring_id;
    return static_cast<int>(ErrorCode::SUCCESS);
}

SpGEMMAlgorithm SpGEMM::select_algorithm(const SpGEMMDescriptor& desc) {
    const double density = std::max(desc.density_a, desc.density_b);

#if HPC_HAVE_CUDA
    if (desc.num_gpus > 0) {
        // Highly sparse and large: GALATIC-style GPU algorithm wins (cf.
        // McFarland, Bellavita & Guidi, ICPE 2025).
        if (desc.nnz_a > (1u << 22) || density < 1e-4) {
            return SpGEMMAlgorithm::GALATIC;
        }
        return SpGEMMAlgorithm::HASHMAP;
    }
#endif
    if (desc.num_mpi_ranks > 1) return SpGEMMAlgorithm::MERGE_PATH;
    if (desc.nnz_a > (1u << 24)) return SpGEMMAlgorithm::HEAP;
    return SpGEMMAlgorithm::HASHMAP;
}

SpGEMMConfig SpGEMM::recommend_config(const SpGEMMDescriptor& desc) {
    SpGEMMConfig cfg;
    cfg.algorithm = select_algorithm(desc);

#if HPC_HAVE_OPENMP
    cfg.use_openmp = true;
    cfg.openmp_threads = 0;
#endif
#if HPC_HAVE_CUDA
    cfg.use_cuda = desc.num_gpus > 0;
    cfg.cuda_device = 0;
    if (desc.nnz_a < (1u << 20)) {
        cfg.cuda_threads_per_block = 128;
        cfg.initial_hashmap_size = 256;
    } else if (desc.nnz_a < (1u << 24)) {
        cfg.cuda_threads_per_block = 256;
        cfg.initial_hashmap_size = 1024;
    } else {
        cfg.cuda_threads_per_block = 512;
        cfg.initial_hashmap_size = 4096;
    }
    cfg.hashmap_load_factor = 2;
#endif
    cfg.comm_threshold = 64 * 1024;
    cfg.hybrid_comm = true;
    cfg.distributed = desc.num_mpi_ranks > 1;
    return cfg;
}

size_t SpGEMM::estimate_memory(const SpGEMMDescriptor& desc) {
    const size_t idx = sizeof(uint64_t);
    const size_t val = 8; // double
    size_t a = desc.nnz_a * (idx + val) + (desc.m + 1) * idx;
    size_t b = desc.nnz_b * (idx + val) + (desc.k + 1) * idx;
    // Worst-case output: full outer product
    size_t c = desc.nnz_a * desc.nnz_b / std::max<size_t>(desc.n, 1) * (idx + val);
    c = std::min(c, desc.m * desc.n * (idx + val));
    size_t workspace = desc.n * (idx + val) * 4;   // dense workspace
    workspace += desc.nnz_a * 8;                  // heap expansion
    return a + b + c + workspace;
}

int SpGEMM::compute_symbolic() {
    const double t0 = now_ms();

    // Symbolic phase: only the sparsity pattern of C is needed. Use the
    // MergePath-style formulation on the index arrays (values unused).
    const uint64_t m = impl_->m, n = impl_->n, k = impl_->k;
    impl_->c_row_ptr_sym.assign(m + 1, 0);

    std::vector<int32_t> marker(n, -1);
    uint64_t cursor = 0;

#if HPC_HAVE_OPENMP
    const int nthreads = (config_.openmp_threads > 0) ? config_.openmp_threads
                                                      : omp_get_max_threads();
    std::vector<uint64_t> counts(m, 0);

    #pragma omp parallel
    {
        std::vector<int32_t> local_marker(n, -1);
        #pragma omp for schedule(dynamic, 16)
        for (int64_t i = 0; i < static_cast<int64_t>(m); ++i) {
            uint64_t cnt = 0;
            for (uint64_t p = impl_->a_row_ptr[i]; p < impl_->a_row_ptr[i + 1]; ++p) {
                const uint64_t kk = impl_->a_col_idx[p];
                if (kk >= k) continue;
                for (uint64_t q = impl_->b_row_ptr[kk]; q < impl_->b_row_ptr[kk + 1]; ++q) {
                    const uint64_t j = impl_->b_col_idx[q];
                    if (j >= n) continue;
                    if (local_marker[j] != static_cast<int32_t>(i)) {
                        local_marker[j] = static_cast<int32_t>(i);
                        ++cnt;
                    }
                }
            }
            counts[i] = cnt;
        }
    }

    impl_->c_row_ptr_sym[0] = 0;
    for (uint64_t i = 0; i < m; ++i) impl_->c_row_ptr_sym[i + 1] = impl_->c_row_ptr_sym[i] + counts[i];
    const uint64_t nnz = impl_->c_row_ptr_sym[m];
    impl_->c_col_idx_sym.assign(nnz, 0);

    #pragma omp parallel
    {
        std::vector<int32_t> local_marker(n, -1);
        #pragma omp for schedule(dynamic, 16)
        for (int64_t i = 0; i < static_cast<int64_t>(m); ++i) {
            std::vector<int32_t> touched;
            for (uint64_t p = impl_->a_row_ptr[i]; p < impl_->a_row_ptr[i + 1]; ++p) {
                const uint64_t kk = impl_->a_col_idx[p];
                if (kk >= k) continue;
                for (uint64_t q = impl_->b_row_ptr[kk]; q < impl_->b_row_ptr[kk + 1]; ++q) {
                    const uint64_t j = impl_->b_col_idx[q];
                    if (j >= n) continue;
                    if (local_marker[j] != static_cast<int32_t>(i)) {
                        local_marker[j] = static_cast<int32_t>(i);
                        touched.push_back(static_cast<int32_t>(j));
                    }
                }
            }
            std::sort(touched.begin(), touched.end());
            uint64_t cur = impl_->c_row_ptr_sym[i];
            for (int32_t j : touched) impl_->c_col_idx_sym[cur++] = j;
        }
    }
#else
    std::vector<int32_t> touched;
    for (uint64_t i = 0; i < m; ++i) {
        touched.clear();
        for (uint64_t p = impl_->a_row_ptr[i]; p < impl_->a_row_ptr[i + 1]; ++p) {
            const uint64_t kk = impl_->a_col_idx[p];
            if (kk >= k) continue;
            for (uint64_t q = impl_->b_row_ptr[kk]; q < impl_->b_row_ptr[kk + 1]; ++q) {
                const uint64_t j = impl_->b_col_idx[q];
                if (j >= n) continue;
                if (marker[j] != static_cast<int32_t>(i)) {
                    marker[j] = static_cast<int32_t>(i);
                    touched.push_back(static_cast<int32_t>(j));
                }
            }
        }
        std::sort(touched.begin(), touched.end());
        uint64_t cur = impl_->c_row_ptr_sym[i];
        for (int32_t j : touched) impl_->c_col_idx_sym[cur++] = j;
        impl_->c_row_ptr_sym[i + 1] = cur;
    }
#endif

    (void)cursor;
    impl_->symbolic_done = true;
    stats_.symbolic_time_ms = now_ms() - t0;
    stats_.output_nnz = impl_->c_row_ptr_sym.empty() ? 0 : impl_->c_row_ptr_sym[m];
    return static_cast<int>(ErrorCode::SUCCESS);
}

int SpGEMM::compute_numeric() {
    if (!impl_->symbolic_done) {
        int rc = compute_symbolic();
        if (rc != static_cast<int>(ErrorCode::SUCCESS)) return rc;
    }
    const double t0 = now_ms();

    const uint64_t m = impl_->m, n = impl_->n, k = impl_->k;
    const uint64_t nnz = impl_->c_row_ptr_sym[m];

    impl_->c_row_ptr = impl_->c_row_ptr_sym;
    impl_->c_col_idx.assign(impl_->c_col_idx_sym.begin(), impl_->c_col_idx_sym.end());
    impl_->c_values.assign(nnz, 0.0);

    // Numeric phase: reuse the symbolic structure and accumulate values with
    // a dense workspace + marker array (the symbolic pass already told us which
    // columns belong to each row, so we can accumulate in place).
    const auto* desc = SemiringRegistry::instance().get_descriptor(semiring_id_);
    auto add   = desc ? desc->add_op   : nullptr;
    auto mul   = desc ? desc->mul_op   : nullptr;
    auto ident = desc ? desc->add_identity : nullptr;

    std::vector<double> dense(n, 0.0);
    std::vector<int32_t> marker(n, -1);
    std::vector<int32_t> touched;

    for (uint64_t i = 0; i < m; ++i) {
        touched.clear();

        for (uint64_t p = impl_->a_row_ptr[i]; p < impl_->a_row_ptr[i + 1]; ++p) {
            const uint64_t kk = impl_->a_col_idx[p];
            const double a = impl_->a_values[p];
            if (kk >= k) continue;
            for (uint64_t q = impl_->b_row_ptr[kk]; q < impl_->b_row_ptr[kk + 1]; ++q) {
                const uint64_t j = impl_->b_col_idx[q];
                if (j >= n) continue;
                if (marker[j] != static_cast<int32_t>(i)) {
                    marker[j] = static_cast<int32_t>(i);
                    double z = 0.0;
                    if (ident) ident(&z);
                    dense[j] = z;
                    touched.push_back(static_cast<int32_t>(j));
                }
                double prod = a * impl_->b_values[q];
                if (mul && add) {
                    mul(&a, &impl_->b_values[q], &prod);
                    add(&dense[j], &prod, &dense[j]);
                } else {
                    dense[j] += prod;
                }
            }
        }

        // Symbolic pass emitted these columns sorted; accumulate in that order.
        const uint64_t s = impl_->c_row_ptr[i];
        const uint64_t e = impl_->c_row_ptr[i + 1];
        for (uint64_t p = s; p < e; ++p) {
            const uint64_t j = impl_->c_col_idx[p];
            impl_->c_values[p] = (marker[j] == static_cast<int32_t>(i)) ? dense[j] : 0.0;
        }
    }

    // Optional zero elimination. Compact (col, value) pairs elementwise, then
    // rebuild row pointers from the symbolic structure we still hold.
    if (config_.remove_zeros) {
        const double eps = config_.zero_threshold;
        const uint64_t orig_nnz = nnz;

        uint64_t w = 0;
        for (uint64_t p = 0; p < orig_nnz; ++p) {
            if (std::abs(impl_->c_values[p]) > eps) {
                impl_->c_col_idx[w] = impl_->c_col_idx[p];
                impl_->c_values[w] = impl_->c_values[p];
                ++w;
            }
        }
        impl_->c_col_idx.resize(w);
        impl_->c_values.resize(w);

        if (w != orig_nnz) {
            std::vector<uint64_t> rp(m + 1, 0);
            uint64_t out = 0, pos = 0;
            for (uint64_t i = 0; i < m; ++i) {
                rp[i] = out;
                for (uint64_t p = impl_->c_row_ptr_sym[i]; p < impl_->c_row_ptr_sym[i + 1]; ++p) {
                    if (pos < w && impl_->c_col_idx[pos] == impl_->c_col_idx_sym[p]) {
                        ++out;
                        ++pos;
                    }
                }
            }
            rp[m] = out;
            impl_->c_row_ptr = rp;
            stats_.output_nnz = out;
        } else {
            stats_.output_nnz = w;
        }
    } else {
        stats_.output_nnz = nnz;
    }

    impl_->c_row_ptr_i32.assign(impl_->c_row_ptr.begin(), impl_->c_row_ptr.end());
    impl_->c_col_idx_i32.assign(impl_->c_col_idx.begin(), impl_->c_col_idx.end());

    impl_->numeric_done = true;
    stats_.numeric_time_ms = now_ms() - t0;
    stats_.output_nnz = impl_->c_row_ptr.empty() ? 0 : impl_->c_row_ptr[m];
    return static_cast<int>(ErrorCode::SUCCESS);
}

int SpGEMM::compute() {
    const double t_all = now_ms();
    impl_->symbolic_done = false;
    impl_->numeric_done = false;

    if (config_.phase == SpGEMMPhase::FUSED) {
        int rc = compute_symbolic();
        if (rc) return rc;
        rc = compute_numeric();
        if (rc) return rc;
    } else if (config_.phase == SpGEMMPhase::SYMBOLIC) {
        return compute_symbolic();
    } else {
        int rc = compute_symbolic();
        if (rc) return rc;
        return compute_numeric();
    }

    stats_.total_time_ms = now_ms() - t_all;
    stats_.input_nnz_a = impl_->a_row_ptr.empty() ? 0 : impl_->a_row_ptr[impl_->m];
    stats_.input_nnz_b = impl_->b_row_ptr.empty() ? 0 : impl_->b_row_ptr[impl_->k];
    stats_.flops = 2ull * stats_.input_nnz_a * stats_.input_nnz_b /
                   std::max<uint64_t>(1, std::max(impl_->k, 1));
    stats_.memory_bytes =
        (stats_.input_nnz_a + stats_.input_nnz_b + stats_.output_nnz) * 16;
    stats_.compute_derived();
    return static_cast<int>(ErrorCode::SUCCESS);
}

int SpGEMM::autotune(const SpGEMMDescriptor& desc, int iterations) {
    // Empirical search over a small candidate set.
    const SpGEMMConfig base = recommend_config(desc);
    std::vector<SpGEMMConfig> candidates;

    candidates.push_back(base);

    if (base.use_cuda) {
        SpGEMMConfig c = base;
        c.cuda_threads_per_block = 128;
        candidates.push_back(c);
        c.cuda_threads_per_block = 512;
        candidates.push_back(c);
        c.cuda_threads_per_block = 1024;
        candidates.push_back(c);
    }
    if (base.use_openmp) {
        SpGEMMConfig c = base;
        c.openmp_threads = 1;
        candidates.push_back(c);
        c.openmp_threads = 2;
        candidates.push_back(c);
    }
    SpGEMMConfig c = base;
    c.algorithm = SpGEMMAlgorithm::HEAP;
    candidates.push_back(c);
    c.algorithm = SpGEMMAlgorithm::MERGE_PATH;
    candidates.push_back(c);

    double best = 1e300;
    SpGEMMConfig best_cfg = base;

    for (auto& cand : candidates) {
        SpGEMMConfig saved = config_;
        initialize(cand);
        // Re-apply matrices (they live in impl_, unaffected by config change)
        double total = 0.0;
        bool ok = true;
        for (int it = 0; it < iterations; ++it) {
            stats_.reset();
            if (compute() != static_cast<int>(ErrorCode::SUCCESS)) { ok = false; break; }
            total += stats_.total_time_ms;
        }
        if (ok && total / iterations < best) {
            best = total / iterations;
            best_cfg = cand;
        }
        config_ = saved;
    }

    config_ = best_cfg;
    return static_cast<int>(ErrorCode::SUCCESS);
}

std::unique_ptr<SpGEMM> create_spgemm(const SpGEMMConfig& config) {
    auto s = std::make_unique<SpGEMM>();
    s->initialize(config);
    return s;
}

std::vector<BenchmarkResult> run_spgemm_benchmark(
    const std::vector<SpGEMMDescriptor>& descriptors,
    const std::vector<SpGEMMConfig>& configs,
    int warmup_runs, int benchmark_runs)
{
    std::vector<BenchmarkResult> results;

    for (const auto& desc : descriptors) {
        // Generate representative matrices for the descriptor.
        auto A = generate_random_sparse(desc.m, desc.k,
                                        desc.nnz_a / static_cast<double>(desc.m * desc.k),
                                        MatrixFormat::COO);
        auto B = generate_random_sparse(desc.k, desc.n,
                                        desc.nnz_b / static_cast<double>(desc.k * desc.n),
                                        MatrixFormat::COO);

        for (const auto& cfg : configs) {
            BenchmarkResult r;
            r.config = cfg;
            r.descriptor = desc;

            try {
                auto spgemm = create_spgemm(cfg);
                spgemm->set_matrix_a<>(A.get());
                spgemm->set_matrix_b<>(B.get());

                for (int w = 0; w < warmup_runs; ++w) spgemm->compute();

                spgemm->reset_stats();
                for (int b = 0; b < benchmark_runs; ++b) spgemm->compute();
                r.stats = spgemm->stats();
                r.stats.total_time_ms /= std::max(benchmark_runs, 1);
                r.success = true;
                r.name = "m" + std::to_string(desc.m) + "_n" + std::to_string(desc.n) +
                         "_k" + std::to_string(desc.k);
            } catch (const std::exception& e) {
                r.success = false;
                r.error_message = e.what();
            }
            results.push_back(r);
        }
    }

    return results;
}

// ===========================================================================
// Explicit instantiations of the SpGEMM matrix-setter templates
// ===========================================================================

template int SpGEMM::set_matrix_a<double>(const MatrixBase*);
template int SpGEMM::set_matrix_a<float>(const MatrixBase*);
template int SpGEMM::set_matrix_b<double>(const MatrixBase*);
template int SpGEMM::set_matrix_b<float>(const MatrixBase*);
template int SpGEMM::set_matrix_c<double>(MatrixBase*);
template int SpGEMM::set_matrix_c<float>(MatrixBase*);

HPC_NAMESPACE_END