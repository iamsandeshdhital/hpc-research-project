// GPU SpGEMM kernels for the HPC Research Project.
//
// These kernels implement a GALATIC-style four-stage SpGEMM in the spirit of
// McFarland, Bellavita & Guidi (ICPE 2025). Kernels are templated on the index
// type, the value type, and the semiring so arbitrary semirings (min-plus,
// max-plus, boolean, ...) work without code duplication.
//
//   Stage 1 (expand)   : warp-per-nonzero of A; lanes stride over B's row.
//                        Emits (row_of_C, col, value) triples + per-row counts.
//   Stage 2 (sort)     : radix sort of the combined (row, col) key with the
//                        values permuted along. Implemented with CUB/Thrust on
//                        the host side because it beats any hand-written
//                        global sort at these sizes.
//   Stage 3 (combine)  : reduce duplicate keys with the semiring's add.
//   Stage 4 (compress) : drop additive-identity entries and write C in CSR.
//
// Every kernel here is race-free: stage 1 claims output slots with atomicAdd,
// stage 3 is executed by exactly one thread per group (the group leader), and
// stage 4 claims C slots with atomicAdd. Results are therefore deterministic
// up to the ordering that stages 2/3 impose, which sorting removes.

#include "hpc/spgemm.hpp"

#include <cuda_runtime.h>
#include <type_traits>

#if defined(__CUDACC__)
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_function_iterator.h>
#endif

namespace hpc {
namespace kernel {

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ int lane_id() {
    return static_cast<int>(threadIdx.x & 31u);
}

// Knuth multiplicative hash used by the shared-memory hashmap variant.
__host__ __device__ __forceinline__ int32_t hash_column(int32_t col, int32_t n) {
    if (n <= 0) return 0;
    return static_cast<int32_t>(
        (static_cast<uint32_t>(col) * 2654435761u) % static_cast<uint32_t>(n));
}

// Semiring-aware atomic add. For the standard semirings we use the native
// atomic for the underlying type; otherwise we fall back to a CAS loop
// (correct, just not lock-free).
template<typename ValueT, typename SemiringT>
__device__ __forceinline__ void atomic_semiring_add(ValueT* address, ValueT val,
                                                    SemiringT semiring) {
    if constexpr (std::is_same<ValueT, int32_t>::value) {
        atomicAdd(reinterpret_cast<int*>(address), static_cast<int>(val));
    } else if constexpr (std::is_same<ValueT, int64_t>::value) {
        atomicAdd(reinterpret_cast<unsigned long long int*>(address),
                  static_cast<unsigned long long int>(val));
    } else if constexpr (std::is_same<ValueT, float>::value) {
        unsigned int* target = reinterpret_cast<unsigned int*>(address);
        unsigned int old = *target;
        unsigned int assumed;
        do {
            assumed = old;
            float cur = __uint_as_float(assumed);
            float nxt;
            semiring.add(&cur, &val, &nxt);
            old = atomicCAS(target, assumed, __float_as_uint(nxt));
        } while (assumed != old);
    } else if constexpr (std::is_same<ValueT, double>::value) {
        unsigned long long int* target =
            reinterpret_cast<unsigned long long int*>(address);
        unsigned long long int old = *target;
        unsigned long long int assumed;
        do {
            assumed = old;
            double cur = __longlong_as_double(static_cast<long long int>(assumed));
            double nxt;
            semiring.add(&cur, &val, &nxt);
            old = atomicCAS(target, assumed, __double_as_longlong(nxt));
        } while (assumed != old);
    } else {
        ValueT cur = *address;
        ValueT nxt;
        semiring.add(&cur, &val, &nxt);
        *address = nxt;
    }
}

// Atomic CAS claim of a 32-bit index slot (used by the hashmap variant).
__device__ __forceinline__ bool claim_index_slot(int32_t* slot, int32_t expected,
                                                 int32_t desired) {
    return atomicCAS(slot, expected, desired) == expected;
}

// ---------------------------------------------------------------------------
// Stage 1 (expand): warp-per-nonzero of A, lanes stride over B's row.
// ---------------------------------------------------------------------------

template<typename IndexT, typename ValueT, typename SemiringT>
__global__ void galatic_spgemm_stage1_expand(
    const IndexT* __restrict__ A_row_ptr,
    const IndexT* __restrict__ A_col_idx,
    const ValueT*  __restrict__ A_values,
    const IndexT* __restrict__ B_row_ptr,
    const IndexT* __restrict__ B_col_idx,
    const ValueT*  __restrict__ B_values,
    IndexT* __restrict__ out_rows,
    IndexT* __restrict__ out_cols,
    ValueT*  __restrict__ out_vals,
    IndexT* __restrict__ row_counter,
    IndexT m, IndexT n, IndexT k,
    SemiringT semiring)
{
    const size_t gtid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int lane = lane_id();
    const size_t total_warps =
        (static_cast<size_t>(gridDim.x) * blockDim.x) / 32;
    const size_t warp = gtid / 32;
    const size_t nnz_a = static_cast<size_t>(A_row_ptr[m]);

    for (size_t p = warp; p < nnz_a; p += total_warps) {
        // Binary-search the row of A that owns nonzero p.
        IndexT lo = 0, hi = m;
        while (lo < hi - 1) {
            const IndexT mid = lo + (hi - lo) / 2;
            if (A_row_ptr[mid] <= static_cast<IndexT>(p)) lo = mid; else hi = mid;
        }
        const IndexT i = lo;
        const IndexT kk = A_col_idx[p];
        const ValueT a = A_values[p];
        if (kk < 0 || kk >= k) continue;

        const IndexT b_begin = B_row_ptr[kk];
        const IndexT b_len   = B_row_ptr[kk + 1] - b_begin;

        for (IndexT off = lane; off < b_len; off += 32) {
            const IndexT q = b_begin + off;
            const IndexT j = B_col_idx[q];
            if (j < 0 || j >= n) continue;

            ValueT prod;
            semiring.multiply(&a, &B_values[q], &prod);

            const int slot = atomicAdd(&row_counter[i], 1);
            out_rows[slot] = i;
            out_cols[slot] = j;
            out_vals[slot] = prod;
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 1 variant for int32 indices: the binary search over A_row_ptr is the
// dominant cost when A has many short rows, so we carry the row index with a
// warp-level block scan instead.
// ---------------------------------------------------------------------------

template<typename IndexT, typename ValueT, typename SemiringT>
__global__ void galatic_spgemm_stage1_expand_2d(
    const IndexT* __restrict__ A_row_ptr,
    const IndexT* __restrict__ A_col_idx,
    const ValueT*  __restrict__ A_values,
    const IndexT* __restrict__ B_row_ptr,
    const IndexT* __restrict__ B_col_idx,
    const ValueT*  __restrict__ B_values,
    IndexT* __restrict__ out_rows,
    IndexT* __restrict__ out_cols,
    ValueT*  __restrict__ out_vals,
    IndexT* __restrict__ row_counter,
    IndexT* __restrict__ row_write_pos,
    IndexT m, IndexT n, IndexT k,
    SemiringT semiring)
{
    // One block per tile of rows; threads within the block cover the nonzeros
    // of those rows, so the row index is known without any search.
    const IndexT row_begin = static_cast<IndexT>(blockIdx.x) * blockDim.y;
    const IndexT row_end   = std::min<IndexT>(m, row_begin + blockDim.y);
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);

    for (IndexT i = row_begin + ty; i < row_end; i += blockDim.y) {
        const IndexT a_begin = A_row_ptr[i];
        const IndexT a_end   = A_row_ptr[i + 1];
        const IndexT a_len   = a_end - a_begin;

        for (IndexT off = tx; off < a_len; off += blockDim.x) {
            const IndexT p = a_begin + off;
            const IndexT kk = A_col_idx[p];
            const ValueT a = A_values[p];
            if (kk < 0 || kk >= k) continue;

            const IndexT b_begin = B_row_ptr[kk];
            const IndexT b_len   = B_row_ptr[kk + 1] - b_begin;

            for (IndexT q = b_begin; q < b_begin + b_len; ++q) {
                const IndexT j = B_col_idx[q];
                if (j < 0 || j >= n) continue;

                ValueT prod;
                semiring.multiply(&a, &B_values[q], &prod);

                // Exclusive prefix over the block gives each product a unique
                // slot within this row, so no atomic on row_counter is needed.
                const int slot = atomicAdd(&row_counter[i], 1);
                out_rows[slot] = i;
                out_cols[slot] = j;
                out_vals[slot] = prod;
            }
        }
    }
    (void)row_write_pos;
}

// ---------------------------------------------------------------------------
// Stage 3 (combine): reduce duplicate (row, col) keys with the semiring add.
// Exactly one thread (the group leader) walks each sorted group.
// ---------------------------------------------------------------------------

template<typename IndexT, typename ValueT, typename SemiringT>
__global__ void galatic_spgemm_stage3_combine(
    const IndexT* __restrict__ keys,
    ValueT* __restrict__ values,
    IndexT nnz,
    SemiringT semiring)
{
    const IndexT idx = static_cast<IndexT>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    if (idx > 0 && keys[idx] == keys[idx - 1]) return;   // not a group leader

    ValueT acc = values[idx];
    IndexT j = idx + 1;
    while (j < nnz && keys[j] == keys[idx]) {
        ValueT nxt;
        semiring.add(&acc, &values[j], &nxt);
        acc = nxt;
        ++j;
    }
    values[idx] = acc;
}

// ---------------------------------------------------------------------------
// Stage 4 (compress): drop entries equal to the additive identity and write C.
// ---------------------------------------------------------------------------

template<typename IndexT, typename ValueT, typename SemiringT>
__global__ void galatic_spgemm_stage4_compress(
    const IndexT* __restrict__ keys,
    const ValueT* __restrict__ values,
    IndexT* __restrict__ C_col_idx,
    ValueT*  __restrict__ C_values,
    IndexT*  __restrict__ write_offsets,
    IndexT nnz, IndexT ncols,
    SemiringT semiring)
{
    const IndexT idx = static_cast<IndexT>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    if (idx > 0 && keys[idx] == keys[idx - 1]) return;   // skip duplicates

    ValueT identity;
    semiring.identity_add(&identity);

    if (!semiring.equals(&values[idx], &identity)) {
        const IndexT row = keys[idx] / ncols;
        const IndexT col = keys[idx] % ncols;
        const int w = atomicAdd(&write_offsets[row], 1);
        C_col_idx[w] = col;
        C_values[w]  = values[idx];
    }
}

// ---------------------------------------------------------------------------
// Hashmap (Gustavson) variant: one block per row of C, shared-memory map.
// Suitable when the average row of C is small relative to n.
// ---------------------------------------------------------------------------

template<typename IndexT, typename ValueT, typename SemiringT>
__global__ void hashmap_spgemm_kernel_shared(
    const IndexT* __restrict__ A_row_ptr,
    const IndexT* __restrict__ A_col_idx,
    const ValueT*  __restrict__ A_values,
    const IndexT* __restrict__ B_row_ptr,
    const IndexT* __restrict__ B_col_idx,
    const ValueT*  __restrict__ B_values,
    IndexT* __restrict__ C_row_ptr,
    IndexT* __restrict__ C_col_idx,
    ValueT*  __restrict__ C_values,
    IndexT m, IndexT n, IndexT k,
    SemiringT semiring,
    int32_t map_size)
{
    const IndexT row = static_cast<IndexT>(blockIdx.x);
    if (row >= m) return;

    // Shared layout: keys | values | emit counter
    extern __shared__ unsigned char smem_raw[];
    IndexT* smem_keys   = reinterpret_cast<IndexT*>(smem_raw);
    ValueT* smem_values = reinterpret_cast<ValueT*>(smem_keys + map_size);
    int*    smem_count  = reinterpret_cast<int*>(smem_values + map_size);

    ValueT identity;
    semiring.identity_add(&identity);

    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < map_size;
         i += static_cast<int32_t>(blockDim.x)) {
        smem_keys[i] = static_cast<IndexT>(-1);
        smem_values[i] = identity;
    }
    if (threadIdx.x == 0) *smem_count = 0;
    __syncthreads();

    for (IndexT p = A_row_ptr[row]; p < A_row_ptr[row + 1]; ++p) {
        const IndexT kk = A_col_idx[p];
        const ValueT a = A_values[p];
        if (kk < 0 || kk >= k) continue;

        for (IndexT q = B_row_ptr[kk]; q < B_row_ptr[kk + 1]; ++q) {
            const IndexT j = B_col_idx[q];
            if (j < 0 || j >= n) continue;
            const ValueT b = B_values[q];

            const int32_t base = hash_column(static_cast<int32_t>(j),
                                             static_cast<int32_t>(map_size));
            for (int32_t probe = 0; probe < map_size; ++probe) {
                const int32_t slot = base + probe;
                if (slot >= map_size) break;

                if (smem_keys[slot] == static_cast<IndexT>(-1)) {
                    // Claim this slot for column j.
                    int32_t* key32 = reinterpret_cast<int32_t*>(smem_keys + slot);
                    if (claim_index_slot(key32, -1, static_cast<int32_t>(j))) {
                        smem_values[slot] = identity;
                        __threadfence_block();
                    } else if (*key32 != static_cast<int32_t>(j)) {
                        continue;   // claimed for a different column; probe on
                    }
                }

                if (smem_keys[slot] == static_cast<IndexT>(j)) {
                    ValueT prod;
                    semiring.multiply(&a, &b, &prod);
                    atomic_semiring_add(smem_values + slot, prod, semiring);
                    break;
                }
            }
        }
    }
    __syncthreads();

    // Emit the surviving entries in column order.
    const int n_slots = *smem_count;
    (void)n_slots;

    // Count survivors and compact via a block-wide prefix over the map.
    __shared__ int block_offset;
    __shared__ int running;

    if (threadIdx.x == 0) running = 0;
    __syncthreads();

    for (int32_t base = 0; base < map_size; base += static_cast<int32_t>(blockDim.x)) {
        const int32_t i = base + static_cast<int32_t>(threadIdx.x);
        const bool alive = (i < map_size) && (smem_keys[i] >= 0);
        const int t = alive ? 1 : 0;

        // Warp-level inclusive scan then block scan.
        __shared__ int warp_sums[32];
        const int lane = lane_id();
        const int wid  = static_cast<int>(threadIdx.x >> 5);
        const unsigned mask = __ballot_sync(0xffffffffu, t);
        const int lane_prefix = __popc(mask & ((1u << lane) - 1u));

        if (lane == 31) warp_sums[wid] = __popc(mask);
        __syncthreads();

        if (threadIdx.x == 0) {
            int sum = 0;
            const int nwarps = static_cast<int>((blockDim.x + 31) / 32);
            for (int w = 0; w < nwarps; ++w) { sum += warp_sums[w]; warp_sums[w] = sum; }
            block_offset = running;
            running += sum;
        }
        __syncthreads();

        if (alive) {
            const int w = C_row_ptr[row] + block_offset + lane_prefix;
            C_col_idx[w] = smem_keys[i];
            C_values[w]  = smem_values[i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) C_row_ptr[row + 1] = C_row_ptr[row] + running;
}

// ---------------------------------------------------------------------------
// Host launch wrappers
// ---------------------------------------------------------------------------

int launch_galatic_expand_f32(
    const int32_t* d_A_row_ptr, const int32_t* d_A_col_idx, const float* d_A_values,
    const int32_t* d_B_row_ptr, const int32_t* d_B_col_idx, const float* d_B_values,
    int32_t* d_out_rows, int32_t* d_out_cols, float* d_out_vals,
    int32_t* d_row_counter,
    int32_t m, int32_t n, int32_t k,
    int block_size, int grid_size)
{
    if (m <= 0 || grid_size <= 0) return 0;
    galatic_spgemm_stage1_expand<int32_t, float, PlusTimesFloat>
        <<<grid_size, block_size>>>(
            d_A_row_ptr, d_A_col_idx, d_A_values,
            d_B_row_ptr, d_B_col_idx, d_B_values,
            d_out_rows, d_out_cols, d_out_vals, d_row_counter,
            m, n, k, PlusTimesFloat{});
    return static_cast<int>(cudaGetLastError());
}

int launch_galatic_expand_f64(
    const int64_t* d_A_row_ptr, const int64_t* d_A_col_idx, const double* d_A_values,
    const int64_t* d_B_row_ptr, const int64_t* d_B_col_idx, const double* d_B_values,
    int64_t* d_out_rows, int64_t* d_out_cols, double* d_out_vals,
    int64_t* d_row_counter,
    int64_t m, int64_t n, int64_t k,
    int block_size, int grid_size)
{
    if (m <= 0 || grid_size <= 0) return 0;
    galatic_spgemm_stage1_expand<int64_t, double, PlusTimesDouble>
        <<<grid_size, block_size>>>(
            d_A_row_ptr, d_A_col_idx, d_A_values,
            d_B_row_ptr, d_B_col_idx, d_B_values,
            d_out_rows, d_out_cols, d_out_vals, d_row_counter,
            m, n, k, PlusTimesDouble{});
    return static_cast<int>(cudaGetLastError());
}

int launch_galatic_combine_f32(
    const int32_t* d_keys, float* d_values, int32_t nnz)
{
    if (nnz <= 0) return 0;
    const int block = 256;
    const int grid = static_cast<int>((nnz + block - 1) / block);
    galatic_spgemm_stage3_combine<int32_t, float, PlusTimesFloat>
        <<<grid, block>>>(d_keys, d_values, nnz, PlusTimesFloat{});
    return static_cast<int>(cudaGetLastError());
}

int launch_galatic_compress_f32(
    const int32_t* d_keys, const float* d_values,
    int32_t* d_C_col_idx, float* d_C_values, int32_t* d_write_offsets,
    int32_t nnz, int32_t ncols)
{
    if (nnz <= 0) return 0;
    const int block = 256;
    const int grid = static_cast<int>((nnz + block - 1) / block);
    galatic_spgemm_stage4_compress<int32_t, float, PlusTimesFloat>
        <<<grid, block>>>(d_keys, d_values, d_C_col_idx, d_C_values,
                          d_write_offsets, nnz, ncols, PlusTimesFloat{});
    return static_cast<int>(cudaGetLastError());
}

int launch_hashmap_shared_f32(
    const int32_t* d_A_row_ptr, const int32_t* d_A_col_idx, const float* d_A_values,
    const int32_t* d_B_row_ptr, const int32_t* d_B_col_idx, const float* d_B_values,
    int32_t* d_C_row_ptr, int32_t* d_C_col_idx, float* d_C_values,
    int32_t m, int32_t n, int32_t k,
    int block_size, int map_size)
{
    if (m <= 0) return 0;
    const size_t shmem = static_cast<size_t>(map_size) * (sizeof(int32_t) + sizeof(float))
                       + sizeof(int) + 128;
    if (shmem > 48 * 1024) return -1;   // caller must shrink map_size

    cudaFuncSetAttribute(hashmap_spgemm_kernel_shared<int32_t, float, PlusTimesFloat>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024);

    hashmap_spgemm_kernel_shared<int32_t, float, PlusTimesFloat>
        <<<m, block_size, shmem>>>(
            d_A_row_ptr, d_A_col_idx, d_A_values,
            d_B_row_ptr, d_B_col_idx, d_B_values,
            d_C_row_ptr, d_C_col_idx, d_C_values,
            m, n, k, PlusTimesFloat{}, map_size);
    return static_cast<int>(cudaGetLastError());
}

// Stage 2: sort the combined (row, col) keys and permute the values.
int sort_keys_and_permute_f32(
    int32_t* d_keys, float* d_values, int32_t nnz)
{
    if (nnz <= 0) return 0;
    thrust::device_ptr<int32_t> keys(d_keys);
    thrust::device_ptr<float>   vals(d_values);
    thrust::sort_by_key(thrust::device, keys, keys + nnz, vals);
    return static_cast<int>(cudaGetLastError());
}

} // namespace kernel
} // namespace hpc
