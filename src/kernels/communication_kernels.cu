// CUDA communication kernels: helper routines for the hybrid host/device
// communication path described by McFarland, Bellavita & Guidi (ICPE 2025).
//
// The hybrid scheme stages large messages through (pinned) host memory because
// an intra-node PCIe copy is cheaper than a device-direct NIC DMA, while small
// messages go device-to-device because the DMA engine's fixed cost dominates.
// These kernels implement the pieces of that path that are not expressible
// with plain cudaMemcpy: asynchronous bulk staging and pinned-buffer packing.

#include "hpc/communication.hpp"

#include <cuda_runtime.h>
#include <cstring>

namespace hpc {
namespace comm {

// ---------------------------------------------------------------------------
// Bulk device -> pinned-host staging using a 2-D tile copy so that several
// chunks are in flight concurrently.
// ---------------------------------------------------------------------------

cudaError_t stage_to_host_async(const void* d_src, void* h_dst, size_t bytes,
                                cudaStream_t stream, int chunks) {
    if (bytes == 0) return cudaSuccess;
    if (chunks < 1) chunks = 1;

    const size_t chunk = (bytes + static_cast<size_t>(chunks) - 1) /
                         static_cast<size_t>(chunks);

    for (int i = 0; i < chunks; ++i) {
        const size_t off = static_cast<size_t>(i) * chunk;
        if (off >= bytes) break;
        const size_t len = std::min(chunk, bytes - off);
        cudaError_t err = cudaMemcpyAsync(
            static_cast<char*>(h_dst) + off,
            static_cast<const char*>(d_src) + off,
            len, cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

cudaError_t stage_from_host_async(const void* h_src, void* d_dst, size_t bytes,
                                  cudaStream_t stream, int chunks) {
    if (bytes == 0) return cudaSuccess;
    if (chunks < 1) chunks = 1;

    const size_t chunk = (bytes + static_cast<size_t>(chunks) - 1) /
                         static_cast<size_t>(chunks);

    for (int i = 0; i < chunks; ++i) {
        const size_t off = static_cast<size_t>(i) * chunk;
        if (off >= bytes) break;
        const size_t len = std::min(chunk, bytes - off);
        cudaError_t err = cudaMemcpyAsync(
            static_cast<char*>(d_dst) + off,
            static_cast<const char*>(h_src) + off,
            len, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// Device-to-device copy split across chunks. Used by the GPU-direct path, which
// on a CUDA-aware MPI library turns into an RDMA/NIC transfer but for
// intra-node peers is a pure PCIe/NVLink hop.
// ---------------------------------------------------------------------------

cudaError_t copy_device_chunks(const void* d_src, void* d_dst, size_t bytes,
                               cudaStream_t stream, int chunks) {
    if (bytes == 0) return cudaSuccess;
    if (chunks < 1) chunks = 1;

    const size_t chunk = (bytes + static_cast<size_t>(chunks) - 1) /
                         static_cast<size_t>(chunks);

    for (int i = 0; i < chunks; ++i) {
        const size_t off = static_cast<size_t>(i) * chunk;
        if (off >= bytes) break;
        const size_t len = std::min(chunk, bytes - off);
        cudaError_t err = cudaMemcpyAsync(
            static_cast<char*>(d_dst) + off,
            static_cast<const char*>(d_src) + off,
            len, cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// Pinned host-buffer registration helper. Registering staging buffers with
// cudaHostRegister lets the driver use the DMA engine for the host copies.
// ---------------------------------------------------------------------------

cudaError_t register_pinned(void* ptr, size_t bytes) {
    if (!ptr || bytes == 0) return cudaSuccess;
    return cudaHostRegister(ptr, bytes, cudaHostRegisterDefault);
}

cudaError_t unregister_pinned(void* ptr) {
    if (!ptr) return cudaSuccess;
    return cudaHostUnregister(ptr);
}

// ---------------------------------------------------------------------------
// Copy CSR column indices from int32 to int64 in place on the device. The
// generic API traffics in 64-bit indices, but 32-bit indices halve the index
// traffic for matrices below 2^31 nonzeros, which is why the kernels are
// specialised on the index type.
// ---------------------------------------------------------------------------

template<typename OutT, typename InT>
__global__ void widen_indices_kernel(const InT* __restrict__ src,
                                     OutT* __restrict__ dst, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<OutT>(src[i]);
}

cudaError_t widen_indices(const void* src, void* dst, size_t n, size_t in_elem,
                          size_t out_elem) {
    if (n == 0) return cudaSuccess;
    if (in_elem == out_elem) {
        return cudaMemcpy(dst, src, n * in_elem, cudaMemcpyDeviceToDevice);
    }

    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);

    if (in_elem == 4 && out_elem == 8) {
        widen_indices_kernel<int64_t, int32_t>
            <<<grid, block>>>(static_cast<const int32_t*>(src),
                              static_cast<int64_t*>(dst), n);
    } else if (in_elem == 8 && out_elem == 4) {
        widen_indices_kernel<int32_t, int64_t>
            <<<grid, block>>>(static_cast<const int64_t*>(src),
                              static_cast<int32_t*>(dst), n);
    } else {
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Overlap helper: prefetch the next message's device->host staging while the
// current MPI transfer is still in flight.
// ---------------------------------------------------------------------------

cudaError_t prefetch_device_to_host(const void* d_src, void* h_dst, size_t bytes,
                                    cudaStream_t stream) {
    if (bytes == 0) return cudaSuccess;
    return cudaMemcpyAsync(h_dst, d_src, bytes, cudaMemcpyDeviceToHost, stream);
}

} // namespace comm
} // namespace hpc
