// ============================================================================
// contiguous.cu — 连续化(gather) kernel：把任意 strides 描述的视图物化为连续内存
//
// 每个线程负责一个输出元素：
//   dst 侧按线性下标写入，天然合并访存；
//   src 侧按 strides 聚集读取，合并程度取决于视图布局——
//   例如转置视图中相邻线程的读取相距 K 个元素（K 越大越不合并）。
//   这正是"转置拷贝"访存效率低的根源；生产实现会在 shared memory 里做
//   tile 重排让两端都合并，留作性能课的优化素材。
// ============================================================================
#include "kernel.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace mlspirit {

template <typename T>
__global__ void contiguous_kernel(const T* __restrict__ src, T* __restrict__ dst,
                                  size_t numel, StrideDesc desc) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    size_t src_off = 0, t = idx;
    for (int d = desc.ndim - 1; d >= 0; --d) {
        const size_t coord = t % desc.shape[d];
        src_off += coord * static_cast<size_t>(desc.strides[d]);
        t /= desc.shape[d];
    }
    dst[idx] = src[src_off];
}

void launch_contiguous_kernel(const void* src, void* dst, size_t numel,
                              const StrideDesc& desc, DataType dtype) {
    if (numel == 0) return;
    const int block = 256;
    const size_t grid = (numel + block - 1) / block;

    switch (dtype) {
    case DataType::FP32:
        contiguous_kernel<float><<<grid, block>>>(
            static_cast<const float*>(src), static_cast<float*>(dst), numel, desc);
        break;
    case DataType::FP16:
        contiguous_kernel<uint16_t><<<grid, block>>>(
            static_cast<const uint16_t*>(src), static_cast<uint16_t*>(dst), numel, desc);
        break;
    case DataType::INT8:
        contiguous_kernel<int8_t><<<grid, block>>>(
            static_cast<const int8_t*>(src), static_cast<int8_t*>(dst), numel, desc);
        break;
    default:
        contiguous_kernel<float><<<grid, block>>>(
            static_cast<const float*>(src), static_cast<float*>(dst), numel, desc);
        break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_contiguous_kernel: ") +
                                 cudaGetErrorString(err));
}

} // namespace mlspirit
