#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "kernel.h"
#include "Tensor.hpp"

namespace mlspirit {

// 使用模板核函数，一种实现覆盖 FP32/FP16/INT8，无需为每种类型手写一份
template<typename T>
__global__ void add_kernel(const T* a, const T* b, T* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// FP16 特化：用 __hadd 做半精度加法（若环境支持 __half 的 operator+ 也可直接用）
__global__ void add_kernel_half(const __half* a, const __half* b, __half* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = __hadd(a[i], b[i]);
}

void launch_add_kernel(void* a, void* b, void* c, size_t n, DataType dtype) {
    if (n == 0) return;
    int in = static_cast<int>(n);
    const int block = 256;
    const int grid = (in + block - 1) / block;

    switch (dtype) {
    case DataType::FP32:
        add_kernel<float><<<grid, block>>>(static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(c), in);
        break;
    case DataType::FP16:
        add_kernel_half<<<grid, block>>>(static_cast<const __half*>(a), static_cast<const __half*>(b), static_cast<__half*>(c), in);
        break;
    case DataType::INT8:
        add_kernel<int8_t><<<grid, block>>>(static_cast<const int8_t*>(a), static_cast<const int8_t*>(b), static_cast<int8_t*>(c), in);
        break;
    default:
        add_kernel<float><<<grid, block>>>(static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(c), in);
        break;
    }
}

} // namespace mlspirit
