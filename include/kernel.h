#pragma once
#include "Tensor.hpp"

namespace mlspirit {

// a, b, c 均为设备指针；in-place 时 c 可与 a 相同。dtype 为元素类型。
void launch_add_kernel(void* a, void* b, void* c, size_t n, DataType dtype);

// C[M,N] = A[M,K] x B[K,N]；三者均为设备指针、行主序、连续 FP32
void launch_mm_kernel(const float* A, const float* B, float* C, int M, int N, int K);

// 批维描述（最多 8 个 batch 维）。按值传入 kernel，驻留参数区(constant memory)，
// 同 block 全员访问同一地址，命中 constant 缓存广播，无需额外 cudaMalloc
struct BatchInfo {
    int ndim;          // batch 维数，0 = 纯 2D
    int shape[8];      // 广播对齐后的 batch 形状
    int strides_a[8];  // A 的 batch strides（广播维为 0）
    int strides_b[8];  // B 的 batch strides（广播维为 0）
};

// Batched GEMM：对 z in [0, batch_count) 计算
//   C[z] = A[z] x B[z]，其中 A/B 的 batch 基址由 info 广播对齐，C 恒连续(z * M * N)
// A/B/C 为设备指针；矩阵维要求行主序连续
void launch_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K,
                          size_t batch_count, const BatchInfo& info);

} // namespace mlspirit