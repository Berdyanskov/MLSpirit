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

// 连续化(gather) kernel 的形状描述：≤8 维按值传入 kernel（驻留 constant 参数区，
// 省去为几个整数单独 cudaMalloc + HtoD 的开销）
struct StrideDesc {
    int ndim;
    int shape[8];
    int strides[8];  // 单位：元素（与 Tensor::strides_ 一致）
};

// dst[idx] = src[ idx 反推多维坐标后按 strides 的偏移 ]；src 为视图起始地址
void launch_contiguous_kernel(const void* src, void* dst, size_t numel,
                              const StrideDesc& desc, DataType dtype);

// ---- 逐元素算子（任务二）----
// 二元算子：输出形状为广播对齐结果，da/db 是 a/b 各自对齐后的 StrideDesc（广播维 0）
enum class ElementwiseBinaryOp { Add, Sub, Mul, Div };
void launch_elementwise_binary_kernel(const void* a, const void* b, void* dst, size_t numel,
                                      const StrideDesc& da, const StrideDesc& db,
                                      DataType dtype, ElementwiseBinaryOp op);
// 一元 exp：单操作数
void launch_exp_kernel(const void* src, void* dst, size_t numel,
                       const StrideDesc& desc, DataType dtype);
// 全归约：dst_scalar 指向设备上的单个元素（调用方负责清零）
void launch_sum_kernel(const void* src, void* dst_scalar, size_t numel,
                       const StrideDesc& desc, DataType dtype);

// ---- 形状归约（广播的逆运算）----
// sum_to 的形状描述：输入与输出按同一维数（= x.ndim）对齐，目标形状左侧补 1。
// 折叠维(out_shape[d]==1 && in_shape[d]>1)单独成表，按值传入 kernel。
struct SumToDesc {
    int ndim;            // 对齐后的维数
    int out_shape[8];    // 左补 1 后的目标形状
    int in_strides[8];   // 输入 views 的 strides（单位：元素）
    int n_collapsed;     // 折叠维个数
    int col_shape[8];    // 折叠维在输入中的大小
    int col_strides[8];  // 折叠维在输入中的 strides
};

// 以输出为中心：dst[idx] = Σ_{折叠维坐标组合} src[off_base + Σ c[d]·col_strides[d]]，
// fan_in = Π col_shape（折叠维组合的个数）；src 为视图起始地址，dst 连续
void launch_sum_to_kernel(const void* src, void* dst, size_t dst_numel, size_t fan_in,
                          const SumToDesc& desc, DataType dtype);

} // namespace mlspirit
