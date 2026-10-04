// ============================================================================
// elementwise.cu — 逐元素算子 CUDA kernel（任务二，骨架）
//
// 索引结构与 contiguous.cu 的 gather kernel 相同：每线程负责一个输出元素，
// 扁平下标 -> 多维坐标 -> 按 strides 求偏移；唯一的区别是二元算子有两组
// StrideDesc（a、b 各自的广播对齐描述），同一输出下标换算出两个源偏移。
// ============================================================================
#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace mlspirit {

// 单点求值：FP16 用半精度内建函数（不能把 bit pattern 当整数算），其余走原生运算符
template <typename T, ElementwiseBinaryOp OP>
__device__ __forceinline__ T apply_binary_op(T x, T y) {
    if constexpr (std::is_same_v<T, __half>) {
        if constexpr (OP == ElementwiseBinaryOp::Add)      return __hadd(x, y);
        else if constexpr (OP == ElementwiseBinaryOp::Sub) return __hsub(x, y);
        else if constexpr (OP == ElementwiseBinaryOp::Mul) return __hmul(x, y);
        else                                               return __hdiv(x, y);
    } else {
        if constexpr (OP == ElementwiseBinaryOp::Add)      return x + y;
        else if constexpr (OP == ElementwiseBinaryOp::Sub) return x - y;
        else if constexpr (OP == ElementwiseBinaryOp::Mul) return x * y;
        else                                               return x / y;
    }
}

// 二元算子 kernel 模板：OP 决定最后一行运算，索引逻辑与 OP 无关
template <typename T, ElementwiseBinaryOp OP>
__global__ void elementwise_binary_kernel(const T* __restrict__ a, const T* __restrict__ b,
                                          T* __restrict__ dst, size_t numel,
                                          StrideDesc da, StrideDesc db) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    size_t off_a = 0, off_b = 0, t = idx;
    for (int d = da.ndim - 1; d >= 0; --d) {
        const size_t coord = t % da.shape[d];
        off_a += coord * static_cast<size_t>(da.strides[d]);
        off_b += coord * static_cast<size_t>(db.strides[d]);
        t /= da.shape[d];
    }

    dst[idx] = apply_binary_op<T, OP>(a[off_a], b[off_b]);
}

// 一元 exp kernel：单操作数版本，索引逻辑与 gather 一致（视图直读）。
// 无块内协作（无 __syncthreads），越界线程直接 return 是安全的——
// 与 sum_kernel 的"全员必须到场"形成对照，正好体现两条边界规则各自的适用条件。
template <typename T>
__global__ void exp_kernel(const T* __restrict__ src, T* __restrict__ dst,
                           size_t numel, StrideDesc desc) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    size_t off = 0, t = idx;
    for (int d = desc.ndim - 1; d >= 0; --d) {
        const size_t coord = t % desc.shape[d];
        off += coord * static_cast<size_t>(desc.strides[d]);
        t /= desc.shape[d];
    }
    dst[idx] = static_cast<T>(expf(static_cast<float>(src[off])));
}


__device__ inline float warp_reduce_sum(float val) {
    // 蝶形归约：lane i 每轮读到 lane i+offset 的值并累加，log2(32)=5 轮后
    // lane 0 持有整个 warp 的总和；0xffffffff 表示 warp 内 32 个 lane 全员参与
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline float block_reduce_sum(float val) {
    __shared__ float shared[32]; // 每个 warp 一个坑位
    const int lane = threadIdx.x % 32;
    const int wid  = threadIdx.x / 32;

    val = warp_reduce_sum(val);          // 1. warp 内部归约，lane 0 持有 warp 和
    if (lane == 0) shared[wid] = val;    // 2. 各 warp 的 lane 0 写入 shared memory
    __syncthreads();

    // 3. 只有第 0 个 warp 读回 ≤32 个部分和，再做一次 warp 归约
    val = (static_cast<int>(threadIdx.x) < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

template <typename T>
__global__ void sum_kernel(const T* __restrict__ src, float* __restrict__ dst_scalar,
                           size_t numel, StrideDesc desc) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    float sum = 0.0f;
    for (size_t i = idx; i < numel; i += stride) {
        size_t off = 0, t = i;
        for (int d = desc.ndim - 1; d >= 0; --d) {
            const size_t coord = t % desc.shape[d];
            off += coord * static_cast<size_t>(desc.strides[d]);
            t /= desc.shape[d];
        }
        sum += static_cast<float>(src[off]);
    }

    sum = block_reduce_sum(sum);
    // 块间无同步原语，各 block 的 0 号线程原子累加到全局标量
    if (threadIdx.x == 0) atomicAdd(dst_scalar, sum);
}

// ---------------- launcher：grid/block 计算 + dtype × op 二层分派 ----------------

// 内层按 op 分派：运行时 op 映射到 4 个模板实例（模板实参必须是编译期常量）
template <typename T>
static void dispatch_binary_op(const void* a, const void* b, void* dst, size_t numel,
                               const StrideDesc& da, const StrideDesc& db,
                               ElementwiseBinaryOp op, size_t grid, int block) {
    const T* ta = static_cast<const T*>(a);
    const T* tb = static_cast<const T*>(b);
    T* td = static_cast<T*>(dst);
    switch (op) {
    case ElementwiseBinaryOp::Add:
        elementwise_binary_kernel<T, ElementwiseBinaryOp::Add><<<grid, block>>>(ta, tb, td, numel, da, db);
        break;
    case ElementwiseBinaryOp::Sub:
        elementwise_binary_kernel<T, ElementwiseBinaryOp::Sub><<<grid, block>>>(ta, tb, td, numel, da, db);
        break;
    case ElementwiseBinaryOp::Mul:
        elementwise_binary_kernel<T, ElementwiseBinaryOp::Mul><<<grid, block>>>(ta, tb, td, numel, da, db);
        break;
    case ElementwiseBinaryOp::Div:
        elementwise_binary_kernel<T, ElementwiseBinaryOp::Div><<<grid, block>>>(ta, tb, td, numel, da, db);
        break;
    }
}

void launch_elementwise_binary_kernel(const void* a, const void* b, void* dst, size_t numel,
                                      const StrideDesc& da, const StrideDesc& db,
                                      DataType dtype, ElementwiseBinaryOp op) {
    if (numel == 0) return;
    const int block = 256;
    const size_t grid = (numel + block - 1) / block;

    switch (dtype) {
    case DataType::FP32:
        dispatch_binary_op<float>(a, b, dst, numel, da, db, op, grid, block);
        break;
    case DataType::FP16:
        dispatch_binary_op<__half>(a, b, dst, numel, da, db, op, grid, block);
        break;
    case DataType::INT8:
        dispatch_binary_op<int8_t>(a, b, dst, numel, da, db, op, grid, block);
        break;
    default:
        dispatch_binary_op<float>(a, b, dst, numel, da, db, op, grid, block);
        break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_elementwise_binary_kernel: ") +
                                 cudaGetErrorString(err));
}

void launch_exp_kernel(const void* src, void* dst, size_t numel,
                       const StrideDesc& desc, DataType dtype) {
    if (numel == 0) return;
    if (dtype != DataType::FP32)
        throw std::runtime_error("launch_exp_kernel: FP32 only for now (task 2 first version)");

    const int block = 256;
    const size_t grid = (numel + block - 1) / block;
    exp_kernel<float><<<grid, block>>>(static_cast<const float*>(src),
                                       static_cast<float*>(dst), numel, desc);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_exp_kernel: ") +
                                 cudaGetErrorString(err));
}

void launch_sum_kernel(const void* src, void* dst_scalar, size_t numel,
                       const StrideDesc& desc, DataType dtype) {
    if (numel == 0) return;
    if (dtype != DataType::FP32)
        throw std::runtime_error("launch_sum_kernel: FP32 only for now (task 2 first version)");

    const int block = 256;
    const size_t grid = (numel + block - 1) / block;
    sum_kernel<float><<<grid, block>>>(static_cast<const float*>(src),
                                       static_cast<float*>(dst_scalar), numel, desc);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_sum_kernel: ") +
                                 cudaGetErrorString(err));
}

} // namespace mlspirit
