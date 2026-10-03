// ============================================================================
// cuda_matmul.cu — Tiled GEMM（FP32, 行主序, 连续内存）
//
// 算法：C = A x B，A:[M,K] B:[K,N] C:[M,N]
//
// 性能来自三级存储复用（这是理解这个文件的主线）：
//   Global Memory (HBM, ~TB/s 但延迟几百周期)
//     -> Shared Memory (片上, 每 block 独享, 带宽高一个量级)
//       -> Register (每线程私有, 零延迟)
//
// 分块结构（自顶向下三层）：
//   1. C 被划分成 BM x BN = 128 x 128 的大 Tile，每个 Block 负责一整块；
//   2. Block 内 256 个线程排成 16 x 16 网格，每个线程负责 TM x TN = 8 x 8 的
//      输出子块（Register Blocking，把内层复用从 SMEM 再推进到寄存器）；
//   3. 沿 K 轴以 BK = 8 为步长推进：每次把 A 的 128x8 行块、B 的 8x128 列块
//      协同搬进 Shared Memory，算完再搬下一块。
//
// 一个 Block 的输出 Tile 与线程分工示意：
//        <------------ BN = 128 ------------>
//   ^    +--------+--------+-----+--------+
//   |    | t(0,0)  | t(0,1) | ... | t(0,15) |   每格 = 一个线程负责的 8x8 子块
//  BM=128+--------+--------+-----+--------+
//   |    | t(1,0)  | ...                                          |
//   v    +--------+---------------------------------------------+
//        (共 16x16 = 256 线程)
// ============================================================================

#include <cuda_runtime.h>
#include "kernel.h"
#include "Tensor.hpp"
#include <stdexcept>
#include <algorithm>

// Tile 尺寸配置（均为编译期常量，便于 #pragma unroll 和寄存器分配）
#define BM 128   // Block 负责的输出行数
#define BN 128   // Block 负责的输出列数
#define BK 8     // 每次沿 K 轴搬运/计算的深度
#define TM 8     // 每个线程负责的输出行数
#define TN 8     // 每个线程负责的输出列数

namespace mlspirit {

// Block 线程布局：(BN/TN, BM/TM) = (16, 16) = 256 线程/Block。
// 单块 2D GEMM 核心抽成 __device__ 函数：2D kernel 直接调用；
// batched kernel 只负责算 batch 偏移，然后复用本函数，核心逻辑不写第二遍
__device__ void gemm_tile_core(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K)
{
    // ---------------- 1. 定位：本 Block / 本线程负责谁 ----------------
    const int bx = blockIdx.x;   // block 的列号（沿 N 方向）
    const int by = blockIdx.y;   // block 的行号（沿 M 方向）

    const int tx = threadIdx.x;  // 线程在 block 内的列号, 0..15
    const int ty = threadIdx.y;  // 线程在 block 内的行号, 0..15

    // 展平成 1D 线程号 (0..255)。搬运 shared memory 时把全 block 线程当作
    // 一维流水线使用，tid 连续 <=> 全局地址连续 <=> 访存合并(coalescing)
    const int tid = ty * blockDim.x + tx;

    // ---------------- 2. Shared Memory：本 block 的两块中转站 ----------------
    // s_a: A 的当前 K 块, 128x8 = 1024 floats; s_b: B 的当前 K 块, 8x128 = 1024 floats
    // 合计 8KB / block（H100/A100 每 SM 有 100+KB SMEM，不是瓶颈）
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    // ---------------- 3. 寄存器：线程私有的工作区 ----------------
    // 累加器：本线程负责的 8x8=64 个输出元素，整个 K 循环期间一直住在寄存器里
    float accumulate[TM][TN] = {0.0f};
    // 当前 k 步用到的一行 A 切条(8 个)和一列 B 切条(8 个)，先取进寄存器再做
    // 8x8 次乘加 —— 每个 SMEM 读取被复用 TN(或 TM) 次，这就是 Register Blocking
    float reg_a[TM];
    float reg_b[TN];

    // ---------------- 4. 协同搬运的分工映射 ----------------
    // s_a 共 128x8=1024 个元素 / 256 线程 = 每人 4 个。
    // a_tile_row = tid / 8 (0..31): tid 相邻 => 列相邻 => 读 A 时地址连续(合并)
    const int a_tile_row = tid / BK;
    const int a_tile_col = tid % BK;

    // s_b 共 8x128=1024 个元素 / 256 线程 = 每人 4 个。
    // b_tile_col = tid % 128: 半个 block 正好铺满 B 的一整行 128 floats
    // (=512 字节，完美的合并访问，一个 warp 一条 cache line 事务)
    const int b_tile_row = tid / BN;
    const int b_tile_col = tid % BN;

    // ---------------- 5. 主循环：沿 K 轴每次推进 BK=8 ----------------
    for (int ph = 0; ph < (K + BK - 1) / BK; ++ph) {

        // ---- 5a. 协同加载 A 的 128x8 行块进 s_a ----
        // 每轮循环全体 256 线程覆盖 32 行 x 8 列 = 256 元素；128 行需 128/32=4 轮。
        // 步进 = 256 线程 / 每行 8 列 = 32 行/轮
        #pragma unroll
        for (int i = 0; i < BM; i += (blockDim.x * blockDim.y) / BK) {
            int cur_row = by * BM + a_tile_row + i;   // A 的全局行
            int cur_col = ph * BK + a_tile_col;       // A 的全局列(K 轴第 ph 块)
            if (cur_row < M && cur_col < K) {
                s_a[a_tile_row + i][a_tile_col] = A[cur_row * K + cur_col];
            } else {
                // 越界(矩阵尺寸不是整块倍数)填 0：0 参与乘加不改变结果，
                // 这样 kernel 主体完全不需要分支，一律按整块算
                s_a[a_tile_row + i][a_tile_col] = 0.0f;
            }
        }

        // ---- 5b. 协同加载 B 的 8x128 列块进 s_b ----
        // 每轮全体线程覆盖 2 行 x 128 列 = 256 元素；8 行需 8/2=4 轮。
        // 步进 = 256 线程 / 每行 128 列 = 2 行/轮
        #pragma unroll
        for (int i = 0; i < BK; i += (blockDim.x * blockDim.y) / BN) {
            int cur_row = ph * BK + b_tile_row + i;   // B 的全局行(K 轴第 ph 块)
            int cur_col = bx * BN + b_tile_col;       // B 的全局列
            if (cur_row < K && cur_col < N) {
                s_b[b_tile_row + i][b_tile_col] = B[cur_row * N + cur_col];
            } else {
                s_b[b_tile_row + i][b_tile_col] = 0.0f;
            }
        }

        // 屏障①：必须等全 block 都把数据搬完，才能开始用 s_a/s_b 计算。
        // 竞态形态：某线程搬得慢，别的线程已经读到旧数据/默认值。
        __syncthreads();

        // ---- 5c. 当前 K 块的 8 步乘加（全部命中 SMEM + 寄存器）----
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // 本线程子块在 tile 内的左上角是 (ty*TM, tx*TN)。
            // 取第 k 步要用的 A 行切条 s_a[ty*8 .. ty*8+7][k] 进寄存器
            #pragma unroll
            for (int index = 0; index < TM; ++index) {
                reg_a[index] = s_a[ty * TM + index][k];
            }
            // 取 B 列切条 s_b[k][tx*8 .. tx*8+7] 进寄存器
            #pragma unroll
            for (int index = 0; index < TN; ++index) {
                reg_b[index] = s_b[k][tx * TN + index];
            }

            // 外积累加：reg_a(8 个数) x reg_b(8 个数) -> 64 次 FMA，
            // 全程只碰寄存器。BK 步循环结束 = 当前 K 块的贡献全部累加完
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accumulate[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        // 屏障②：必须等全 block 都用完 s_a/s_b，才能开始搬下一个 K 块。
        // 竞态形态：某线程算得慢，别的线程已经把 s_a 覆盖成下一块数据。
        __syncthreads();
    }

    // ---------------- 6. 写回：寄存器里的 8x8 结果落回 C ----------------
    // accumulate[i][j] 的 C 全局坐标 = (by*BM + ty*TM + i, bx*BN + tx*TN + j)
    // 边界部分块直接丢弃（加载时填的 0 保证越界位置恰好没有被写回的需要）
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int global_r = by * BM + ty * TM + i;
        if (global_r < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int global_c = bx * BN + tx * TN + j;
                if (global_c < N) {
                    C[global_r * N + global_c] = accumulate[i][j];
                }
            }
        }
    }
}

// 纯 2D 的 kernel 入口：grid 只有 (x=列 tile, y=行 tile) 两维
__global__ void mm_kernel_optimized(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K)
{
    gemm_tile_core(A, B, C, M, N, K);
}

// Batched 入口：gridDim.z = batch 序号（grid 折叠第三维）。
// k 这里只做“地址簿”工作，真正的计算全部委托给 gemm_tile_core：
//   1. 把 z 按广播后的 batch 形状从右往左取模分解出各 batch 维坐标，
//      乘上带广播（stride=0）的 strides 得到 A/B 的基地址偏移；
//      —— 与 Tensor::matmul 里 CPU 路径的 offset 反推是同一段数学，只是搬进了 kernel
//   2. C 无广播且恒连续，偏移直接 = z * M * N
__global__ void matmul_batched_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, int batch_base, BatchInfo info)
{
    const int zb = batch_base + static_cast<int>(blockIdx.z); // 全局 batch 序号
    int t = zb;
    long long off_a = 0, off_b = 0; // 大张量下 offset 可能超 int，用 64 位
    for (int d = info.ndim - 1; d >= 0; --d) {
        const int coord = t % info.shape[d];
        t /= info.shape[d];
        off_a += static_cast<long long>(coord) * info.strides_a[d];
        off_b += static_cast<long long>(coord) * info.strides_b[d];
    }
    gemm_tile_core(A + off_a, B + off_b, C + static_cast<long long>(zb) * M * N, M, N, K);
}

// ============================================================================
// Host 端启动封装。A/B/C 均为【设备指针】、行主序、连续内存的 FP32。
// （带 strides 的视图请先 contiguous()；当前 Tensor 均为连续，直接可用）
// ============================================================================
void launch_mm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    dim3 block(BN / TN, BM / TM);                 // (16, 16) = 256 线程
    // grid 按输出 C 的 tile 数组织：x=列方向(N)，y=行方向(M)，向上取整覆盖尾块
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    mm_kernel_optimized<<<grid, block>>>(A, B, C, M, N, K);

    // kernel 启动是异步的：这里只能捕获"启动本身"的错误（如配置非法、显存不足），
    // 真正的计算错误要等后续 cudaMemcpy/cudaDeviceSynchronize 才会上报
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_mm_kernel: ") + cudaGetErrorString(err));
}

// Batched 启动封装。gridDim.z 的硬件上限是 65535（远小于 x/y 的上限），
// batch_count 超出时分块多次启动，用 batch_base 把全局 batch 序号传给 kernel
void launch_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K,
                          size_t batch_count, const BatchInfo& info) {
    if (M <= 0 || N <= 0 || K <= 0 || batch_count == 0) return;

    dim3 block(BN / TN, BM / TM);
    const unsigned gx = (N + BN - 1) / BN, gy = (M + BM - 1) / BM;
    constexpr size_t Z_MAX = 65535;
    for (size_t base = 0; base < batch_count; base += Z_MAX) {
        const unsigned z = static_cast<unsigned>(std::min(Z_MAX, batch_count - base));
        matmul_batched_kernel<<<dim3(gx, gy, z), block>>>(A, B, C, M, N, K,
                                                          static_cast<int>(base), info);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_matmul_kernel: ") + cudaGetErrorString(err));
}

// ============================================================================
// 进阶笔记（不影响正确性，作为下一步优化的路标）：
// 1. Bank conflict: 计算阶段读 s_a[ty*8+i][k] 时，同一 warp 内相邻 ty 的地址
//    相差 64 个 float，64 % 32 = 0，落同一 bank => 约 2 路冲突。
//    经典解法是把 s_a 声明成 [BM][BK+?] 做 padding 错开 bank（改动搬运循环）。
//    s_b[k][tx*8+j] 同行内地址连续，无冲突。
// 2. Double buffering: 当前"搬运-同步-计算-同步"中，搬运期间计算单元空闲。
//    进阶做法是用两份 SMEM 缓冲 + cp.async 让下一块的加载与本块计算重叠。
// 3. Tile 尺寸的性能玄学：BM/BN/BK/TM/TN 决定寄存器用量与 occupancy，
//    需要用 ncu 实测调整；cuBLAS 仍有显著优势（它用了 Tensor Core）。
// ============================================================================

} // namespace mlspirit
