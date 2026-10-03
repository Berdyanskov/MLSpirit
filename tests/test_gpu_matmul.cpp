// GPU 端到端测试：2D tiled GEMM + batched matmul。
// 数值判定姿势：用 double 参考值做"真值"，要求 CUDA 结果与同等精度的 CPU FP32
// 基线误差同量级（而不是直接比容差——点积在 0 附近相消会让相对误差虚高）。
// 无 GPU 的机器上跳过并返回成功（编译照常，用于 CI 语法验证）。
#include "mlspirit.hpp"
#include <cuda_runtime.h>

using namespace mlspirit;
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>

static size_t numel_of(const std::vector<int>& s) { size_t n = 1; for (int d : s) n *= d; return n; }

// ---------------- 2D mm 用例 ----------------
static std::vector<double> mm_ref64(const std::vector<float>& A, const std::vector<float>& B,
                                    int M, int N, int K) {
    std::vector<double> C(M * N, 0.0);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) s += (double)A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
    return C;
}

static std::vector<float> mm_cpu32(const std::vector<float>& A, const std::vector<float>& B,
                                   int M, int N, int K) {
    std::vector<float> C(M * N, 0.f);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.f;
            for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
    return C;
}

static float max_err(const float* x, const std::vector<double>& ref) {
    float e = 0.f;
    for (size_t t = 0; t < ref.size(); ++t)
        e = std::max(e, (float)(std::fabs(x[t] - ref[t]) / (1.0 + std::fabs(ref[t]))));
    return e;
}

static int run_mm_case(int M, int N, int K) {
    srand(42 + M + N + K);
    std::vector<float> av(M * K), bv(K * N);
    for (auto& x : av) x = (rand() % 200 - 100) / 50.f;
    for (auto& x : bv) x = (rand() % 200 - 100) / 50.f;

    Tensor a({M, K}, DataType::FP32, DeviceType::CPU);
    Tensor b({K, N}, DataType::FP32, DeviceType::CPU);
    a.copy_from(av.data()); b.copy_from(bv.data());
    a.to_device(DeviceType::CUDA); b.to_device(DeviceType::CUDA);

    auto c = Tensor::mm(a, b);
    std::vector<float> cv(M * N);
    c->copy_to(cv.data());

    auto ref = mm_ref64(av, bv, M, N, K);
    auto cpu = mm_cpu32(av, bv, M, N, K);
    float err_gpu = max_err(cv.data(), ref);
    float err_cpu = max_err(cpu.data(), ref);
    bool ok = err_gpu <= std::max(5.f * err_cpu, 1e-6f);
    printf("mm M=%4d N=%4d K=%3d  err_gpu=%.3e err_cpu=%.3e  %s\n",
           M, N, K, err_gpu, err_cpu, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ---------------- batched matmul 用例：与已验证的 CPU 路径对比 ----------------
static void fill(std::vector<float>& v, int seed) {
    srand(seed);
    for (auto& x : v) x = (rand() % 200 - 100) / 50.f;
}

static float max_diff(const std::vector<float>& x, const std::vector<float>& y) {
    float e = 0.f;
    for (size_t t = 0; t < x.size(); ++t)
        e = std::max(e, std::fabs(x[t] - y[t]) / (1.f + std::fabs(y[t])));
    return e;
}

static int run_batched(std::vector<int> sa, std::vector<int> sb, std::vector<int> expect_shape, int seed) {
    std::vector<float> av(numel_of(sa)), bv(numel_of(sb));
    fill(av, seed); fill(bv, seed * 7 + 1);

    Tensor ac(sa, DataType::FP32, DeviceType::CPU); ac.copy_from(av.data());
    Tensor bc(sb, DataType::FP32, DeviceType::CPU); bc.copy_from(bv.data());
    auto cref = Tensor::matmul(ac, bc);

    Tensor ag(sa, DataType::FP32, DeviceType::CUDA); ag.copy_from(av.data());
    Tensor bg(sb, DataType::FP32, DeviceType::CUDA); bg.copy_from(bv.data());
    auto cg = Tensor::matmul(ag, bg);

    bool shape_ok = cg->shape() == expect_shape;
    std::vector<float> gv(numel_of(expect_shape)), rv(numel_of(expect_shape));
    cg->copy_to(gv.data()); cref->copy_to(rv.data());
    float e = max_diff(gv, rv);
    bool ok = shape_ok && e < 1e-4f;
    printf("matmul %zuD x %zuD  shape %s, err=%.3e  %s\n",
           sa.size(), sb.size(), shape_ok ? "OK" : "WRONG", e, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main() {
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) {
        printf("no CUDA device found, SKIPPED\n");
        return 0;
    }

    int fails = 0;
    fails += run_mm_case(128, 128, 8);
    fails += run_mm_case(256, 256, 64);
    fails += run_mm_case(100, 77, 33);
    fails += run_mm_case(1000, 777, 333);
    fails += run_mm_case(1, 1, 1);
    fails += run_mm_case(2048, 2048, 512);

    fails += run_batched({2,3,4}, {2,4,5}, {2,3,5}, 1);   // 等 batch
    fails += run_batched({2,3,4}, {4,5},   {2,3,5}, 2);   // b 广播到 batch
    fails += run_batched({3,4},   {2,4,5}, {2,3,5}, 3);   // a 广播到 batch
    fails += run_batched({2,1,3,4},{2,4,5},{2,2,3,5}, 4); // 双侧多维广播
    fails += run_batched({4,3,5}, {4,5,6}, {4,3,6}, 5);   // 常规 3D
    fails += run_batched({16,137,96},{96,205},{16,137,205}, 6); // 非整块 + 大批次
    fails += run_batched({5},     {5},     {},         7);  // 1D@1D -> 标量
    fails += run_batched({5},     {5,7},   {7},        8);  // 1D@2D
    fails += run_batched({6,5},   {5},     {6},        9);  // 2D@1D

    // 手工值复核：batch0 = I, batch1 = 2I
    Tensor A({2,2,2}, DataType::FP32, DeviceType::CUDA);
    std::vector<float> Av = {1,0, 0,1, 2,0, 0,2};
    Tensor B({2,3}, DataType::FP32, DeviceType::CUDA);
    std::vector<float> Bv = {1,2,3, 4,5,6};
    A.copy_from(Av.data()); B.copy_from(Bv.data());
    auto C = Tensor::matmul(A, B);
    std::vector<float> Cv(12);
    C->copy_to(Cv.data());
    float expv[] = {1,2,3, 4,5,6, 2,4,6, 8,10,12};
    bool ok = true;
    for (int t = 0; t < 12; ++t) ok &= (Cv[t] == expv[t]);
    printf("hand-check identity batches  %s\n", ok ? "PASS" : "FAIL");
    fails += ok ? 0 : 1;

    // ---- 转置视图进入 CUDA 路径（入口处逐元素 D2D 物化）----
    // 2D: [4,3](=A^T) @ [3,5]
    {
        Tensor ta({3, 4}, DataType::FP32, DeviceType::CPU);
        Tensor tb({3, 5}, DataType::FP32, DeviceType::CPU);
        std::vector<float> av(12), bv(15);
        fill(av, 21); fill(bv, 22);
        ta.copy_from(av.data()); tb.copy_from(bv.data());
        auto cref = Tensor::matmul(ta.transpose(), tb);   // CPU strides 路径(已验证)

        ta.to_device(DeviceType::CUDA); tb.to_device(DeviceType::CUDA);
        auto cg = Tensor::matmul(ta.transpose(), tb);     // 视图 -> 物化 -> kernel
        std::vector<float> gv(20), rv(20);
        cg->copy_to(gv.data()); cref->copy_to(rv.data());
        float e = max_diff(gv, rv);
        bool vok = e < 1e-4f;
        printf("2D transposed-view matmul  err=%.3e  %s\n", e, vok ? "PASS" : "FAIL");
        fails += vok ? 0 : 1;
    }
    // batched: [2,4,3](=A^T) @ [2,3,5]
    {
        Tensor ta({2, 3, 4}, DataType::FP32, DeviceType::CPU);
        Tensor tb({2, 3, 5}, DataType::FP32, DeviceType::CPU);
        std::vector<float> av(24), bv(30);
        fill(av, 31); fill(bv, 32);
        ta.copy_from(av.data()); tb.copy_from(bv.data());
        auto cref = Tensor::matmul(ta.transpose(), tb);

        ta.to_device(DeviceType::CUDA); tb.to_device(DeviceType::CUDA);
        auto cg = Tensor::matmul(ta.transpose(), tb);
        std::vector<float> gv(40), rv(40);
        cg->copy_to(gv.data()); cref->copy_to(rv.data());
        float e = max_diff(gv, rv);
        bool vok = e < 1e-4f;
        printf("batched transposed-view matmul  err=%.3e  %s\n", e, vok ? "PASS" : "FAIL");
        fails += vok ? 0 : 1;
    }

    printf(fails ? "RESULT: FAIL (%d errors)\n" : "RESULT: ALL PASS\n", fails);
    return fails ? 1 : 0;
}
