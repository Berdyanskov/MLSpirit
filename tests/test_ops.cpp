// 任务二：逐元素算子核 测试（assert 驱动）。
// 已实现算子的用例逐项落实；未实现算子（sub/mul/div/exp）的用例按 TODO 挂起。
#include "mlspirit.hpp"
#include <vector>
#include <cmath>
#include <cassert>
#include <cstdio>

using namespace mlspirit;

static bool close(float a, float b) { return std::fabs(a - b) < 1e-5f; }

static std::vector<float> read(const Tensor& t) {
    std::vector<float> v(t.numel());
    t.copy_to(v.data());
    return v;
}

int main() {
    // ---- 1) 同形状 add vs 手算 ----
    {
        Tensor a({2, 3}, DataType::FP32, DeviceType::CPU);
        Tensor b({2, 3}, DataType::FP32, DeviceType::CPU);
        std::vector<float> av = {1,2,3, 4,5,6}, bv = {10,20,30, 40,50,60};
        a.copy_from(av.data()); b.copy_from(bv.data());
        auto r = add(a, b);
        assert((r.shape() == std::vector<int>{2, 3}));
        auto rv = read(r);
        for (int i = 0; i < 6; ++i)
            assert(close(rv[i], av[i] + bv[i]));
    }

    // ---- 2) 广播：行向量 [2,3] + [3]，列广播 [2,3] + [2,1] ----
    {
        Tensor a({2, 3}, DataType::FP32, DeviceType::CPU);
        Tensor r3({3}, DataType::FP32, DeviceType::CPU);
        Tensor c2({2, 1}, DataType::FP32, DeviceType::CPU);
        std::vector<float> av = {1,2,3, 4,5,6}, rv3 = {100, 200, 300}, cv2 = {1000, 2000};
        a.copy_from(av.data()); r3.copy_from(rv3.data()); c2.copy_from(cv2.data());

        auto r = add(a, r3);            // 期望 [101,202,303, 104,205,306]
        auto r2 = add(a, c2);           // 期望 [1001,1002,1003, 2004,2005,2006]
        assert((r.shape() == std::vector<int>{2, 3}) && (r2.shape() == std::vector<int>{2, 3}));
        auto v1 = read(r), v2 = read(r2);
        float e1[] = {101,202,303, 104,205,306};
        float e2[] = {1001,1002,1003, 2004,2005,2006};
        for (int i = 0; i < 6; ++i) {
            assert(close(v1[i], e1[i]));
            assert(close(v2[i], e2[i]));
        }
        // 广播可交换：add(r3, a) 与 add(a, r3) 同值
        assert(read(add(r3, a)) == v1);
    }

    // ---- 3) 0 维标量参与广播（strides 为空的边界情形）----
    {
        Tensor a({2, 2}, DataType::FP32, DeviceType::CPU);
        Tensor s({}, DataType::FP32, DeviceType::CPU);
        std::vector<float> av = {1, 2, 3, 4};
        const float five = 5.f;
        a.copy_from(av.data()); s.copy_from(&five);
        auto r = add(a, s);
        assert((r.shape() == std::vector<int>{2, 2}));
        auto rv = read(r);
        for (int i = 0; i < 4; ++i)
            assert(close(rv[i], av[i] + 5.f));
    }

    // ---- 6) 转置视图直接参与 add（无需先 contiguous）----
    {
        Tensor a({2, 3}, DataType::FP32, DeviceType::CPU);
        Tensor b({3, 2}, DataType::FP32, DeviceType::CPU);
        std::vector<float> av = {1,2,3, 4,5,6}, bv = {1,1, 2,2, 3,3};
        a.copy_from(av.data()); b.copy_from(bv.data());
        // a.T 是 [3,2] 视图：{1,4, 2,5, 3,6}；加 b -> {2,5, 4,7, 6,9}
        auto r = add(a.transpose(), b);
        assert((r.shape() == std::vector<int>{3, 2}));
        auto rv = read(r);
        float e[] = {2,5, 4,7, 6,9};
        for (int i = 0; i < 6; ++i) assert(close(rv[i], e[i]));
    }

    // ---- 5) sum：已知值 + 0 维输出 + 转置不变性 ----
    {
        Tensor a({2, 3}, DataType::FP32, DeviceType::CPU);
        std::vector<float> av = {1,2,3, 4,5,6};
        a.copy_from(av.data());
        auto r = sum(a);
        assert(r.shape().empty() && r.numel() == 1);
        float sv = -1.f;
        r.copy_to(&sv);
        assert(close(sv, 21.f));

        // 转置只是视图变换，总和不变
        float sv2 = -1.f;
        sum(a.transpose()).copy_to(&sv2);
        assert(close(sv2, 21.f));
    }

    // ---- sub/mul/div：同形状 + 广播 ----
    {
        Tensor a({2, 3}, DataType::FP32, DeviceType::CPU);
        Tensor b({2, 3}, DataType::FP32, DeviceType::CPU);
        std::vector<float> av = {2,4,6, 8,10,12}, bv = {1,2,3, 4,5,6};
        a.copy_from(av.data()); b.copy_from(bv.data());
        auto rs = read(sub(a, b)), rm = read(mul(a, b)), rd = read(div(a, b));
        for (int i = 0; i < 6; ++i) {
            assert(close(rs[i], av[i] - bv[i]));
            assert(close(rm[i], av[i] * bv[i]));
            assert(close(rd[i], av[i] / bv[i]));
        }
        // 广播用例（b 退化为行向量）
        Tensor r3({3}, DataType::FP32, DeviceType::CPU);
        std::vector<float> rv3 = {1, 2, 3};
        r3.copy_from(rv3.data());
        auto rb = read(mul(a, r3));
        for (int i = 0; i < 6; ++i)
            assert(close(rb[i], av[i] * rv3[i % 3]));
    }

    // ---- exp：已知值 + 转置视图输入 ----
    {
        Tensor a({2, 2}, DataType::FP32, DeviceType::CPU);
        std::vector<float> av = {0.f, 1.f, -1.f, 2.f};
        a.copy_from(av.data());
        auto rv = read(exp(a));
        for (int i = 0; i < 4; ++i)
            assert(close(rv[i], std::exp(av[i])));
        auto rtv = read(exp(a.transpose()));
        std::vector<float> order = {0, 2, 1, 3}; // 转置后的读取顺序
        for (int i = 0; i < 4; ++i)
            assert(close(rtv[i], std::exp(av[order[i]])));
    }

    // ---- sum_to：广播的逆运算 ----
    {
        Tensor a({2, 3}, DataType::FP32, DeviceType::CPU);
        std::vector<float> av = {1,2,3, 4,5,6};
        a.copy_from(av.data());

        auto r = sum_to(a, {3});                 // 沿 dim0 坍缩: 列和
        assert((r.shape() == std::vector<int>{3}));
        auto rv = read(r);
        assert(close(rv[0], 5.f) && close(rv[1], 7.f) && close(rv[2], 9.f));

        auto r2 = sum_to(a, {2, 1});             // 沿 dim1 坍缩: 行和
        assert((r2.shape() == std::vector<int>{2, 1}));
        auto rv2 = read(r2);
        assert(close(rv2[0], 6.f) && close(rv2[1], 15.f));

        auto r3 = sum_to(a, {});                 // 全部坍缩到 0 维
        assert(r3.shape().empty());
        float sv = 0; r3.copy_to(&sv);
        assert(close(sv, 21.f));

        // batched 左侧维坍缩: [4,2,3] -> [2,3]，4 个 [2,3] 块逐元素相加
        Tensor b({4, 2, 3}, DataType::FP32, DeviceType::CPU);
        std::vector<float> bv(24);
        for (int i = 0; i < 24; ++i) bv[i] = static_cast<float>(i + 1);
        b.copy_from(bv.data());
        auto r4 = sum_to(b, {2, 3});
        auto rv4 = read(r4);
        float e4[] = {40, 44, 48, 52, 56, 60};   // (i+1)+(i+7)+(i+13)+(i+19)
        for (int i = 0; i < 6; ++i) assert(close(rv4[i], e4[i]));

        // 视图输入: a.T=[3,2] 沿行坍缩到 [2] => {1+2+3, 4+5+6}
        auto rv5 = read(sum_to(a.transpose(), {2}));
        assert(close(rv5[0], 6.f) && close(rv5[1], 15.f));
    }

    std::puts("test_ops: all cases passed");
    return 0;
}
