// ============================================================================
// ops_elementwise_cpu.cpp — 逐元素算子：广播计划(已实现) + 算子本体(任务二 TODO)
//
// 结构：所有二元算子共享同一个 BroadcastPlan——把两个输入的 shape/strides
// 对齐到输出形状（广播维 stride=0），之后 CPU 串行循环与 CUDA kernel 用的是
// 同一份"双游标"下标换算。算子本体的差异只剩最后一行计算。
// ============================================================================
#include "ops.hpp"
#include "kernel.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace mlspirit {

namespace {

struct BroadcastPlan {
    std::vector<int> shape;      // 对齐后的输出形状
    std::vector<int> strides_a;  // A 对齐到输出的 strides
    std::vector<int> strides_b;  // B 对齐到输出的 strides
};

BroadcastPlan plan_binary(const Tensor& a, const Tensor& b, const char* op) {
    const std::vector<int>& sa = a.shape();
    const std::vector<int>& sta = a.strides();
    const std::vector<int>& sb = b.shape();
    const std::vector<int>& stb = b.strides();
    if (a.dtype() != b.dtype())
        throw std::runtime_error(std::string(op) + ": dtype mismatch");
    if (a.device() != b.device())
        throw std::runtime_error(std::string(op) + ": device mismatch");

    const int na = static_cast<int>(sa.size());
    const int nb = static_cast<int>(sb.size());
    const int n = std::max(na, nb);
    BroadcastPlan plan;
    plan.shape.assign(n, 1);
    plan.strides_a.assign(n, 0);
    plan.strides_b.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        const int ia = na - 1 - i, ib = nb - 1 - i, io = n - 1 - i;
        const int da = (ia >= 0) ? sa[ia] : 1;
        const int db = (ib >= 0) ? sb[ib] : 1;
        if (da == db) {
            plan.shape[io] = da;
            plan.strides_a[io] = (ia >= 0) ? sta[ia] : 0;
            plan.strides_b[io] = (ib >= 0) ? stb[ib] : 0;
        } else if (da == 1) {
            plan.shape[io] = db;
            plan.strides_a[io] = 0;                  // 广播维 stride 置 0
            plan.strides_b[io] = (ib >= 0) ? stb[ib] : 0;
        } else if (db == 1) {
            plan.shape[io] = da;
            plan.strides_a[io] = (ia >= 0) ? sta[ia] : 0;
            plan.strides_b[io] = 0;
        } else {
            throw std::runtime_error(std::string(op) + ": broadcast shape mismatch");
        }
    }
    return plan;
}

} // namespace

Tensor add(const Tensor& a, const Tensor& b) {
    const BroadcastPlan plan = plan_binary(a, b, "add");
    if (a.dtype() != DataType::FP32)
        throw std::runtime_error("add: FP32 only for now (CPU path casts to float)");
    Tensor out(plan.shape, a.dtype(), a.device());

    if (a.device() == DeviceType::CUDA) {

        StrideDesc da{}, db{};
        da.ndim = db.ndim = static_cast<int>(plan.shape.size());
        for (size_t d = 0; d < plan.shape.size(); ++d) {
            da.shape[d] = db.shape[d] = plan.shape[d];
            da.strides[d] = plan.strides_a[d];
            db.strides[d] = plan.strides_b[d];
        }
        launch_elementwise_binary_kernel(a.data(), b.data(), out.data(),
                                         out.numel(), da, db, a.dtype(),
                                         ElementwiseBinaryOp::Add);
        return out;
    }

    const float* pa = static_cast<const float*>(a.data());
    const float* pb = static_cast<const float*>(b.data());
    float* po = static_cast<float*>(out.data());
    const int ndim = static_cast<int>(plan.shape.size());
    for (size_t idx = 0; idx < out.numel(); ++idx) {
        size_t off_a = 0, off_b = 0, t = idx;
        for (int d = ndim - 1; d >= 0; --d) {
            const size_t coord = t % static_cast<size_t>(plan.shape[d]);
            off_a += coord * static_cast<size_t>(plan.strides_a[d]);
            off_b += coord * static_cast<size_t>(plan.strides_b[d]);
            t /= static_cast<size_t>(plan.shape[d]);
        }
        po[idx] = pa[off_a] + pb[off_b];
    }
    return out;
}

// sub/mul/div 与 add 共享的公共骨架：广播计划 + dtype 守卫 + 分配 + 双端分派，
// CPU 端由调用方传入最后一行运算的 lambda，CUDA 端走统一的 op 枚举模板分派
namespace {

template <typename F>
Tensor binary_op_impl(const Tensor& a, const Tensor& b, const char* op_name,
                      ElementwiseBinaryOp op, F cpu_f) {
    const BroadcastPlan plan = plan_binary(a, b, op_name);
    if (a.dtype() != DataType::FP32)
        throw std::runtime_error(std::string(op_name) +
                                 ": FP32 only for now (CPU path casts to float)");
    Tensor out(plan.shape, a.dtype(), a.device());

    if (a.device() == DeviceType::CUDA) {
        StrideDesc da{}, db{};
        da.ndim = db.ndim = static_cast<int>(plan.shape.size());
        for (size_t d = 0; d < plan.shape.size(); ++d) {
            da.shape[d] = db.shape[d] = plan.shape[d];
            da.strides[d] = plan.strides_a[d];
            db.strides[d] = plan.strides_b[d];
        }
        launch_elementwise_binary_kernel(a.data(), b.data(), out.data(),
                                         out.numel(), da, db, a.dtype(), op);
        return out;
    }

    const float* pa = static_cast<const float*>(a.data());
    const float* pb = static_cast<const float*>(b.data());
    float* po = static_cast<float*>(out.data());
    const int ndim = static_cast<int>(plan.shape.size());
    for (size_t idx = 0; idx < out.numel(); ++idx) {
        size_t off_a = 0, off_b = 0, t = idx;
        for (int d = ndim - 1; d >= 0; --d) {
            const size_t coord = t % static_cast<size_t>(plan.shape[d]);
            off_a += coord * static_cast<size_t>(plan.strides_a[d]);
            off_b += coord * static_cast<size_t>(plan.strides_b[d]);
            t /= static_cast<size_t>(plan.shape[d]);
        }
        po[idx] = cpu_f(pa[off_a], pb[off_b]);
    }
    return out;
}

} // namespace

Tensor sub(const Tensor& a, const Tensor& b) {
    return binary_op_impl(a, b, "sub", ElementwiseBinaryOp::Sub,
                          [](float x, float y) { return x - y; });
}

Tensor mul(const Tensor& a, const Tensor& b) {
    return binary_op_impl(a, b, "mul", ElementwiseBinaryOp::Mul,
                          [](float x, float y) { return x * y; });
}

Tensor div(const Tensor& a, const Tensor& b) {
    return binary_op_impl(a, b, "div", ElementwiseBinaryOp::Div,
                          [](float x, float y) { return x / y; });
}

Tensor exp(const Tensor& x) {
    Tensor out(x.shape(), x.dtype(), x.device());
    if (x.dtype() != DataType::FP32)
        throw std::runtime_error("exp: FP32 only for now (task 2 first version)");
    if (x.numel() == 0) return out;

    if (x.device() == DeviceType::CUDA) {
        if (x.shape().size() > 8)
            throw std::runtime_error("exp: CUDA path supports at most 8 dims");
        StrideDesc desc{};
        desc.ndim = static_cast<int>(x.shape().size());
        for (size_t d = 0; d < x.shape().size(); ++d) {
            desc.shape[d] = x.shape()[d];
            desc.strides[d] = x.strides()[d];
        }
        launch_exp_kernel(x.data(), out.data(), x.numel(), desc, x.dtype());
        return out;
    }

    // CPU：视图直读 + std::exp（与 kernel 同一段索引数学的串行版）
    const float* px = static_cast<const float*>(x.data());
    float* po = static_cast<float*>(out.data());
    const int ndim = static_cast<int>(x.shape().size());
    for (size_t idx = 0; idx < x.numel(); ++idx) {
        size_t off = 0, t = idx;
        for (int d = ndim - 1; d >= 0; --d) {
            const size_t coord = t % static_cast<size_t>(x.shape()[d]);
            off += coord * static_cast<size_t>(x.strides()[d]);
            t /= static_cast<size_t>(x.shape()[d]);
        }
        po[idx] = std::exp(px[off]);
    }
    return out;
}

Tensor sum(const Tensor& x) {
    // 输出是 0 维张量（shape 为空，numel=1）
    Tensor out({}, x.dtype(), x.device());
    if (x.dtype() != DataType::FP32)
        throw std::runtime_error("sum: FP32 only for now (task 2 first version)");

    if (x.device() == DeviceType::CUDA) {
        // kernel 约定输出端已清零（atomicAdd 是累加语义），用 copy_from 写 0
        const float zero = 0.0f;
        out.copy_from(&zero);
        if (x.numel() == 0) return out;
        if (x.shape().size() > 8)
            throw std::runtime_error("sum: CUDA path supports at most 8 dims");
        StrideDesc desc{};
        desc.ndim = static_cast<int>(x.shape().size());
        for (size_t d = 0; d < x.shape().size(); ++d) {
            desc.shape[d] = x.shape()[d];
            desc.strides[d] = x.strides()[d];
        }
        launch_sum_kernel(x.data(), out.data(), x.numel(), desc, x.dtype());
        return out;
    }

    // CPU：按 strides 坐标反推累加（视图直读，与 kernel 同一段数学的串行版）
    const float* px = static_cast<const float*>(x.data());
    float acc = 0.0f;
    const int ndim = static_cast<int>(x.shape().size());
    for (size_t idx = 0; idx < x.numel(); ++idx) {
        size_t off = 0, t = idx;
        for (int d = ndim - 1; d >= 0; --d) {
            const size_t coord = t % static_cast<size_t>(x.shape()[d]);
            off += coord * static_cast<size_t>(x.strides()[d]);
            t /= static_cast<size_t>(x.shape()[d]);
        }
        acc += px[off];
    }
    out.copy_from(&acc);
    return out;
}

} // namespace mlspirit
