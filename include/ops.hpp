#pragma once
// ============================================================================
// ops.hpp — 逐元素算子与归约算子的公开 API（任务二）
//
// 设计说明：
// - 全部是自由函数（aten 风格），不挂进 Tensor 类，保持 Tensor 只负责存储/视图；
// - 二元算子遵循 NumPy 广播语义：shape 从右对齐，为 1 的维 stride 置 0 复用数据；
// - 视图输入直接可用（迭代器按 strides 寻址），不需要调用方先 contiguous()。
//
// 实现位置：CPU 分派与广播对齐见 src/ops_elementwise_cpu.cpp，
//           CUDA kernel 见 src/elementwise.cu。
// ============================================================================
#include "Tensor.hpp"

namespace mlspirit {

// 逐元素二元算子（NumPy 广播语义）
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

// 逐元素一元算子
Tensor exp(const Tensor& x);

// 全归约：所有元素求和，输出 0 维标量张量
Tensor sum(const Tensor& x);

// 形状归约（广播的逆运算、广播算子反向的基石）：
// 沿"广播时被复制"的维度求和，把 x 的形状收回到 shape。
// 约定：len(shape) ≤ x.ndim；从右对齐后每维要么等于 x 对应维、要么为 1。
Tensor sum_to(const Tensor& x, const std::vector<int>& shape);

} // namespace mlspirit
