// ============================================================================
// mlspirit.hpp — 库的单入口伞形头文件
//
// 使用者只需要：
//     #include "mlspirit.hpp"
//     using namespace mlspirit;        // 或显式 mlspirit::Tensor / mls::Tensor
//
// 等价于 Python 侧的 `import mlspirit as mls`：一个 include 拿到全部公开 API。
// ============================================================================
#pragma once

#include "Tensor.hpp"   // Tensor / DeviceType / DataType
#include "kernel.h"     // CUDA launcher 接口 / BatchInfo / StrideDesc
#include "ops.hpp"      // 逐元素与归约算子（任务二）

// 短命名空间别名：`mls::Tensor t({2,3}, DataType::FP32, DeviceType::CPU);`
namespace mls = mlspirit;
