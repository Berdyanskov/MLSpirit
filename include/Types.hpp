#pragma once
#include <cstddef>
#include <cstdint>

namespace mlspirit {

enum class DeviceType { CPU, CUDA };
enum class DataType { FP32, FP16, INT8 };

// 各 dtype 在 CPU 上对应的元素字节数（FP16 用 2 字节存储）
inline size_t dtype_element_size(DataType dtype) {
    switch (dtype) {
        case DataType::FP32: return 4;
        case DataType::FP16: return 2;
        case DataType::INT8: return 1;
    }
    return 4;
}

} // namespace mlspirit
