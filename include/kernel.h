#pragma once
#include "Tensor.hpp"

// a, b, c 均为设备指针；in-place 时 c 可与 a 相同。dtype 为元素类型。
void launch_add_kernel(void* a, void* b, void* c, size_t n, DataType dtype);