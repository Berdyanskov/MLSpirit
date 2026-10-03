#include "mlspirit.hpp"
#include <vector>

using namespace mlspirit;
#include <cassert>
#include <cstdio>

int main() {
    // shape/stride 推导检查
    Tensor t({2, 3, 4}, DataType::FP32, DeviceType::CPU);
    assert((t.strides() == std::vector<int>{12, 4, 1}));
    assert(t.numel() == 24);

    // 2D matmul: A[2,3] x B[3,2]
    Tensor a({2, 3}, DataType::FP32, DeviceType::CPU);
    Tensor b({3, 2}, DataType::FP32, DeviceType::CPU);
    std::vector<float> av = {1,2,3, 4,5,6};           // 行主序
    std::vector<float> bv = {7,8, 9,10, 11,12};
    a.copy_from(av.data());
    b.copy_from(bv.data());

    auto c = Tensor::matmul(a, b);                    // 期望 [[58,64],[139,154]]
    std::vector<float> cv(4);
    c->copy_to(cv.data());
    assert(cv[0]==58 && cv[1]==64 && cv[2]==139 && cv[3]==154);

    // 标量（0 维）不崩
    Tensor s({}, DataType::FP32, DeviceType::CPU);
    assert(s.numel() == 1 && s.strides().empty());

    std::puts("all tests passed");
    return 0;
}
