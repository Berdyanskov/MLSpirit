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

    // ---- 视图：transpose 零拷贝 ----
    Tensor mt = a.transpose();                        // 默认交换最后两维: [3,2]
    assert((mt.shape() == std::vector<int>{3, 2}));
    assert((mt.strides() == std::vector<int>{1, 3}));
    assert(a.is_contiguous() && !mt.is_contiguous());
    assert(mt.data() == a.data());                    // 共享存储：拷贝零成本

    // 视图读出：copy_to 沿 strides 物化，得到转置后的行主序内容
    std::vector<float> tv(6);
    mt.copy_to(tv.data());
    assert((tv == std::vector<float>{1,4, 2,5, 3,6}));

    // contiguous() 物化后与视图读出一致
    Tensor mc = mt.contiguous();
    assert(mc.is_contiguous());
    std::vector<float> mcv(6);
    mc.copy_to(mcv.data());
    assert(mcv == tv);

    // 连续张量的 transpose 撤销：再转回去即连续
    assert(mt.transpose().is_contiguous());

    // 3D 任意两维交换（默认参数为最后两维）
    Tensor t3 = t.transpose(0, 2);
    assert((t3.shape() == std::vector<int>{4, 3, 2}));
    assert((t3.strides() == std::vector<int>{1, 4, 12}));

    // 转置视图参与 mm（CPU 路径直接吃 strides）
    Tensor e({2, 2}, DataType::FP32, DeviceType::CPU);
    std::vector<float> ev = {1,0, 0,1};               // 单位阵
    e.copy_from(ev.data());
    auto ce = Tensor::mm(mt, e);                      // [3,2] @ [2,2] = mt 本身
    std::vector<float> cev(6);
    ce->copy_to(cev.data());
    assert(cev == tv);

    std::puts("all tests passed");
    return 0;
}
