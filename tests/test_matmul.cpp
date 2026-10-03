#include "mlspirit.hpp"
#include <vector>

using namespace mlspirit;
#include <cassert>
#include <cstdio>
#include <cmath>

static bool close(float a, float b) { return std::fabs(a - b) < 1e-5f; }

int main() {
    // 1) mm: 2D x 2D
    Tensor a({2, 3}, DataType::FP32, DeviceType::CPU);
    Tensor b({3, 2}, DataType::FP32, DeviceType::CPU);
    std::vector<float> av = {1,2,3, 4,5,6};
    std::vector<float> bv = {7,8, 9,10, 11,12};
    a.copy_from(av.data()); b.copy_from(bv.data());
    auto c = Tensor::mm(a, b);
    std::vector<float> cv(4);
    c->copy_to(cv.data());
    assert(cv[0]==58 && cv[1]==64 && cv[2]==139 && cv[3]==154);

    // 2) matmul: batched 3D x 2D（y 广播）。A[2,2,2]: batch0=I, batch1=2I
    Tensor A({2, 2, 2}, DataType::FP32, DeviceType::CPU);
    std::vector<float> Av = {1,0, 0,1,  2,0, 0,2};
    Tensor B({2, 3}, DataType::FP32, DeviceType::CPU);
    std::vector<float> Bv = {1,2,3, 4,5,6};
    A.copy_from(Av.data()); B.copy_from(Bv.data());
    auto C = Tensor::matmul(A, B);
    assert((C->shape() == std::vector<int>{2, 2, 3}));
    std::vector<float> Cv(12);
    C->copy_to(Cv.data());
    float expect[] = {1,2,3, 4,5,6, 2,4,6, 8,10,12};
    for (int t = 0; t < 12; ++t) assert(close(Cv[t], expect[t]));

    // 3) matmul: 1D x 1D -> 0维标量 = 内积
    Tensor v({3}, DataType::FP32, DeviceType::CPU);
    Tensor w({3}, DataType::FP32, DeviceType::CPU);
    std::vector<float> vv = {1, 2, 3}, wv = {4, 5, 6};
    v.copy_from(vv.data()); w.copy_from(wv.data());
    auto s = Tensor::matmul(v, w);
    assert(s->shape().empty());
    float sv; s->copy_to(&sv);
    assert(close(sv, 32.f));

    // 4) matmul: 1D x 2D -> 1D  [1,2,3] @ B
    auto r1 = Tensor::matmul(v, b); // b 是 3x2
    assert((r1->shape() == std::vector<int>{2}));
    std::vector<float> r1v(2);
    r1->copy_to(r1v.data());
    assert(close(r1v[0], 1*7+2*9+3*11) && close(r1v[1], 1*8+2*10+3*12));

    // 5) matmul: 2D x 1D -> 1D  a @ w
    auto r2 = Tensor::matmul(a, w); // a 2x3, w [3]
    assert((r2->shape() == std::vector<int>{2}));
    std::vector<float> r2v(2);
    r2->copy_to(r2v.data());
    assert(close(r2v[0], 1*4+2*5+3*6) && close(r2v[1], 4*4+5*5+6*6));

    // 6) 非方阵 2D x 2D（K != N，专治 stride/shape 混用 bug）
    Tensor x({1, 4}, DataType::FP32, DeviceType::CPU);
    Tensor y({4, 3}, DataType::FP32, DeviceType::CPU);
    std::vector<float> xv = {1, 1, 1, 1};
    std::vector<float> yv = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
    x.copy_from(xv.data()); y.copy_from(yv.data());
    auto z = Tensor::matmul(x, y);
    std::vector<float> zv(3);
    z->copy_to(zv.data());
    assert(close(zv[0], 22.f) && close(zv[1], 26.f) && close(zv[2], 30.f));

    std::puts("all matmul tests passed");
    return 0;
}
