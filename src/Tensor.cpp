#include "Tensor.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include "kernel.h"
#include <cuda_runtime.h>

namespace mlspirit {

void Tensor::allocate_memory() {
    if (device_ == DeviceType::CUDA) {
        // 设备端直接构造的 Tensor（如 mm 的输出）必须分配显存；
        // 一度只有 new[]，host 指针被 CUDA kernel 当设备指针写入 => 非法内存访问
        cudaError_t err = cudaMalloc(&data_ptr_, size_bytes());
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("Tensor: cudaMalloc failed: ") + cudaGetErrorString(err));
        return;
    }
    switch (dtype_) {
    case DataType::FP32:
        data_ptr_ = new float[numel_];
        break;
    case DataType::FP16:
        data_ptr_ = new uint16_t[numel_];
        break;
    case DataType::INT8:
        data_ptr_ = new int8_t[numel_];
        break;
    default:
        data_ptr_ = new float[numel_];
        break;
    }
}

void Tensor::free_memory() {
    if (data_ptr_ == nullptr) return;
    if (device_ == DeviceType::CUDA) {
        // 设备内存必须走 cudaFree；对它 delete[] 会直接段错误
        cudaFree(data_ptr_);
    } else {
        switch (dtype_) {
        case DataType::FP32:
            delete[] static_cast<float*>(data_ptr_);
            break;
        case DataType::FP16:
            delete[] static_cast<uint16_t*>(data_ptr_);
            break;
        case DataType::INT8:
            delete[] static_cast<int8_t*>(data_ptr_);
            break;
        default:
            delete[] static_cast<float*>(data_ptr_);
            break;
        }
    }
    data_ptr_ = nullptr;
}

Tensor::Tensor(const std::vector<int>& shape, DataType dtype, DeviceType device)
    : shape_(shape), device_(device), dtype_(dtype) {
    numel_ = 1;
    for (int d : shape_) numel_ *= d;
    strides_ = std::vector<int>(shape_.size());
    if (strides_.size()){
        strides_.back() = 1;
        for (int i = static_cast<int>(strides_.size()) - 2; i >= 0; --i){
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }
    data_ptr_ = nullptr;
    allocate_memory();
}

Tensor::~Tensor() {
    free_memory();
}

Tensor::Tensor(Tensor&& other) noexcept: shape_(std::move(other.shape_)), strides_(std::move(other.strides_)), numel_(other.numel_), device_(other.device_),
    dtype_(other.dtype_), data_ptr_(other.data_ptr_) {
    other.data_ptr_ = nullptr;
    other.numel_ = 0;
    other.strides_.clear();
    other.shape_.clear();
}

void Tensor::to_device(DeviceType target_device) {
    if (device_ == target_device) return;
    if (target_device == DeviceType::CUDA) {
        void* host_ptr = data_ptr_;
        data_ptr_ = nullptr;
        cudaMalloc(&data_ptr_, size_bytes());
        cudaMemcpy(data_ptr_, host_ptr, size_bytes(), cudaMemcpyHostToDevice);
        // delete[] 必须按元素类型释放，void* 无法正确析构，需按 dtype_ 转换后释放
        switch (dtype_) {
        case DataType::FP32: delete[] static_cast<float*>(host_ptr); break;
        case DataType::FP16: delete[] static_cast<uint16_t*>(host_ptr); break;
        case DataType::INT8: delete[] static_cast<int8_t*>(host_ptr); break;
        default: delete[] static_cast<float*>(host_ptr); break;
        }
    } else {
        void* dev_ptr = data_ptr_;
        data_ptr_ = nullptr;
        // 与 allocate_memory() 一致，用 new[] 分配，以便 free_memory() 正确 delete[]
        switch (dtype_) {
        case DataType::FP32: data_ptr_ = new float[numel_]; break;
        case DataType::FP16: data_ptr_ = new uint16_t[numel_]; break;
        case DataType::INT8: data_ptr_ = new int8_t[numel_]; break;
        default: data_ptr_ = new float[numel_]; break;
        }
        cudaMemcpy(data_ptr_, dev_ptr, size_bytes(), cudaMemcpyDeviceToHost);
        cudaFree(dev_ptr);
    }
    device_ = target_device;
}

void Tensor::copy_from(const void* host_ptr) {
    if (host_ptr == nullptr) return;
    if (device_ == DeviceType::CUDA)
        cudaMemcpy(data_ptr_, host_ptr, size_bytes(), cudaMemcpyHostToDevice);
    else
        std::memcpy(data_ptr_, host_ptr, size_bytes());
}

void Tensor::copy_to(void* host_ptr) const {
    if (host_ptr == nullptr) return;
    if (device_ == DeviceType::CUDA)
        cudaMemcpy(host_ptr, data_ptr_, size_bytes(), cudaMemcpyDeviceToHost);
    else
        std::memcpy(host_ptr, data_ptr_, size_bytes());
}

void Tensor::reshape_in_place(const std::vector<int>& new_shape) {
    size_t new_numel = 1;
    for (int d : new_shape) new_numel *= d;
    if (new_numel != numel_)
        throw std::runtime_error("reshape_in_place: element count mismatch");
    shape_ = new_shape;
    // 与构造函数相同的行主序 strides 推导（连续张量前提）
    strides_.assign(new_shape.size(), 0);
    if (!strides_.empty()) {
        strides_.back() = 1;
        for (int i = static_cast<int>(strides_.size()) - 2; i >= 0; --i)
            strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

std::unique_ptr<Tensor> Tensor::mm(const Tensor& a, const Tensor& b) {
    const std::vector<int>& sa = a.shape();
    const std::vector<int>& sb = b.shape();
    if (sa.size() != 2 || sb.size() != 2)
        throw std::runtime_error("mm: only 2D tensors supported");
    int M = sa[0], K = sa[1], N = sb[1];
    if (sb[0] != K)
        throw std::runtime_error("mm: shape mismatch (K)");
    if (a.dtype() != DataType::FP32 || b.dtype() != DataType::FP32)
        throw std::runtime_error("mm: only FP32 supported in this stub");
    if (a.device() == DeviceType::CUDA) {
        if (b.device() != DeviceType::CUDA)
            throw std::runtime_error("mm: device mismatch (a on CUDA, b on CPU)");
        // CUDA kernel 假定行主序连续内存（当前所有 Tensor 均为连续；
        // 将来支持非连续视图时需在入口处先 contiguous()）
        auto out = std::make_unique<Tensor>(std::vector<int>{M, N}, DataType::FP32, DeviceType::CUDA);
        launch_mm_kernel(static_cast<const float*>(a.data()),
                         static_cast<const float*>(b.data()),
                         static_cast<float*>(out->data()), M, N, K);
        return out;
    }
    auto out = std::make_unique<Tensor>(std::vector<int>{M, N}, DataType::FP32, a.device());
    const float* pa = static_cast<const float*>(a.data());
    const float* pb = static_cast<const float*>(b.data());
    float* pc = static_cast<float*>(out->data());
    const std::vector<int>& sta = a.strides();
    const std::vector<int>& stb = b.strides();
    const std::vector<int>& stc = out->strides();

    // 下标换算统一走 strides：a[i,k] = pa[i*sta[0] + k*sta[1]]
    // 对连续张量退化为原来的 i*K + k，将来支持转置/切片视图时也无需改动
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.f;
            for (int k = 0; k < K; ++k)
                sum += pa[i * sta[0] + k * sta[1]] * pb[k * stb[0] + j * stb[1]];
            pc[i * stc[0] + j * stc[1]] = sum;
        }
    return out;
}

static std::vector<int> compute_broadcast_batch_shape(
    const std::vector<int>& sa, const std::vector<int>& sta,
    const std::vector<int>& sb, const std::vector<int>& stb,
    std::vector<int>& out_batch_strides_a,
    std::vector<int>& out_batch_strides_b) 
{
    int ndim_a = sa.size();
    int ndim_b = sb.size();
    int max_batch_ndim = std::max(ndim_a, ndim_b);

    std::vector<int> batch_shape(max_batch_ndim, 1);
    out_batch_strides_a.assign(max_batch_ndim, 0);
    out_batch_strides_b.assign(max_batch_ndim, 0);

    // 从右往左逐维比对 Batch 维度
    for (int i = 0; i < max_batch_ndim; ++i) {
        int idx_a = ndim_a - 1 - i;
        int idx_b = ndim_b - 1 - i;
        int idx_out = max_batch_ndim - 1 - i;

        int dim_a = (idx_a >= 0) ? sa[idx_a] : 1;
        int dim_b = (idx_b >= 0) ? sb[idx_b] : 1;
        int str_a = (idx_a >= 0) ? sta[idx_a] : 0;
        int str_b = (idx_b >= 0) ? stb[idx_b] : 0;

        if (dim_a == dim_b) {
            batch_shape[idx_out] = dim_a;
            out_batch_strides_a[idx_out] = str_a;
            out_batch_strides_b[idx_out] = str_b;
        } else if (dim_a == 1) {
            batch_shape[idx_out] = dim_b;
            out_batch_strides_a[idx_out] = 0; // 广播维度 stride 设为 0
            out_batch_strides_b[idx_out] = str_b;
        } else if (dim_b == 1) {
            batch_shape[idx_out] = dim_a;
            out_batch_strides_a[idx_out] = str_a;
            out_batch_strides_b[idx_out] = 0; // 广播维度 stride 设为 0
        } else {
            throw std::runtime_error("matmul: batch dimensions broadcast mismatch");
        }
    }
    return batch_shape;
}

std::unique_ptr<Tensor> Tensor::matmul(const Tensor& a, const Tensor& b) {
    if (a.dtype() != DataType::FP32 || b.dtype() != DataType::FP32)
        throw std::runtime_error("matmul: only FP32 supported in this stub");
    if (a.device() != b.device())
        throw std::runtime_error("matmul: device mismatch");

    const std::vector<int>& sa = a.shape();
    const std::vector<int>& sb = b.shape();
    const std::vector<int>& sta = a.strides();
    const std::vector<int>& stb = b.strides();

    bool squeeze_a = false;
    bool squeeze_b = false;

    // 1. 处理 1D 向量的自动提升 (1D -> 2D)
    std::vector<int> eff_sa = sa;
    std::vector<int> eff_sta = sta;
    if (eff_sa.size() == 1) {
        eff_sa = {1, sa[0]};
        eff_sta = {0, sta[0]}; // 前面补 1 维，stride 设为 0
        squeeze_a = true;
    }

    std::vector<int> eff_sb = sb;
    std::vector<int> eff_stb = stb;
    if (eff_sb.size() == 1) {
        eff_sb = {sb[0], 1};
        eff_stb = {stb[0], 0}; // 后面补 1 维，stride 设为 0
        squeeze_b = true;
    }

    int ndim_a = eff_sa.size();
    int ndim_b = eff_sb.size();

    // 2. 提取矩阵维 M, K, N 并校验 K 维一致性
    int M = eff_sa[ndim_a - 2];
    int K_a = eff_sa[ndim_a - 1];
    int K_b = eff_sb[ndim_b - 2];
    int N = eff_sb[ndim_b - 1];

    if (K_a != K_b) {
        throw std::runtime_error("matmul: inner dimension mismatch (K)");
    }
    int K = K_a;

    // 3. 提取 Batch 维度并计算广播
    std::vector<int> batch_sa(eff_sa.begin(), eff_sa.end() - 2);
    std::vector<int> batch_sta(eff_sta.begin(), eff_sta.end() - 2);
    std::vector<int> batch_sb(eff_sb.begin(), eff_sb.end() - 2);
    std::vector<int> batch_stb(eff_stb.begin(), eff_stb.end() - 2);

    std::vector<int> b_strides_a, b_strides_b;
    std::vector<int> out_batch_shape = compute_broadcast_batch_shape(
        batch_sa, batch_sta, batch_sb, batch_stb, b_strides_a, b_strides_b
    );

    // 计算总 Batch 元素数量
    size_t batch_count = 1;
    for (int dim : out_batch_shape) {
        batch_count *= dim;
    }

    // 4. 构建输出张量的 Shape
    std::vector<int> out_shape = out_batch_shape;
    out_shape.push_back(M);
    out_shape.push_back(N);

    auto out = std::make_unique<Tensor>(out_shape, DataType::FP32, a.device());
    const float* pa = static_cast<const float*>(a.data());
    const float* pb = static_cast<const float*>(b.data());
    float* pc = static_cast<float*>(out->data());

    const std::vector<int>& stc = out->strides();
    int num_batch_dims = out_batch_shape.size();

    // 5. 计算：CUDA 走 batched kernel（batch 折叠进 gridDim.z），CPU 走朴素循环
    int stride_a_M = eff_sta[ndim_a - 2];
    int stride_a_K = eff_sta[ndim_a - 1];
    int stride_b_K = eff_stb[ndim_b - 2];
    int stride_b_N = eff_stb[ndim_b - 1];

    int stride_c_M = stc[stc.size() - 2];
    int stride_c_N = stc[stc.size() - 1];

    if (a.device() == DeviceType::CUDA) {
        // kernel 内部矩阵维按 r*K+c / r*N+c 直接寻址，要求行主序连续。
        // 当前所有张量都连续；将来支持非连续视图时需在入口处先 contiguous()。
        // （M/N/K 为 1 的维对应的 stride 不会被 kernel 读到，豁免检查）
        if ((M > 1 && stride_a_M != K) || stride_a_K != 1 ||
            (K > 1 && stride_b_K != N) || (N > 1 && stride_b_N != 1))
            throw std::runtime_error("matmul: CUDA path requires row-major contiguous matrix dims");
        if (num_batch_dims > 8)
            throw std::runtime_error("matmul: CUDA path supports at most 8 batch dims");

        BatchInfo info{};
        info.ndim = num_batch_dims;
        for (int d = 0; d < num_batch_dims; ++d) {
            info.shape[d]     = out_batch_shape[d];
            info.strides_a[d] = b_strides_a[d];   // 广播维的 stride 已是 0
            info.strides_b[d] = b_strides_b[d];
        }
        launch_matmul_kernel(pa, pb, pc, M, N, K, batch_count, info);
    } else {
        for (size_t b_idx = 0; b_idx < batch_count; ++b_idx) {
            // 计算当前 batch 索引在输入 A、B 和输出 C 中的基地址 offset
            size_t offset_a = 0;
            size_t offset_b = 0;
            size_t offset_c = 0;

            size_t temp = b_idx;
            for (int d = num_batch_dims - 1; d >= 0; --d) {
                size_t coord = temp % out_batch_shape[d];
                temp /= out_batch_shape[d];

                offset_a += coord * b_strides_a[d];
                offset_b += coord * b_strides_b[d];
                offset_c += coord * stc[d];
            }

            const float* cur_pa = pa + offset_a;
            const float* cur_pb = pb + offset_b;
            float* cur_pc = pc + offset_c;

            // 2D GEMM 内层循环
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += cur_pa[i * stride_a_M + k * stride_a_K] *
                               cur_pb[k * stride_b_K + j * stride_b_N];
                    }
                    cur_pc[i * stride_c_M + j * stride_c_N] = sum;
                }
            }
        }
    }

    // 6. 根据初始一维提升情况对输出进行 Squeeze 还原
    std::vector<int> final_shape = out->shape();
    if (squeeze_a) {
        // 移除 M 维 (倒数第二个维度)
        final_shape.erase(final_shape.end() - 2);
    }
    if (squeeze_b) {
        // 移除 N 维 (倒数第一个维度)
        final_shape.erase(final_shape.end() - 1);
    }

    // 将计算好的 Tensor 重新 reshape 为 final_shape (零拷贝修改 shape 与 strides)
    out->reshape_in_place(final_shape);

    return out;
}

void Tensor::add_(const Tensor& other) {
    if (numel_ != other.numel_)
        throw std::runtime_error("add_: shape mismatch");
    if (dtype_ != other.dtype_)
        throw std::runtime_error("add_: dtype mismatch");

    if (device_ == DeviceType::CUDA) {
        launch_add_kernel(data_ptr_, const_cast<void*>(other.data()), data_ptr_, numel_, dtype_);
        return;
    }

    switch (dtype_) {
    case DataType::FP32: {
        float* p = static_cast<float*>(data_ptr_);
        const float* q = static_cast<const float*>(other.data_ptr_);
        for (size_t i = 0; i < numel_; ++i) p[i] += q[i];
        break;
    }
    case DataType::FP16: {
        uint16_t* p = static_cast<uint16_t*>(data_ptr_);
        const uint16_t* q = static_cast<const uint16_t*>(other.data_ptr_);
        for (size_t i = 0; i < numel_; ++i) p[i] = static_cast<uint16_t>(p[i] + q[i]); // 简化：未做半精度舍入
        break;
    }
    case DataType::INT8: {
        int8_t* p = static_cast<int8_t*>(data_ptr_);
        const int8_t* q = static_cast<const int8_t*>(other.data_ptr_);
        for (size_t i = 0; i < numel_; ++i) p[i] = static_cast<int8_t>(p[i] + q[i]);
        break;
    }
    default:
        break;
    }
}

} // namespace mlspirit
