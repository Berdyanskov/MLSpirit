#include "Tensor.hpp"
#include <cstring>
#include <stdexcept>
#include "kernel.h"
#include <cuda_runtime.h>

void Tensor::allocate_memory() {
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
    data_ptr_ = nullptr;
}

Tensor::Tensor(const std::vector<int>& shape, DataType dtype, DeviceType device)
    : shape_(shape), device_(device), dtype_(dtype) {
    numel_ = 1;
    for (const int& d : shape_) numel_ *= d;
    data_ptr_ = nullptr;
    allocate_memory();
}

Tensor::~Tensor() {
    free_memory();
}

Tensor::Tensor(Tensor&& other) noexcept: shape_(std::move(other.shape_)), numel_(other.numel_), device_(other.device_),
    dtype_(other.dtype_), data_ptr_(other.data_ptr_) {
    other.data_ptr_ = nullptr;
    other.numel_ = 0;
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

std::unique_ptr<Tensor> Tensor::matmul(const Tensor& a, const Tensor& b) {
    const std::vector<int>& sa = a.shape();
    const std::vector<int>& sb = b.shape();
    if (sa.size() != 2 || sb.size() != 2)
        throw std::runtime_error("matmul: only 2D tensors supported");
    int M = sa[0], K = sa[1], N = sb[1];
    if (sb[0] != K)
        throw std::runtime_error("matmul: shape mismatch (K)");
    if (a.dtype() != DataType::FP32 || b.dtype() != DataType::FP32)
        throw std::runtime_error("matmul: only FP32 supported in this stub");

    auto out = std::make_unique<Tensor>(std::vector<int>{M, N}, DataType::FP32, a.device());
    const float* pa = static_cast<const float*>(a.data());
    const float* pb = static_cast<const float*>(b.data());
    float* pc = static_cast<float*>(out->data());

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.f;
            for (int k = 0; k < K; ++k)
                sum += pa[i * K + k] * pb[k * N + j];
            pc[i * N + j] = sum;
        }
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
