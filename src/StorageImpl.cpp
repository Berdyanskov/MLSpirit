#include "StorageImpl.hpp"
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

namespace mlspirit {

void StorageImpl::allocate_memory() {
    if (size_bytes_ == 0) return;
    if (device_ == DeviceType::CUDA) {
        cudaError_t err = cudaMalloc(&data_ptr_, size_bytes_);
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("StorageImpl: cudaMalloc failed: ") +
                                     cudaGetErrorString(err));
    } else {
        // Storage 只认字节、不认识 dtype——类型属于 view 层（Tensor）
        data_ptr_ = std::malloc(size_bytes_);
        if (data_ptr_ == nullptr)
            throw std::runtime_error("StorageImpl: malloc failed");
    }
}

void StorageImpl::free_memory() {
    if (data_ptr_ == nullptr) return;
    if (device_ == DeviceType::CUDA)
        cudaFree(data_ptr_);
    else
        std::free(data_ptr_); // 必须与 malloc 配对；对 malloc 的结果 delete 是未定义行为
    data_ptr_ = nullptr;
}

} // namespace mlspirit
