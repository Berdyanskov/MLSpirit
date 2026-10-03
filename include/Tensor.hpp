#pragma once
#include <vector>
#include <memory>
#include <string>
#include <cstddef>
#include <cstdint>

#include "Types.hpp"
#include "StorageImpl.hpp"

namespace mlspirit {

class Tensor {
private:
    std::vector<int> shape_;
    std::vector<int> strides_;
    size_t numel_;
    DeviceType device_;
    DataType dtype_;

    // 存储与视图解耦：StorageImpl 是一块带设备信息的字节内存（引用计数共享），
    // Tensor 只是它的一个"视角"（shape / strides / offset）。
    // 因此 Tensor 的拷贝是廉价的视图复制——这正是 transpose / slice 零拷贝的地基。
    std::shared_ptr<StorageImpl> storage_;
    size_t offset_ = 0;                 // 视图起点，单位：元素（与 strides 同单位）

    // 视图构造：复用既有 Storage，携带自己的 shape / strides / offset
    Tensor(std::shared_ptr<StorageImpl> storage, std::vector<int> shape,
           std::vector<int> strides, size_t offset, DataType dtype, DeviceType device);

public:
    // 构造即分配：新建的张量总是拥有一块自己的连续存储
    Tensor(const std::vector<int>& shape, DataType dtype, DeviceType device);
    ~Tensor() = default;

    // 拷贝/移动均为视图语义：共享同一块 StorageImpl
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept = default;

    // 核心元数据
    const std::vector<int>& shape() const { return shape_; }
    const std::vector<int>& strides() const { return strides_; } // 单位：元素个数，非字节
    size_t numel() const { return numel_; }
    DeviceType device() const { return device_; }
    DataType dtype() const { return dtype_; }

    // 按元素字节数拷贝，与 dtype 一致；总字节数 = numel() * element_size()
    size_t element_size() const { return dtype_element_size(dtype_); }
    size_t size_bytes() const { return numel_ * element_size(); }

    // 内存同步：迁移本质上是换一块新 StorageImpl，同一块存储上的其他视图不受影响
    void to_device(DeviceType target_device);
    void copy_from(const void* host_ptr); // 从外部缓冲区填充（长度至少 size_bytes()）
    void copy_to(void* host_ptr) const;   // 拷出到 host_ptr（长度至少 size_bytes()）

    // 零拷贝重塑：仅当新形状元素总数与 numel_ 一致且张量连续时成立
    void reshape_in_place(const std::vector<int>& new_shape);

    // 视图操作：transpose 纯元数据交换，零拷贝共享存储
    bool is_contiguous() const;
    Tensor transpose(int dim0 = -2, int dim1 = -1) const;
    Tensor contiguous() const; // 非连续 -> 按 strides 物化连续副本；连续 -> 自身浅拷贝

    // 算子接口（示例）
    static std::unique_ptr<Tensor> mm(const Tensor& a, const Tensor& b);
    static std::unique_ptr<Tensor> matmul(const Tensor& a, const Tensor& b);
    void add_(const Tensor& other); // In-place 加法

    // 视图起始地址（= storage 基址 + offset 个元素），供算子与 CUDA kernel 使用
    void* data() {
        return static_cast<char*>(storage_->data()) + offset_ * element_size();
    }
    const void* data() const {
        return static_cast<const char*>(storage_->data()) + offset_ * element_size();
    }
};

} // namespace mlspirit
