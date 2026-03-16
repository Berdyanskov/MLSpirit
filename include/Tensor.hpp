#include <vector>
#include <memory>
#include <string>
#include <cstddef>
#include <cstdint>

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

class Tensor {
private:
    std::vector<int> shape_;
    size_t numel_;
    DeviceType device_;
    DataType dtype_;
    void* data_ptr_ = nullptr;

    void allocate_memory();
    void free_memory();
public:
    // 构造与析构
    Tensor(const std::vector<int>& shape, DataType dtype, DeviceType device);
    ~Tensor();

    // 禁止隐式拷贝，鼓励使用显式 CopyToDevice 或 Clone
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // 移动构造函数（支持高效返回对象）
    Tensor(Tensor&& other) noexcept;

    // 核心元数据
    const std::vector<int>& shape() const { return shape_; }
    size_t numel() const { return numel_; }
    DeviceType device() const { return device_; }
    DataType dtype() const { return dtype_; }

    // 按元素字节数拷贝，与 dtype 一致；总字节数 = numel() * element_size()
    size_t element_size() const { return dtype_element_size(dtype_); }
    size_t size_bytes() const { return numel_ * element_size(); }

    // 内存同步
    void to_device(DeviceType target_device);
    void copy_from(const void* host_ptr); // 从外部缓冲区填充（长度至少 size_bytes()）
    void copy_to(void* host_ptr) const;   // 拷出到 host_ptr（长度至少 size_bytes()）

    // 算子接口（示例）
    static std::unique_ptr<Tensor> matmul(const Tensor& a, const Tensor& b);
    void add_(const Tensor& other); // In-place 加法
    
    // 获取原始数据指针（供 CUDA Kernel 调用）
    void* data() { return data_ptr_; }
    const void* data() const { return data_ptr_; }
};