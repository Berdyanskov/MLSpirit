#pragma once
#include <vector>
#include <memory>
#include <string>
#include <cstddef>
#include <cstdint>

namespace mlspirit {

class StorageImpl {
private:
    void* data_ptr_ = nullptr;
    size_t size_bytes_ = 0;
    DeviceType device_;

    void allocate_memory();
    void free_memory();

public:
    StorageImpl(size_t size_bytes, DeviceType device) 
        : size_bytes_(size_bytes), device_(device) {
        allocate_memory();
    }

    ~StorageImpl() { free_memory(); }

    StorageImpl(const StorageImpl&) = delete;
    StorageImpl& operator=(const StorageImpl&) = delete;

    void* data() { return data_ptr_; }
    const void* data() const { return data_ptr_; }
    size_t size_bytes() const { return size_bytes_; }
    DeviceType device() const { return device_; }
};

}