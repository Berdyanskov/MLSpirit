#include "StorageImpl.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include "kernel.h"
#include <cuda_runtime.h>

void StorageImpl::allocate_memory(){
    data_ptr_ = (void *)malloc(size_bytes_);
}

void StorageImpl::free_memory(){
    delete data_ptr_;
}