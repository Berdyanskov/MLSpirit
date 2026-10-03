#include "Tensor.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include "kernel.h"
#include <cuda_runtime.h>

namespace mlspirit {

Tensor::Tensor(const std::vector<int>& shape, DataType dtype, DeviceType device)
    : shape_(shape), device_(device), dtype_(dtype) {
    numel_ = 1;
    for (int d : shape_) numel_ *= d;
    strides_ = std::vector<int>(shape_.size());
    if (!strides_.empty()) {
        strides_.back() = 1;
        for (int i = static_cast<int>(strides_.size()) - 2; i >= 0; --i)
            strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
    // 分配策略全部收拢进 StorageImpl；Tensor 层再也见不到 cudaMalloc / delete
    storage_ = std::make_shared<StorageImpl>(size_bytes(), device_);
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
      numel_(other.numel_), device_(other.device_), dtype_(other.dtype_),
      storage_(std::move(other.storage_)), offset_(other.offset_) {
    other.numel_ = 0;
    other.offset_ = 0;
    other.shape_.clear();
    other.strides_.clear();
}

void Tensor::to_device(DeviceType target_device) {
    if (device_ == target_device) return;
    // 迁移只发生在"实体"上：非连续视图先物化，再做平板跨端拷贝。
    // *this 重指向新 storage，共享旧存储的其他视图不受影响（shared_ptr 为旧块续命）
    if (!is_contiguous())
        *this = contiguous();
    auto new_storage = std::make_shared<StorageImpl>(size_bytes(), target_device);
    cudaMemcpyKind kind = (target_device == DeviceType::CUDA)
                              ? cudaMemcpyHostToDevice
                              : cudaMemcpyDeviceToHost;
    cudaError_t err = cudaMemcpy(new_storage->data(), data(), size_bytes(), kind);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("Tensor::to_device: ") + cudaGetErrorString(err));
    storage_ = std::move(new_storage);
    device_ = target_device;
}

void Tensor::copy_from(const void* host_ptr) {
    if (host_ptr == nullptr) return;
    // 向非连续视图平板写入会把数据写错位置；散写由未来的 copy kernel 负责
    if (!is_contiguous())
        throw std::runtime_error("copy_from: writing into a non-contiguous view is not supported; call contiguous() first");
    if (device_ == DeviceType::CUDA)
        cudaMemcpy(data(), host_ptr, size_bytes(), cudaMemcpyHostToDevice);
    else
        std::memcpy(data(), host_ptr, size_bytes());
}

void Tensor::copy_to(void* host_ptr) const {
    if (host_ptr == nullptr) return;
    // 视图也能读：先物化（contiguous 对连续张量是零成本浅拷贝）再平板拷出
    if (!is_contiguous()) {
        contiguous().copy_to(host_ptr);
        return;
    }
    if (device_ == DeviceType::CUDA)
        cudaMemcpy(host_ptr, data(), size_bytes(), cudaMemcpyDeviceToHost);
    else
        std::memcpy(host_ptr, data(), size_bytes());
}

void Tensor::reshape_in_place(const std::vector<int>& new_shape) {
    size_t new_numel = 1;
    for (int d : new_shape) new_numel *= d;
    if (new_numel != numel_)
        throw std::runtime_error("reshape_in_place: element count mismatch");
    if (!is_contiguous())
        throw std::runtime_error("reshape_in_place: tensor is not contiguous; call contiguous() first");
    shape_ = new_shape;
    // 与构造函数相同的行主序 strides 推导（连续张量前提）
    strides_.assign(new_shape.size(), 0);
    if (!strides_.empty()) {
        strides_.back() = 1;
        for (int i = static_cast<int>(strides_.size()) - 2; i >= 0; --i)
            strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

// ---------------- 视图（view）----------------
// 视图 = 共享 Storage + 自己的 shape/strides/offset。
// 关键认知：transpose 不碰任何数据，只交换两条元数据；读元素时按
//   地址 = base + offset + Σ idx[d] * strides[d]
// 换算，所以"转置后的读取"自然成立。

Tensor::Tensor(std::shared_ptr<StorageImpl> storage, std::vector<int> shape,
               std::vector<int> strides, size_t offset, DataType dtype, DeviceType device)
    : shape_(std::move(shape)), strides_(std::move(strides)), device_(device),
      dtype_(dtype), storage_(std::move(storage)), offset_(offset) {
    numel_ = 1;
    for (int d : shape_) numel_ *= d;
}

bool Tensor::is_contiguous() const {
    // 从最后一维向前验证行主序布局；shape 为 1 的维豁免（其 stride 不影响布局）
    size_t expected = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        if (shape_[i] == 1) continue;
        if (strides_[i] != static_cast<int>(expected)) return false;
        expected *= static_cast<size_t>(shape_[i]);
    }
    return true;
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    const int ndim = static_cast<int>(shape_.size());
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim)
        throw std::runtime_error("transpose: dim out of range");
    auto new_shape = shape_;
    auto new_strides = strides_;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);
    return Tensor(storage_, std::move(new_shape), std::move(new_strides),
                  offset_, dtype_, device_);
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;   // 已连续：浅拷贝即"自身"，零成本
    // 唯一会真正搬数据的地方：按 strides 逐元素寻址，物化为连续张量。
    // 正确性优先版：CUDA 上逐元素 D2D memcpy（生产框架用 elementwise copy kernel）
    Tensor out(shape_, dtype_, device_);
    if (numel_ == 0) return out; // 空张量无元素可搬（也避免后续对 0 维取模）
    const bool cuda = (device_ == DeviceType::CUDA);
    const size_t es = element_size();
    const char* src = static_cast<const char*>(data());
    char* dst = static_cast<char*>(out.data());
    for (size_t idx = 0; idx < numel_; ++idx) {
        // 扁平下标 -> 多维坐标 -> 按 strides 换算源偏移（视图读取的唯一入口）
        size_t off = 0, t = idx;
        for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
            const size_t dim = static_cast<size_t>(shape_[d]);
            off += (t % dim) * static_cast<size_t>(strides_[d]);
            t /= dim;
        }
        if (cuda)
            cudaMemcpy(dst + idx * es, src + off * es, es, cudaMemcpyDeviceToDevice);
        else
            std::memcpy(dst + idx * es, src + off * es, es);
    }
    return out;
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
        // kernel 只认行主序连续内存：非连续视图在入口处统一物化
        if (!a.is_contiguous() || !b.is_contiguous()) {
            Tensor a2 = a.contiguous(), b2 = b.contiguous();
            return mm(a2, b2);   // 最多递归一层
        }
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
    // CUDA 路径只认连续张量：非连续视图在此统一物化（最多递归一层）
    if (a.device() == DeviceType::CUDA && (!a.is_contiguous() || !b.is_contiguous())) {
        Tensor a2 = a.contiguous(), b2 = b.contiguous();
        return matmul(a2, b2);
    }

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
    // 平板循环只适用于连续张量；视图的逐元素散写由后续的广播算子核负责
    if (!is_contiguous() || !other.is_contiguous())
        throw std::runtime_error("add_: non-contiguous tensors not supported yet; call contiguous() first");

    if (device_ == DeviceType::CUDA) {
        launch_add_kernel(data(), const_cast<void*>(other.data()), data(), numel_, dtype_);
        return;
    }

    switch (dtype_) {
    case DataType::FP32: {
        float* p = static_cast<float*>(data());
        const float* q = static_cast<const float*>(other.data());
        for (size_t i = 0; i < numel_; ++i) p[i] += q[i];
        break;
    }
    case DataType::FP16: {
        uint16_t* p = static_cast<uint16_t*>(data());
        const uint16_t* q = static_cast<const uint16_t*>(other.data());
        for (size_t i = 0; i < numel_; ++i) p[i] = static_cast<uint16_t>(p[i] + q[i]); // 简化：未做半精度舍入
        break;
    }
    case DataType::INT8: {
        int8_t* p = static_cast<int8_t*>(data());
        const int8_t* q = static_cast<const int8_t*>(other.data());
        for (size_t i = 0; i < numel_; ++i) p[i] = static_cast<int8_t>(p[i] + q[i]);
        break;
    }
    default:
        break;
    }
}

} // namespace mlspirit
