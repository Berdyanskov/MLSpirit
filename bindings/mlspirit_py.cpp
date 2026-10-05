// ============================================================================
// mlspirit_py.cpp — Python 绑定层（pybind11）
//
// 设计原则：Python 侧只做"翻译"，不含任何计算逻辑——
//   numpy.ndarray  <-->  mlspirit::Tensor  <-->  C++/CUDA 算子
// v0 只暴露 FP32（与 CUDA tiled kernel 支持的精度一致）；
// FP16/INT8 的 dtype 枚举已导出，构造会明确报错而不是静默转换。
// ============================================================================
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "mlspirit.hpp"

namespace py = pybind11;
using namespace mlspirit;

namespace {

// 校验输入是 C 连续的 float32 数组；否则明确报错（教学上不赞成静默 astype）
py::array_t<float> as_f32_contiguous(const py::array& arr, const char* arg_name) {
    if (!py::isinstance<py::array_t<float>>(arr))
        throw std::invalid_argument(std::string(arg_name) +
            ": expected a float32 numpy array (got dtype=" +
            py::str(arr.attr("dtype")).cast<std::string>() + ")");
    auto a = arr.cast<py::array_t<float>>();
    if (!(a.flags() & py::array::c_style))
        throw std::invalid_argument(std::string(arg_name) +
            ": expected a C-contiguous array; call np.ascontiguousarray() first");
    return a;
}

DeviceType parse_device(const std::string& s) {
    if (s == "cpu")  return DeviceType::CPU;
    if (s == "cuda") return DeviceType::CUDA;
    throw std::invalid_argument("device must be \"cpu\" or \"cuda\", got: " + s);
}

std::string device_str(DeviceType d) { return d == DeviceType::CUDA ? "cuda" : "cpu"; }

std::string dtype_str(DataType d) {
    switch (d) {
        case DataType::FP32: return "fp32";
        case DataType::FP16: return "fp16";
        case DataType::INT8: return "int8";
    }
    return "?";
}

} // namespace

PYBIND11_MODULE(_mlspirit, m) {
    m.doc() = "MLSpirit C++/CUDA 扩展核心（下划线前缀 = 包私有）。\n"
              "用户请 import 外层 Python 包: import mlspirit as mp"
              "（对标 torch 的 torch._C 扩展）";

    // 枚举导出（与 C++ 侧一一对应）
    py::enum_<DeviceType>(m, "Device")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA);

    py::enum_<DataType>(m, "DType")
        .value("FP32", DataType::FP32)
        .value("FP16", DataType::FP16)
        .value("INT8", DataType::INT8);

    py::class_<Tensor>(m, "Tensor")
        // mp.Tensor([2, 3], dtype=..., device="cpu")
        .def(py::init([](std::vector<int> shape, DataType dtype, const std::string& device) {
                 if (dtype != DataType::FP32)
                     throw std::invalid_argument("v0 only supports FP32 tensors");
                 return std::make_unique<Tensor>(std::move(shape), dtype, parse_device(device));
             }),
             py::arg("shape"), py::arg("dtype") = DataType::FP32, py::arg("device") = "cpu")

        // 元数据（只读属性，对齐 numpy/torch 的访问习惯）
        .def_property_readonly("shape",   [](const Tensor& t) { return py::tuple(py::cast(t.shape())); })
        .def_property_readonly("strides", [](const Tensor& t) { return py::tuple(py::cast(t.strides())); })
        .def_property_readonly("numel",   &Tensor::numel)
        .def_property_readonly("device",  [](const Tensor& t) { return device_str(t.device()); })
        .def_property_readonly("dtype",   [](const Tensor& t) { return dtype_str(t.dtype()); })

        .def("__repr__", [](const Tensor& t) {
            std::string s = "mlspirit.Tensor(shape=[";
            for (size_t i = 0; i < t.shape().size(); ++i) {
                s += std::to_string(t.shape()[i]);
                if (i + 1 < t.shape().size()) s += ", ";
            }
            s += "], dtype=" + dtype_str(t.dtype()) + ", device=" + device_str(t.device()) + ")";
            return s;
        })

        // t.to("cuda")：原地迁移并返回自身（链式），对齐 torch 的 .to() 手感
        .def("to", [](Tensor& self, const std::string& device) -> Tensor& {
                 self.to_device(parse_device(device));
                 return self;
             },
             py::arg("device"), py::return_value_policy::reference_internal)

        // t.numpy()：拷回 host 并包装为 numpy 数组（CUDA 张量会先 DtoH）
        .def("numpy", [](const Tensor& t) {
            std::vector<py::ssize_t> shape(t.shape().begin(), t.shape().end());
            py::array_t<float> out(shape);
            t.copy_to(out.mutable_data());
            return out;
        })

        // 视图操作：transpose 零拷贝（返回共享存储的视图）
        .def("transpose", &Tensor::transpose, py::arg("dim0") = -2, py::arg("dim1") = -1)
        .def("contiguous", &Tensor::contiguous)
        .def_property_readonly("is_contiguous", &Tensor::is_contiguous)

        .def("add_", &Tensor::add_, py::arg("other"));

    // numpy -> Tensor
    m.def("from_numpy",
          [](const py::array& arr, const std::string& device) {
              auto a = as_f32_contiguous(arr, "from_numpy(arr)");
              std::vector<int> shape(a.shape(), a.shape() + a.ndim());
              auto t = std::make_unique<Tensor>(shape, DataType::FP32, parse_device(device));
              t->copy_from(a.data());
              return t;
          },
          py::arg("arr"), py::arg("device") = "cpu");

    // 自由函数形式算子（对齐 mp.matmul(a, b) 的习惯）
    m.def("mm", &Tensor::mm, py::arg("a"), py::arg("b"));
    m.def("matmul", &Tensor::matmul, py::arg("a"), py::arg("b"));

    // 逐元素与归约算子（任务二；全限定名避免与 <cmath> 的 ::exp 歧义）
    m.def("add", &mlspirit::add, py::arg("a"), py::arg("b"));
    m.def("sub", &mlspirit::sub, py::arg("a"), py::arg("b"));
    m.def("mul", &mlspirit::mul, py::arg("a"), py::arg("b"));
    m.def("div", &mlspirit::div, py::arg("a"), py::arg("b"));
    m.def("exp", &mlspirit::exp, py::arg("x"));
    m.def("sum", &mlspirit::sum, py::arg("x"));

    // 形状归约（广播的逆；广播算子反向的基石）
    m.def("sum_to", &mlspirit::sum_to, py::arg("x"), py::arg("shape"));
}
