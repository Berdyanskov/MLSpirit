# MicroTensor (MLSpirit)

A machine learning library built from scratch for teaching purposes. The main line is
 walking through every detail behind "one block of storage + a handful of metadata": strides, broadcasting, dual-device (CPU/CUDA) memory management, tiled GEMM, batched matmul, and autodiff.

## Project layout

```
├── include/            # Public headers
│   ├── mlspirit.hpp    #   Umbrella header (the only one users need to include)
│   ├── Tensor.hpp      #   Tensor class: shape / strides / dtype / device
│   └── kernel.h        #   Host-side launch interface of the CUDA kernels
├── src/                # Implementation
│   ├── Tensor.cpp      #   Memory management, CPU ops, mm/matmul dispatch
│   ├── cuda_add.cu     #   Element-wise add kernel (template across dtypes)
│   └── cuda_matmul.cu  #   Tiled GEMM (SMEM + register blocking) and batched matmul
├── bindings/           # pybind11 layer (translation only, no compute logic)
├── tests/              # C++/Python tests (no framework dependency), run via `ctest`
├── python/
│   ├── autograd.py     # Autodiff engine on top of mlspirit.Tensor (lesson 2)
│   └── prototypes/
│       └── minitorch_numpy.py  # Historical prototype: numpy-based autodiff (lesson 1)
├── visualize/          # Interactive teaching visualizations (open in a browser)
│   └── gemm-tiled-matmul.html  # Step-by-step animation of the tiled GEMM pipeline
└── CMakeLists.txt
```

## Build and test

Requirements: CMake ≥ 3.18, CUDA Toolkit (with nvcc); the Python bindings additionally
need `pip install pybind11 numpy`.

```bash
cmake -B build
cmake --build build -j
ctest --test-dir build --output-on-failure   # includes the Python binding smoke tests
```

GPU tests skip automatically on machines without a CUDA device (they are still
compiled, so CI can verify them).

## Quick start (Python)

After building, put the extension module directory on `PYTHONPATH` and
`import mlspirit as mp`:

```bash
PYTHONPATH=build python3
```

```python
import numpy as np
import mlspirit as mp

a = mp.from_numpy(np.random.randn(4, 5, 6).astype(np.float32), device="cuda")
b = mp.from_numpy(np.random.randn(6, 3).astype(np.float32), device="cuda")
c = mp.matmul(a, b)          # dispatches to the CUDA batched kernel (b broadcast over batch)
c.shape                      # (4, 5, 3)
c.to("cpu").numpy()          # copy back to host as a numpy array
```

## Quick start (C++)

All public APIs live in the `mlspirit` namespace — one header is all you need:

```cpp
#include "mlspirit.hpp"
using namespace mlspirit;   // or the short alias mp::

Tensor a({2, 3}, DataType::FP32, DeviceType::CPU);
Tensor b({3, 2}, DataType::FP32, DeviceType::CPU);
a.to_device(DeviceType::CUDA);          // move to GPU
b.to_device(DeviceType::CUDA);
auto c = Tensor::matmul(a, b);          // dispatches to the CUDA batched kernel
c->to_device(DeviceType::CPU);
```

## Roadmap (done → in progress)

- [x] Tensor type: shape / strides / numel, dual-device (CPU/CUDA) memory management
- [x] Zero-copy `reshape_in_place`, strides derivation
- [x] `mm` (strict 2D): naive CPU loop + CUDA tiled GEMM (128×128 tile, SMEM + 8×8 register blocking)
- [x] `matmul`: N-D batched + broadcasting (batch folded into `gridDim.z`, broadcast via stride-0)
- [x] Python bindings (pybind11): `import mlspirit as mp`, explicit numpy conversion
- [x] Autodiff engine (`python/autograd.py`): define-by-run graph + iterative backprop +
      gradient accumulation, verified by numerical gradient checks; matmul backward runs
      entirely on the C++ core
- [ ] Views (transpose / slice without copying) and `contiguous()` — unlocks batched matmul backward
- [ ] Broadcast element-wise ops (add/mul/exp/sum as C++ kernels) — replaces the numpy
      placeholders in autograd
- [ ] nn module: `Linear` layer (forward = matmul + bias)
