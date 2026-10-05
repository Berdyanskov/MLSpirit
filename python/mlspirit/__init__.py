"""MLSpirit — 从零实现的机器学习库（教学项目）。

包结构与 PyTorch 同构：
    torch      == python/mlspirit/      （Python 包，用户 import 的入口）
    torch._C   == mlspirit._mlspirit    （pybind11 编译出的 C++/CUDA 扩展）
    torch/autograd == mlspirit/autograd.py （Python 编写的计算图引擎）

本 __init__ 做两件事：
    1. 把扩展核心的全部公开符号 re-export 到包命名空间，
       使用户只需 `import mlspirit as mp`（mp.Tensor / mp.matmul / mp.add ...）；
    2. 把 Python autograd 引擎挂为子命名空间 `mp.autograd`，
       用法如 `mp.autograd.Variable(...)` —— 对应 torch.autograd 的角色。
"""
from ._mlspirit import *      # noqa: F401,F403  Tensor / Device / DType / matmul / add / ...
from ._mlspirit import Tensor, Device, DType, from_numpy, mm, matmul, add, sub, mul, div, exp, sum, sum_to

from . import autograd        # noqa: F401  mp.autograd.Variable / matmul / ...
