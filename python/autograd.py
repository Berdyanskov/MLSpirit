"""mlspirit 自动微分引擎（第 2 课：跑在自家张量核上的计算图）。

与 prototypes/minitorch_numpy.py 的关系：
    架构完全一致（Variable / Function / creator / backward 循环），
    唯一本质变化是数据载体 —— numpy.ndarray 换成了 mlspirit.Tensor。

占位清零（任务二完成后）：
    matmul / 转置 / add / mul / exp / sum 的前向全部走 C++/CUDA 核；
    numpy 只剩两个合法用途：测试数据的 I/O，以及构造 0 维标量常量
    （标量通过广播参与运算，见 Square.backward）。

遗留限制（路线图任务三）：
    若前向存在广播（如 bias 加法、broadcast matmul），反向需要把梯度
    沿广播维 reduce 缩回（sum_to）——"广播的逆是求和"。当前 Add / MatMul
    在"双方等形状"下梯度完全正确，广播输入下会返回未归约的梯度。
"""
import numpy as np
import mlspirit as mp


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _wrap(np_arr, device="cpu"):
    """numpy -> mp.Tensor（float32、C 连续）。"""
    return mp.from_numpy(np.ascontiguousarray(np_arr, dtype=np.float32), device=device)


def _scalar(value, device="cpu"):
    """Python 标量 -> 0 维 mp.Tensor（参与运算时自动广播到任意形状）。"""
    return _wrap(np.array(value, dtype=np.float32), device=device)


def _transpose(t):
    """交换最后两维 —— C++ 零拷贝 transpose 视图（共享存储，不搬数据）。"""
    if len(t.shape) < 2:
        raise ValueError("transpose requires ndim >= 2")
    return t.transpose(len(t.shape) - 2, len(t.shape) - 1)


def _ones_like(t):
    return _wrap(np.ones(t.shape, dtype=np.float32), t.device)


# ---------------------------------------------------------------------------
# 计算图核心（与 minitorch_numpy.py 同构，注释从简，对照阅读）
# ---------------------------------------------------------------------------

class Variable:
    """叶子是参数/输入，非叶子由 Function 创建并记录 creator。"""

    def __init__(self, data: mp.Tensor):
        self.data = data
        self.grad = None       # mp.Tensor，与 data 同形状
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        """循环版反向传播（避免递归爆栈，与 minitorch 第 1 课相同的取舍）。"""
        if self.grad is None:
            self.grad = _ones_like(self.data)
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gy = f.output.grad
            gxs = f.backward(gy)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                if gx is None:
                    continue
                # 梯度累加：一个变量被多条支路使用时，梯度是各支路之和
                x.grad = gx if x.grad is None else mp.add(x.grad, gx)
                if x.creator is not None:
                    funcs.append(x.creator)


class Function:
    def __call__(self, *inputs: Variable) -> Variable:
        ys = self.forward(*[x.data for x in inputs])
        output = Variable(ys)
        output.set_creator(self)
        self.inputs = inputs
        self.output = output
        return output

    def forward(self, *datas: mp.Tensor) -> mp.Tensor:
        raise NotImplementedError

    def backward(self, gy: mp.Tensor):
        """返回与 inputs 对齐的梯度元组。"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 算子（前向全部走 C++/CUDA 核）
# ---------------------------------------------------------------------------

class MatMul(Function):
    """Y = A @ B => dA = GY @ B^T, dB = A^T @ GY。
    B^T 走零拷贝 transpose 视图，2D 与 batched 通吃。
    限制：存在 batch 广播时反向缺 sum_to 归约（任务三）。"""

    def forward(self, a, b):
        self.a, self.b = a, b
        return mp.matmul(a, b)

    def backward(self, gy):
        ga = mp.matmul(gy, _transpose(self.b))
        gb = mp.matmul(_transpose(self.a), gy)
        return ga, gb


class Square(Function):
    """y = x^2 => gx = 2x·gy。前向 mp.mul(x, x)。"""

    def forward(self, x):
        self.x = x
        return mp.mul(x, x)

    def backward(self, gy):
        # 常数 2 以 0 维标量张量参与，广播成与 x 同形状——
        # "标量乘法"在Tensor系统里其实就是"与 0 维张量的广播乘"
        return mp.mul(mp.mul(_scalar(2.0, self.x.device), self.x), gy)


class Exp(Function):
    """y = e^x => gx = e^x·gy。前向 mp.exp。"""

    def forward(self, x):
        self.x = x
        return mp.exp(x)

    def backward(self, gy):
        return mp.mul(mp.exp(self.x), gy)


class Sum(Function):
    """全部元素求和成 0 维标量（作为 loss 的终点站）。
    => gx 是与 x 同形状的全 gy——反向恰好是"广播"：
    0 维标量加到一个全零矩阵上，自动复制到每个位置。"""
    def forward(self, x):
        self.x_shape = x.shape
        return mp.sum(x)

    def backward(self, gy):
        zeros = _wrap(np.zeros(self.x_shape, dtype=np.float32), gy.device)
        return mp.add(zeros, gy)   # 广播 = 反向传播的"复制梯度"


class Add(Function):
    """y = a + b => ga = gy, gb = gy。
    前向已支持广播；反向在广播输入下仍需 sum_to 归约（任务三），
    等形状时梯度完全正确。"""

    def forward(self, a, b):
        return mp.add(a, b)

    def backward(self, gy):
        return gy, gy


# 便捷函数（对齐 torch 的函数式调用习惯）
def matmul(a: Variable, b: Variable) -> Variable: return MatMul()(a, b)
def square(x: Variable) -> Variable: return Square()(x)
def exp(x: Variable) -> Variable: return Exp()(x)
def sum_all(x: Variable) -> Variable: return Sum()(x)
def add(a: Variable, b: Variable) -> Variable: return Add()(a, b)


def variable(np_arr, device="cpu") -> Variable:
    """numpy -> Variable 的快捷构造。"""
    return Variable(_wrap(np_arr, device))
