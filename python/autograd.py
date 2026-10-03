"""mlspirit 自动微分引擎（第 2 课：跑在自家张量核上的计算图）。

与 prototypes/minitorch_numpy.py 的关系：
    架构完全一致（Variable / Function / creator / backward 循环），
    唯一本质变化是数据载体 —— numpy.ndarray 换成了 mlspirit.Tensor，
    前向计算因此自动获得 C++/CUDA 算子的全部能力。

占位实现说明（重要的工程诚实）：
    C++ 核目前只有 mm/matmul/add_ 三个算子。本文件中：
    - matmul 的前向/反向都直接调用 C++ 核；
    - 转置、clone、逐元素 square/exp 暂用 numpy 往返实现，
      每个占位点都标注了对应的 C++ 侧待办（见各函数 docstring）。
    这些占位会随 C++ 算子的补齐逐一替换，Python 侧框架代码不必再动。
"""
import numpy as np
import mlspirit as mp


# ---------------------------------------------------------------------------
# 占位工具（C++ 侧待办：transpose view / clone / 逐元素 exp·mul / out-of-place add）
# ---------------------------------------------------------------------------

def _wrap(np_arr, device="cpu"):
    """numpy -> mp.Tensor（float32、C 连续）。"""
    return mp.from_numpy(np.ascontiguousarray(np_arr, dtype=np.float32), device=device)


def _clone(t):
    """深拷贝。待办：C++ 实现 Tensor::clone() 后替换。"""
    return _wrap(t.numpy(), t.device)


def _transpose(t):
    """交换最后两维 —— C++ 零拷贝 transpose 视图（共享存储，不搬数据）。"""
    if len(t.shape) < 2:
        raise ValueError("transpose requires ndim >= 2")
    return t.transpose(len(t.shape) - 2, len(t.shape) - 1)


def _add(a, b):
    """非原地加法。待办：C++ 实现 out-of-place add 后替换。"""
    c = _clone(a)
    c.add_(b)
    return c


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
                x.grad = gx if x.grad is None else _add(x.grad, gx)
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
# 算子
# ---------------------------------------------------------------------------

class MatMul(Function):
    """Y = A @ B => dA = GY @ B^T, dB = A^T @ GY。
    前向/反向全部由 C++ matmul 完成（含 CUDA 分派），B^T 走零拷贝 transpose 视图，
    2D 与 batched 通吃。
    遗留限制：若前向时 A/B 之间存在 batch 广播，反向还需沿广播维 reduce
    （sum_to，属路线图任务三）；等 batch 或同形状下梯度完全正确。"""

    def forward(self, a, b):
        self.a, self.b = a, b
        return mp.matmul(a, b)

    def backward(self, gy):
        ga = mp.matmul(gy, _transpose(self.b))
        gb = mp.matmul(_transpose(self.a), gy)
        return ga, gb


class Square(Function):
    """y = x^2 => gx = 2x·gy。占位：逐元素运算暂走 numpy。"""

    def forward(self, x):
        self.x = x
        return _wrap(np.square(x.numpy()), x.device)

    def backward(self, gy):
        gx = 2.0 * self.x.numpy() * gy.numpy()
        return _wrap(gx, self.x.device)


class Exp(Function):
    """y = e^x => gx = e^x·gy。占位：逐元素运算暂走 numpy。"""

    def forward(self, x):
        self.x = x
        return _wrap(np.exp(x.numpy()), x.device)

    def backward(self, gy):
        gx = np.exp(self.x.numpy()) * gy.numpy()
        return _wrap(gx, self.x.device)


class Sum(Function):
    """全部元素求和成 0 维标量（作为 loss 的终点站）。
    => gx 是与 x 同形状的全 gy（标量广播），梯度恰好是"广播的逆运算"——
    这是理解反向传播里 reduce 操作的最小例子。"""

    def forward(self, x):
        self.x_shape = x.shape
        return _wrap(np.array(x.numpy().sum(), dtype=np.float32).reshape(()), x.device)

    def backward(self, gy):
        scale = float(gy.numpy())
        return _wrap(np.full(self.x_shape, scale, dtype=np.float32), gy.device)


class Add(Function):
    """y = a + b（同形状）。=> ga = gy, gb = gy。
    注意广播加法的反向需要把 gy 沿广播维 reduce 回去（Sum 的逆），
    留待 C++ 广播逐元素算子落地后一并处理。"""

    def forward(self, a, b):
        return _add(a, b)

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
