"""（历史原型 · 第 1 课）numpy 版微型自动微分引擎。

本文件是项目早期的学习稿：用 numpy 作为数据载体，
走通了 Variable / Function / 计算图 / backward 的完整概念闭环。

它故意保留、不再修改——作为"先理解概念、再深入工程"的对照组。
正式的、跑在 mlspirit.Tensor（C++/CUDA 张量核）之上的 autograd 实现见：
    python/autograd.py
"""
import numpy as np

class Variable:
    def __init__(self, data:np.ndarray):
        self.data = data
        self.grad = None
        self.creator = None
    def set_creator(self, func):
        self.creator = func
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        # 递归实现
        # f = self.creator
        # if f is not None:
        #     x = f.input
        #     x.grad = f.backward(self.grad)
        #     x.backward()
        # 循环实现
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, input:Variable)-> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output
    
    def forward(self, x:np.ndarray):
        raise NotImplementedError()

    def backward(self, x):
        raise NotImplementedError()

def numerical_diff(f, x, eps = 1e-4):
    x1 = Variable(x - eps)
    x2 = Variable(x + eps)
    y1 = f(x1)
    y2 = f(x2)
    return (y1.data - y2.data) / (2 * eps)

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
    
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)


    
