"""python/autograd.py 的数值梯度验证（解析梯度 vs 中心差分）。

ctest 自动注入 PYTHONPATH（扩展模块目录 + python/ 源码目录）。CPU 跑完即过；
CUDA 不参与本测试（逐元素占位算子走 numpy 往返，与设备无关）。
"""
import numpy as np
import mlspirit as mp
from mlspirit import autograd as ag

fails = 0


def report(name, ok):
    global fails
    print(f"{name:<28} {'PASS' if ok else 'FAIL'}")
    fails += 0 if ok else 1


def close(grad_tensor, ref_np, rtol=2e-3, atol=1e-4):
    got = grad_tensor.numpy()
    return got.shape == ref_np.shape and np.allclose(got, ref_np, rtol=rtol, atol=atol)


def numerical_grad(f, x_np, eps=1e-3):
    """对 f: np.ndarray -> 0 维 mp.Tensor 做中心差分。"""
    grad = np.zeros_like(x_np)
    it = np.nditer(x_np, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        plus = x_np.copy(); plus[idx] += eps
        minus = x_np.copy(); minus[idx] -= eps
        grad[idx] = (f(plus).data.numpy() - f(minus).data.numpy()) / (2 * eps)
    return grad


rng = np.random.default_rng(7)

# 1) square：y = sum(x^2) => gx = 2x
x_np = rng.standard_normal((2, 3)).astype(np.float32)
x = ag.variable(x_np)
ag.sum_all(ag.square(x)).backward()
report("square grad (analytic)", close(x.grad, 2 * x_np))
report("square grad (numerical)",
       np.allclose(x.grad.numpy(), numerical_grad(lambda a: ag.sum_all(ag.square(ag.variable(a))), x_np),
                   rtol=1e-2, atol=1e-3))

# 2) exp：y = sum(exp(x)) => gx = exp(x)
x = ag.variable(x_np)
ag.sum_all(ag.exp(x)).backward()
report("exp grad (analytic)", close(x.grad, np.exp(x_np)))

# 3) matmul：y = sum(X @ W) => gX = 1 @ W^T, gW = X^T @ 1
w_np = rng.standard_normal((3, 4)).astype(np.float32)
x, w = ag.variable(x_np), ag.variable(w_np)
ag.sum_all(ag.matmul(x, w)).backward()
ones = np.ones((2, 4), dtype=np.float32)
report("matmul grad X", close(x.grad, ones @ w_np.T))
report("matmul grad W", close(w.grad, x_np.T @ ones))

# 4) 复合链 + 数值验证：y = sum(exp(square(X @ W)))
def chain(a):
    return ag.sum_all(ag.exp(ag.square(ag.matmul(ag.variable(a), ag.variable(w_np)))))

x, w = ag.variable(x_np), ag.variable(w_np)
ag.sum_all(ag.exp(ag.square(ag.matmul(x, w)))).backward()
num = numerical_grad(chain, x_np)
report("chain grad (numerical)", close(x.grad, num, rtol=1e-2, atol=1e-2))

# 5) 梯度的多支路累加：y = sum(square(x) + exp(x)) => gx = 2x + e^x
#    （x 被两条支路使用，backward 必须把两份梯度加起来）
x = ag.variable(x_np)
ag.sum_all(ag.add(ag.square(x), ag.exp(x))).backward()
report("multi-branch accum", close(x.grad, 2 * x_np + np.exp(x_np)))

# 6) batched matmul 反向：transpose 视图打通 batch 维转置
#    Y[b] = X[b] @ W[b]; sum(Y) 的梯度：gX = ones @ W^T, gW = X^T @ ones
x3_np = rng.standard_normal((2, 3, 4)).astype(np.float32)
w3_np = rng.standard_normal((2, 4, 5)).astype(np.float32)
x, w = ag.variable(x3_np), ag.variable(w3_np)
ag.sum_all(ag.matmul(x, w)).backward()
ones3 = np.ones((2, 3, 5), dtype=np.float32)
ref_gx = np.matmul(ones3, w3_np.transpose(0, 2, 1))
ref_gw = np.matmul(x3_np.transpose(0, 2, 1), ones3)
report("batched matmul grad X", close(x.grad, ref_gx))
report("batched matmul grad W", close(w.grad, ref_gw))

# 7) 广播 add 的反向（bias 模式）：y = sum(x + b), x:[2,3], b:[3]
#    gx = gy 原样（x 未被广播），gb = gy 沿 dim0 坍缩 → 每个分量是 2
b_np = rng.standard_normal((3,)).astype(np.float32)
x, b = ag.variable(x_np), ag.variable(b_np)
ag.sum_all(ag.add(x, b)).backward()
report("bcast add grad X", close(x.grad, np.ones((2, 3), np.float32)))
report("bcast add grad b", close(b.grad, np.full((3,), 2.0, np.float32)))

# 8) broadcast matmul 反向：X[2,3,4] @ W[4,5] → sum
#    gX = ones[2,3,5] @ W^T；gW = Σ_b X[b]^T @ ones[3,5]（batch 维坍缩）
w2_np = rng.standard_normal((4, 5)).astype(np.float32)
x, w = ag.variable(x3_np), ag.variable(w2_np)
ag.sum_all(ag.matmul(x, w)).backward()
ref_gx2 = np.matmul(np.ones((2, 3, 5), np.float32), w2_np.T)
ref_gw2 = np.matmul(x3_np.transpose(0, 2, 1), np.ones((2, 3, 5), np.float32)).sum(axis=0)
report("bcast matmul grad X", close(x.grad, ref_gx2))
report("bcast matmul grad W", close(w.grad, ref_gw2))

# 9) broadcast matmul 复合链的数值梯度（sum_to 在更深的图里）
def chain_bmm(a):
    return ag.sum_all(ag.square(ag.matmul(ag.variable(a), ag.variable(w2_np))))
x, w = ag.variable(x3_np), ag.variable(w2_np)
ag.sum_all(ag.square(ag.matmul(x, w))).backward()
num_bmm = numerical_grad(chain_bmm, x3_np)
report("bcast matmul chain (num)", close(x.grad, num_bmm, rtol=1e-2, atol=1e-2))

print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
raise SystemExit(0 if fails == 0 else 1)
