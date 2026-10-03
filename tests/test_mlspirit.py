"""mlspirit Python 绑定的冒烟测试（numpy 对拍，无 pytest 依赖）。

运行方式：ctest 会自动注入 PYTHONPATH 指向构建产物目录；手动运行时：
    PYTHONPATH=build python3 tests/test_mlspirit.py
CUDA 用例在无可用设备时自动跳过。
"""
import numpy as np
import mlspirit as mp

fails = 0


def check_matmul(sa, sb):
    global fails
    rng = np.random.default_rng(hash((sa, sb)) % (2**31))
    a_np = rng.standard_normal(sa).astype(np.float32)
    b_np = rng.standard_normal(sb).astype(np.float32)
    got = mp.matmul(mp.from_numpy(a_np), mp.from_numpy(b_np))
    ref = np.matmul(a_np, b_np)
    ok = got.shape == ref.shape and np.allclose(got.numpy(), ref, rtol=1e-4, atol=1e-5)
    print(f"matmul {sa}x{sb} -> {got.shape}  {'PASS' if ok else 'FAIL'}")
    fails += 0 if ok else 1


check_matmul((2, 3), (3, 4))
check_matmul((4, 5, 6), (4, 6, 3))        # 等 batch
check_matmul((4, 5, 6), (6, 3))           # b 广播到 batch
check_matmul((2, 1, 3, 4), (2, 4, 5))     # 双侧多维广播
check_matmul((5,), (5,))                  # 1D@1D -> 标量
check_matmul((5,), (5, 7))                # 1D@2D
check_matmul((6, 5), (5,))                # 2D@1D

# 元数据
t = mp.from_numpy(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
assert t.shape == (2, 3, 4) and t.strides == (12, 4, 1) and t.numel == 24
assert t.dtype == "fp32" and t.device == "cpu"

# 0 维标量往返
v = mp.from_numpy(np.array([1, 2, 3], dtype=np.float32))
s = mp.matmul(v, v)
assert s.shape == () and abs(float(s.numpy()) - 14.0) < 1e-6

# 非 float32 输入必须明确报错（不允许静默 astype）
try:
    mp.from_numpy(np.arange(3, dtype=np.float64))
    print("dtype guard FAIL")
    fails += 1
except ValueError:
    print("dtype guard PASS")

# CUDA 用例：设备不可用时跳过
try:
    a_np = np.arange(12, dtype=np.float32).reshape(3, 4) / 10
    b_np = np.arange(20, dtype=np.float32).reshape(4, 5) / 10
    a = mp.from_numpy(a_np, device="cuda")
    b = mp.from_numpy(b_np, device="cuda")
    got = mp.matmul(a, b).to("cpu").numpy()  # 也覆盖 .to() 链式调用
    ok = np.allclose(got, a_np @ b_np, rtol=1e-4, atol=1e-5)
    print(f"cuda matmul {'PASS' if ok else 'FAIL'}")
    fails += 0 if ok else 1
except Exception as e:
    print(f"cuda skipped ({e})")

print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
raise SystemExit(0 if fails == 0 else 1)
