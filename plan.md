# MLspirit - 机器学习库开发规划

> 目标：通过从零实现 ML 库，提升项目经验、开发能力，以及**对机器学习本质的深入理解**。

---

## 一、技术选型

### 1.1 主语言：**Python**

| 考量因素 | 说明 |
|----------|------|
| ML 生态 | NumPy、SciPy 等为底层计算提供基础 |
| 可读性 | 便于专注于算法逻辑而非语法细节 |
| 扩展性 | 后期可用 Cython/Rust 加速热点 |
| 社区 | 易于参考 sklearn、PyTorch 等实现 |

### 1.2 依赖策略

- **阶段一**：仅依赖 NumPy（理解向量化与矩阵运算）
- **阶段二**：可引入 SciPy（优化器、稀疏矩阵）
- **阶段三**：可选 JAX/CuPy 支持 GPU 加速

---

## 二、核心模块与类设计

### 模块 1：基础数学与数据结构 `core/`

| 类/模块 | 职责 | 关键方法 |
|---------|------|----------|
| `Vector` | 一维向量封装 | `dot()`, `norm()`, `__add__` |
| `Matrix` | 二维矩阵，支持切片与广播 | `T`, `inv()`, `eig()` |
| `Dataset` | 数据集抽象，支持批量迭代 | `__getitem__`, `__len__`, `shuffle()` |
| `Preprocessor` | 归一化、标准化、编码 | `fit()`, `transform()`, `fit_transform()` |

**学习要点**：理解向量化运算、广播机制、数值稳定性。

---

### 模块 2：线性模型 `linear/`

| 类 | 算法 | 关键属性/方法 |
|----|------|---------------|
| `LinearRegression` | 最小二乘回归 | `coef_`, `intercept_`, `fit()`, `predict()` |
| `Ridge` | L2 正则化回归 | `alpha`, 解析解 `(X'X + αI)^{-1}X'y` |
| `Lasso` | L1 正则化（坐标下降） | `alpha`, `coordinate_descent()` |
| `LogisticRegression` | 二分类逻辑回归 | `sigmoid()`, 梯度下降 / IRLS |

**学习要点**：解析解 vs 迭代优化、正则化、损失函数与梯度。

---

### 模块 3：树模型 `tree/`

| 类 | 算法 | 关键结构/方法 |
|----|------|---------------|
| `TreeNode` | 决策树节点 | `split_feature`, `split_value`, `left`, `right`, `is_leaf` |
| `DecisionTreeRegressor` | CART 回归树 | `_best_split()`, `_mse()`, `fit()`, `predict()` |
| `DecisionTreeClassifier` | CART 分类树 | `_gini()` / `_entropy()`, `_best_split()` |
| `RandomForest` | 随机森林（集成） | `n_estimators`, `bootstrap`, `fit()` |

**学习要点**：递归分裂、信息增益/Gini、过拟合与剪枝。

---

### 模块 4：优化器 `optim/`

| 类 | 算法 | 关键方法 |
|----|------|----------|
| `SGD` | 随机梯度下降 | `step()`, `_update()`, 学习率调度 |
| `Adam` | 自适应学习率 | `m`, `v` 动量, `beta1`, `beta2` |
| `LBFGS` | 拟牛顿（可选） | 利用 SciPy 或自实现 |

**学习要点**：一阶/二阶优化、学习率、收敛条件。

---

### 模块 5：神经网络 `nn/`（进阶）

| 类 | 职责 | 说明 |
|----|------|------|
| `Layer` | 抽象层基类 | `forward()`, `backward()` |
| `Linear` | 全连接层 | `W`, `b`, 矩阵乘法与前向/反向 |
| `ReLU` / `Sigmoid` | 激活函数 | 逐元素运算与梯度 |
| `MLP` | 多层感知机 | `add_layer()`, `fit()`, `predict()` |

**学习要点**：前向传播、反向传播、链式法则、计算图思想。

---

### 模块 6：评估与指标 `metrics/`

| 函数/类 | 用途 |
|---------|------|
| `accuracy(y_true, y_pred)` | 分类准确率 |
| `precision`, `recall`, `f1_score` | 二分类指标 |
| `mean_squared_error`, `mae`, `r2_score` | 回归指标 |
| `confusion_matrix` | 混淆矩阵 |
| `cross_validate(model, X, y, cv=5)` | K 折交叉验证 |

---

### 模块 7：工具与管道 `utils/`

| 类/函数 | 职责 |
|---------|------|
| `train_test_split(X, y, test_size)` | 数据划分 |
| `Pipeline` | 串联 Preprocessor → Model |
| `GridSearchCV` | 超参数网格搜索 |
| `StandardScaler`, `MinMaxScaler` | 常用预处理器 |

---

## 三、推荐目录结构

```
MLspirit/
├── mlspirit/
│   ├── __init__.py
│   ├── core/           # 基础数据结构
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── preprocessor.py
│   ├── linear/         # 线性模型
│   │   ├── __init__.py
│   │   ├── regression.py
│   │   └── logistic.py
│   ├── tree/           # 树模型
│   │   ├── __init__.py
│   │   ├── node.py
│   │   └── forest.py
│   ├── optim/          # 优化器
│   │   ├── __init__.py
│   │   └── sgd.py
│   ├── nn/             # 神经网络（可选）
│   │   ├── __init__.py
│   │   ├── layers.py
│   │   └── mlp.py
│   ├── metrics/        # 评估指标
│   │   ├── __init__.py
│   │   └── scoring.py
│   └── utils/          # 工具
│       ├── __init__.py
│       └── split.py
├── tests/              # 单元测试
├── examples/           # 示例与教程
├── plan.md
├── pyproject.toml      # 或 setup.py
└── README.md
```

---

## 四、实现阶段建议

### Phase 1：基础与线性模型（2–3 周）

1. 实现 `Dataset`、`train_test_split`
2. 实现 `LinearRegression`（解析解）
3. 实现 `Ridge`、`LogisticRegression`
4. 实现 `metrics` 中的回归/分类指标

**检验**：在 UCI 或 sklearn 数据集上复现 sklearn 结果（误差可接受范围内）。

---

### Phase 2：优化与正则化（1–2 周）

1. 实现 `SGD` 优化器
2. 用 SGD 重写 `LinearRegression`、`LogisticRegression`
3. 实现 `Lasso`（坐标下降）
4. 实现 `StandardScaler`、`MinMaxScaler`

**检验**：与 Phase 1 解析解对比，验证梯度下降正确性。

---

### Phase 3：树与集成（2–3 周）

1. 实现 `TreeNode`、`DecisionTreeRegressor`
2. 实现 `DecisionTreeClassifier`（Gini/Entropy）
3. 实现剪枝（预剪枝/后剪枝二选一）
4. 实现 `RandomForest`

**检验**：在分类/回归任务上与 sklearn 树模型对比。

---

### Phase 4：神经网络（2–4 周，可选）

1. 实现 `Linear`、`ReLU`、`Sigmoid` 层
2. 实现 `MLP` 及反向传播
3. 实现 `Adam` 优化器
4. 在 MNIST 子集上测试

---

### Phase 5：工程化（持续）

- 单元测试（pytest）
- 类型注解（mypy）
- 文档（docstring + Sphinx）
- CI/CD（GitHub Actions）
- 发布到 PyPI

---

## 五、学习资源与对照

| 模块 | 建议对照学习 |
|------|--------------|
| 线性模型 | 《统计学习方法》第 1、3、6 章 |
| 树模型 | 《统计学习方法》第 5 章，sklearn 源码 |
| 优化 | 《深度学习》花书第 8 章 |
| 神经网络 | 3Blue1Brown 反向传播视频，CS231n |

---

## 六、成功标准

- [ ] 每个模块有单元测试，覆盖核心逻辑
- [ ] 在至少 3 个标准数据集上达到合理表现
- [ ] 能用自己的话解释每个算法的数学原理
- [ ] 代码结构清晰，便于扩展新模型

---

*最后更新：2025-03-14*
