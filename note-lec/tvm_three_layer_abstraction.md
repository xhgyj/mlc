# TVM 三层抽象：Relax、TE、TIR 优化区别

## 目录

- [1. 架构概览](#1-架构概览)
- [2. Relax 层（高层神经网络抽象）](#2-relax-层高层神经网络抽象)
- [3. TE 层（张量表达式）](#3-te-层张量表达式)
- [4. TIR 层（底层张量程序）](#4-tir-层底层张量程序)
- [5. 三层调优对比](#5-三层调优对比)
- [6. Pipeline 调优为何在 TIR 层](#6-pipeline-调优为何在-tir-层)
- [7. 各层调优的协作关系](#7-各层调优的协作关系)
- [8. 实战示例](#8-实战示例)

---

## 1. 架构概览

### 1.1 三层架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Relax (高层)                              │
│              神经网络计算图 / 用户 API                        │
│                   ↓ LegalizeOps                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                      TE (中层)                               │
│               张量表达式 / 计算图抽象                          │
│                   ↓ (中间表示)                               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                     TIR (底层)                               │
│           显式循环 / 内存管理 / Schedule 优化                 │
│                   ↓ 代码生成                                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
                      机器码
```

### 1.2 层次定位

| 层级 | 抽象级别 | 用户群体 | 主要用途 |
|------|----------|----------|----------|
| **Relax** | 最高 | 深度学习开发者 | 定义神经网络 |
| **TE** | 中等 | 算子开发者 | 描述算子逻辑 |
| **TIR** | 最低 | 性能工程师 | 循环级优化 |

### 1.3 编译流程

```
PyTorch/ONNX 模型
        ↓
    Relax IR (高层算子)
        ↓ LegalizeOps
    Relax IR (call_tir)
        ↓ FuseOps + FuseTIR
    TIR PrimFunc (融合的 kernel)
        ↓ MetaSchedule 调优
    优化后的 TIR
        ↓ 代码生成
    机器码 (LLVM/CUDA)
```

---

## 2. Relax 层（高层神经网络抽象）

### 2.1 定位

- **用途**：高层神经网络 API，类似 PyTorch/TensorFlow
- **用户**：深度学习开发者
- **抽象级别**：最高，操作整个张量和层

### 2.2 典型操作

```python
from tvm.relax.frontend import nn
import tvm.relax as R

class MLP(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 或者使用 Relax 函数式 API
@R.function
def main(x: R.Tensor((1, 784), "float32")) -> R.Tensor((1, 10), "float32"):
    with R.dataflow():
        lv0 = R.nn.relu(x)                      # 高级算子
        lv1 = R.nn.linear(lv0, weight, bias)    # 高级算子
        R.output(lv1)
    return lv1
```

### 2.3 Relax 层优化

#### 图级别优化

| 优化类型 | Pass | 作用 |
|----------|------|------|
| **算子融合** | `FuseOps`, `FuseTIR` | 合并多个操作为一个 kernel |
| **内存优化** | `StaticPlanBlockMemory`, `KillAfterLastUse` | 静态内存规划，提前释放 |
| **布局优化** | `ConvertLayout`, `RewriteDataflowReshape` | NCHW↔NHWC，消除 reshape |
| **精度优化** | `ToMixedPrecision` | FP32→FP16/BF16 |
| **算子替换** | 自定义 Pass | ReLU→GELU，Add+Mul→FMA |
| **死代码消除** | `DeadCodeElimination` | 删除未使用的计算 |

#### 算子融合示例

**融合前**：
```python
@R.function
def main(x):
    with R.dataflow():
        lv0 = R.nn.linear(x, w1, b1)      # kernel 1
        lv1 = R.nn.relu(lv0)              # kernel 2
        lv2 = R.nn.linear(lv1, w2, b2)    # kernel 3
        R.output(lv2)
    return lv2
```

**融合后**：
```python
@R.function
def main(x):
    with R.dataflow():
        # 一个融合 kernel
        lv2 = R.call_tir("fused_linear_relu_linear", ...)
        R.output(lv2)
    return lv2
```

### 2.4 调优特点

- **图级别优化**：关注算子之间的依赖关系和数据流
- **跨算子融合**：将多个 Relax 算子融合为一个 TIR kernel
- **不涉及具体的循环实现**：那是 TIR 层的工作

### 2.5 Dataflow 块

```python
@R.function
def main(x):
    # Dataflow 块：标记纯计算图区域
    with R.dataflow():
        lv0 = R.nn.relu(x)          # 副作用自由
        lv1 = R.nn.linear(lv0, w, b)
        R.output(lv1)

    # 非 Dataflow：可能有副作用
    gv = R.alloc_tensor((128, 128)) # 内存分配副作用
    return gv
```

**作用**：
- 标记计算图区域，便于优化分析
- Dataflow 内的操作保证无副作用，可以安全重排

---

## 3. TE 层（张量表达式）

### 3.1 定位

- **用途**：声明式定义张量计算
- **用户**：算子开发者，编写自定义算子
- **抽象级别**：中等，描述"算什么"而非"怎么算"

### 3.2 典型操作

```python
from tvm import te

# 定义矩阵乘法
def te_matmul(A, B):
    n = A.shape[0]
    m = B.shape[1]
    k = te.reduce_axis((0, A.shape[1]), name="k")

    return te.compute(
        (n, m),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="matmul"
    )

# 定义 ReLU
def te_relu(A):
    return te.compute(
        A.shape,
        lambda *i: te.max(A(*i), 0),
        name="relu"
    )

# 定义卷积
def te_conv2d(data, kernel):
    N, C, H, W = data.shape
    O, I, KH, KW = kernel.shape

    # 计算输出形状
    OH = H - KH + 1
    OW = W - KW + 1

    rc = te.reduce_axis((0, C), name="rc")
    rh = te.reduce_axis((0, KH), name="rh")
    rw = te.reduce_axis((0, KW), name="rw")

    return te.compute(
        (N, O, OH, OW),
        lambda n, o, h, w: te.sum(
            data[n, rc, h + rh, w + rw] * kernel[o, rc, rh, rw],
            axis=[rc, rh, rw]
        ),
        name="conv2d"
    )
```

### 3.3 TE 层优化

| 优化类型 | 作用 | 实现方式 |
|----------|------|----------|
| **表达式简化** | 代数化简，常量传播 | `te.Simplify` |
| **循环融合** | 在表达式层面融合相邻计算 | `te.schedule.fuse` |
| **内存规划** | 分析中间张量的生存期 | 自动推断 |

### 3.4 Schedule 操作

```python
# TE Schedule
def schedule_te(s):
    """为 TE 定义的计算添加调度"""

    # 获取计算 stages
    matmul, relu = s.outputs

    # 循环分割
    _, i, j = s[matmul].op.axis
    k = s[matmul].op.reduce_axis[0]
    ko, ki = s[matmul].split(k, factor=8)

    # 循环重排序
    s[matmul].reorder(i, j, ko, ki)

    # 并行化
    s[matmul].parallel(i)

    return s
```

### 3.5 调优特点

- **中间表示**：主要作为编译器的中间层
- **声明式**：描述计算逻辑，不指定执行细节
- **较少直接调优**：现代 TVM 中 TE 主要用于描述算子，调优由 TIR Schedule 完成

---

## 4. TIR 层（底层张量程序）

### 4.1 定位

- **用途**：显式控制循环和内存，类似 CUDA C
- **用户**：性能工程师，编写高性能 kernel
- **抽象级别**：最低，显式控制每一步

### 4.2 典型操作

```python
import tvm
from tvm.script import tir as T

@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        for i, j, k in T.grid(128, 128, 128):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] += A[vi, vk] * B[vk, vj]

# 或者更复杂的优化版本
@tvm.script.ir_module
class OptimizedMatmul:
    @T.prim_func
    def main(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        # 多级分块
        for io in range(8):
            for jo in range(8):
                for ko in range(8):
                    for ii in range(16):
                        for ji in range(16):
                            with T.block("init"):
                                vi = T.axis.spatial(128, io * 16 + ii)
                                vj = T.axis.spatial(128, jo * 16 + ji)
                                C[vi, vj] = 0.0

                            for ki in range(16):
                                with T.block("update"):
                                    vi = T.axis.spatial(128, io * 16 + ii)
                                    vj = T.axis.spatial(128, jo * 16 + ji)
                                    vk = T.axis.reduce(128, ko * 16 + ki)
                                    C[vi, vj] += A[vi, vk] * B[vk, vj]
```

### 4.3 TIR 层优化（Schedule）

这是 **MetaSchedule 调优的核心层**！

#### 常用 Schedule 原语

| 优化类型 | Schedule 原语 | 作用 | 示例 |
|----------|---------------|------|------|
| **循环分割** | `split(loop, factors=[...])` | 将大循环拆分为多层 | `sch.split(j, factors=[None, 4])` |
| **循环重排** | `reorder(...)` | 改变循环顺序，优化缓存 | `sch.reorder(i, j, k)` |
| **循环融合** | `fuse(...)` | 合并循环，减少开销 | `sch.fuse(i, j)` |
| **向量化** | `vectorize(...)` | SIMD 指令 | `sch.vectorize(inner)` |
| **并行化** | `parallel(...)` | 多线程执行 | `sch.parallel(outer)` |
| **循环展开** | `unroll(...)` | 减少分支开销 | `sch.unroll(inner)` |
| **GPU 线程绑定** | `bind(loop, "threadIdx.x")` | GPU 线程映射 | `sch.bind(tx, "threadIdx.x")` |
| **归约分解** | `decompose_reduction` | 优化归约操作 | `sch.decompose_reduction(block, k)` |

#### TIR 调优示例

**调优前**（朴素实现）：
```python
for i, j, k in T.grid(128, 128, 128):
    with T.block("matmul"):
        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
        with T.init():
            C[vi, vj] = 0.0
        C[vi, vj] += A[vi, vk] * B[vk, vj]
```

**调优后**（MetaSchedule 优化）：
```python
# 循环分块，优化缓存局部性
for io, jo, ko, ii, ji, ki in T.grid(8, 8, 8, 16, 16, 16):
    with T.block("matmul_update"):
        vi = T.axis.spatial(128, io * 16 + ii)
        vj = T.axis.spatial(128, jo * 16 + ji)
        vk = T.axis.reduce(128, ko * 16 + ki)

        T.reads([A[vi, vk], B[vk, vj], C[vi, vj]])
        T.writes([C[vi, vj]])

        C[vi, vj] += A[vi, vk] * B[vk, vj]
```

### 4.4 GPU 编程

```python
@tvm.script.ir_module
class GPUMatmul:
    @T.prim_func
    def main(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        # Grid/Block 配置
        for bx in T.thread_binding(8, thread="blockIdx.x"):
            for by in T.thread_binding(8, thread="blockIdx.y"):
                # 共享内存
                A_shared = T.alloc_buffer((16, 16), "float32", scope="shared")
                B_shared = T.alloc_buffer((16, 16), "float32", scope="shared")

                for tx in T.thread_binding(16, thread="threadIdx.x"):
                    for ty in T.thread_binding(16, thread="threadIdx.y"):
                        # 加载到共享内存
                        A_shared[tx, ty] = A[bx * 16 + tx, ty]
                        B_shared[tx, ty] = B[tx, by * 16 + ty]

                T.sync()  # 同步

                for tx in T.thread_binding(16, thread="threadIdx.x"):
                    for ty in T.thread_binding(16, thread="threadIdx.y"):
                        # 计算
                        for k in range(16):
                            C[bx * 16 + tx, by * 16 + ty] += \
                                A_shared[tx, k] * B_shared[k, ty]
```

---

## 5. 三层调优对比

### 5.1 综合对比表

| 维度 | Relax 层 | TE 层 | TIR 层 |
|------|----------|-------|--------|
| **优化对象** | 算子图 | 计算表达式 | 循环和内存 |
| **典型操作** | 融合、布局转换 | 表达式简化 | Split、Reorder、Vectorize |
| **调优工具** | `transform.FuseOps` | `te.schedule` | `tvm.tir.Schedule` |
| **调优粒度** | 算子级 | 操作级 | 指令级 |
| **MetaSchedule 作用域** | ✗ | △ | ✓（核心） |
| **用户可见性** | 高（主要 API） | 低（中间表示） | 低（需要时手动编写） |
| **主要用户** | DL 开发者 | 算子开发者 | 性能工程师 |
| **代码量级** | 几十行算子 | 单个算子 | 数百行循环 |

### 5.2 优化层次对应

```
┌─────────────────────────────────────────────────────────────┐
│  高层优化 (Relax)                                           │
│  - 算子融合: Conv + BN + ReLU → fused_conv_bn_relu          │
│  - 布局优化: NCHW → NHWC                                    │
│  - 精度优化: FP32 → FP16                                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  中层优化 (TE)                                              │
│  - 表达式简化: x * 1 → x                                    │
│  - 常量折叠: 3 + 5 → 8                                      │
│  - 张量形状推导                                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  底层优化 (TIR)                                             │
│  - 循环分割: for i in range(128) → for io, ii              │
│  - 向量化: 使用 SIMD 指令                                   │
│  - 并行化: 多线程/GPU 线程                                  │
│  - 缓存优化: 分块、重排                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Pipeline 调优为何在 TIR 层？

### 6.1 问题回顾

```python
# static_shape_tuning_pipeline 中
transform.MetaScheduleTuneIRMod(
    params={},
    work_dir=work_dir,
    max_trials_global=total_trials,
)
```

**为什么 Pipeline 调优在 TIR 层，而不是 Relax 或 TE 层？**

### 6.2 原因分析

#### 1. Relax 层已经被降低

```python
# 原始 Relax 代码
@R.function
def main(x):
    with R.dataflow():
        lv0 = R.nn.relu(x)          # 高级算子
        lv1 = R.nn.linear(lv0, w, b)  # 高级算子
        R.output(lv1)
    return lv1

# LegalizeOps 之后
@R.function
def main(x):
    with R.dataflow():
        lv0 = R.call_tir("fused_relax_nn_relu", x)      # 低级调用
        lv1 = R.call_tir("fused_relax_nn_matmul", ...)  # 分解后的操作
        R.output(lv1)
    return lv1
```

经过 `LegalizeOps`，Relax 算子已转换为 `call_tir` 调用。

#### 2. TIR 是实际执行的代码

每个 `call_tir` 对应一个 TIR PrimFunc：

```python
@T.prim_func
def fused_relax_nn_relu(x: T.Buffer((128,), "float32"),
                        out: T.Buffer((128,), "float32")):
    for i in range(128):
        with T.block("relu"):
            vi = T.axis.spatial(128, i)
            out[vi] = T.max(x[vi], 0)  # 这是真正执行的代码
```

**影响性能的关键**（tiling、vectorization）都在 TIR 层。

#### 3. 调优流程

```
Relax 模型
    ↓ LegalizeOps
多个 call_tir 调用（每个对应一个 TIR 函数）
    ↓ MetaScheduleTuneIRMod
对每个 TIR 函数进行 Schedule 搜索
    ↓
最优的 TIR 实现
```

### 6.3 调优示例

假设有一个 MLP 模型：

```python
Relax 模型
├── Linear1 (784 → 256)
├── ReLU
└── Linear2 (256 → 10)
```

**Relax 层优化**：
- 融合 Linear + ReLU
- 结果：2 个 kernel（fused_linear_relu + linear）

**TIR 层调优**：
- 对 fused_linear_relu 的 TIR 实现进行调优
- 对 linear 的 TIR 实现进行调优
- 每个调优都搜索最优的 tiling、vectorization 策略

---

## 7. 各层调优的协作关系

### 7.1 协作图

```
┌─────────────────────────────────────────────────────────────┐
│                    Relax 层调优                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  算子融合：Conv + BN + ReLU → fused_conv_bn_relu    │   │
│  │  布局优化：NCHW → NHWC (CPU)                        │   │
│  │  精度优化：FP32 → FP16                               │   │
│  │  死代码消除：删除未使用的计算                        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          ↓ LegalizeOps
┌─────────────────────────────────────────────────────────────┐
│                    TIR 层调优 (MetaSchedule)                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  对每个 TIR 函数：                                    │   │
│  │  - 搜索最优的 split 因子                             │   │
│  │  - 搜索最优的 reorder 顺序                           │   │
│  │  - 搜索最优的 vectorize 长度                         │   │
│  │  - 搜索最优的 tiling 策略                            │   │
│  │  - 搜索最优的并行化配置                              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          ↓ 代码生成
┌─────────────────────────────────────────────────────────────┐
│                    机器码                                    │
│  LLVM IR / PTX / CUDA C                                    │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 端到端优化示例

```python
# 输入：PyTorch 模型
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 3)
        self.bn = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# 1. 转换为 Relax
mod = from_pytorch(Model())

# 2. Relax 层优化
mod = tvm.relax.transform.FuseOps()(mod)      # Conv+BN+ReLU 融合
mod = tvm.relax.transform.ToMixedPrecision()(mod)  # FP16

# 3. Legalize
mod = tvm.relax.transform.LegalizeOps()(mod)   # 转换为 call_tir

# 4. TIR 层调优
mod = transform.MetaScheduleTuneIRMod(
    max_trials_global=1000,
    work_dir="./tune_logs"
)(mod)

# 5. 编译
ex = tvm.compile(mod, target="llvm")
```

### 7.3 性能收益分解

| 优化层 | 优化技术 | 性能提升 |
|--------|----------|----------|
| **Relax** | 算子融合 | 2-5x |
| **Relax** | 精度优化 (FP16) | 1.5-2x |
| **TIR** | 循环优化 | 1.5-3x |
| **TIR** | 向量化 | 2-8x |
| **TIR** | 并行化 | 接近线性 |
| **总体** | 综合优化 | 10-100x |

---

## 8. 实战示例

### 8.1 完整的端到端示例

```python
import tvm
from tvm import relax
from tvm.script import relax as R, tir as T

# ========== 1. 定义 Relax 模型 ==========
@tvm.script.ir_module
class MyModel:
    @R.function
    def main(x: R.Tensor((1, 784), "float32")) -> R.Tensor((1, 10), "float32"):
        with R.dataflow():
            lv0 = R.nn.linear(x, w0, b0)  # Linear1
            lv1 = R.nn.relu(lv0)          # ReLU
            lv2 = R.nn.linear(lv1, w1, b1)  # Linear2
            R.output(lv2)
        return lv2

    # TIR 实现（简化）
    @T.prim_func
    def relu(x: T.Buffer((128,), "float32"),
             out: T.Buffer((128,), "float32")):
        for i in range(128):
            out[i] = T.max(x[i], 0)

    @T.prim_func
    def linear(x: T.Buffer((1, 784), "float32"),
               w: T.Buffer((128, 784), "float32"),
               b: T.Buffer((128,), "float32"),
               out: T.Buffer((1, 128), "float32")):
        for i, j in T.grid(1, 128):
            with T.block("init"):
                vi, vj = T.axis.remap("SS", [i, j])
                out[vi, vj] = b[vj]

            for k in T.grid(784):
                with T.block("update"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    out[vi, vj] += x[vi, vk] * w[vj, vk]

# ========== 2. Relax 层优化 ==========
mod = tvm.relax.transform.FuseOps()(MyModel)
# 结果：Linear1 + ReLU 融合为一个 TIR 函数

# ========== 3. Legalize ==========
mod = tvm.relax.transform.LegalizeOps()(mod)

# ========== 4. TIR 层调优 ==========
from tvm import meta_schedule as ms

database = ms.tune_tir(
    mod=mod,
    target="llvm --num-cores=4",
    max_trials_global=512,
    work_dir="./tune_logs"
)

# ========== 5. 应用最佳配置 ==========
mod = database.apply_best(mod)

# ========== 6. 编译 ==========
ex = relax.build(mod, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

# ========== 7. 运行 ==========
data_nd = tvm.nd.array(np.random.randn(1, 784).astype("float32"))
result = vm["main"](data_nd)
```

### 8.2 自定义 TIR kernel

```python
# 当内置算子不够用时，编写自定义 TIR

@tvm.script.ir_module
class CustomKernel:
    @T.prim_func
    def swish(x: T.Buffer((n,), "float32"),
              out: T.Buffer((n,), "float32")):
        """Swish 激活函数：x * sigmoid(x)"""
        for i in T.grid(n):
            with T.block("swish"):
                vi = T.axis.spatial(n, i)
                sigmoid_x = 1 / (1 + T.exp(-x[vi]))
                out[vi] = x[vi] * sigmoid_x

# 在 Relax 中使用
@R.function
def main(x: R.Tensor((n,), "float32")):
    with R.dataflow():
        lv = R.call_tir(CustomKernel["swish"], x)
        R.output(lv)
    return lv
```

---

## 9. 关键要点总结

### 9.1 各层职责

| 层级 | 核心职责 | 典型用户 |
|------|----------|----------|
| **Relax** | 图级别优化，算子融合 | DL 开发者 |
| **TE** | 算子逻辑描述 | 算子开发者 |
| **TIR** | 循环级性能优化 | 性能工程师 |

### 9.2 调优协作

```
Pipeline 调优 = Relax 层融合 + TIR 层 Schedule 搜索
```

1. **Relax**：决定"哪些算子应该融合在一起"
2. **TIR**：决定"每个 kernel 具体怎么执行"

### 9.3 何时使用哪一层

| 场景 | 使用层级 |
|------|----------|
| 快速原型开发 | Relax |
| 集成现有模型 | Relax (from_pytorch/onnx) |
| 编写新算子 | TE / TIR |
| 性能关键路径 | TIR + MetaSchedule |
| GPU kernel 开发 | TIR (手动编写) |

---

## 参考资料

- Relax 文档: `/home/lhy/mlc/relax_transformation_guide.md`
- TensorIR 文档: `/home/lhy/mlc/note-lec/3_TensorIR_Tensor_Program_Abstraction_Case_Study_Action_summary.md`
- MetaSchedule 文档: `/home/lhy/mlc/note-lec/metaschedule_search_principles.md`
- 课程笔记: `/home/lhy/mlc/note-lec/5_Automatic_Program_Optimization.ipynb`
- `/home/lhy/mlc/note-lec/6_Integration_with_Machine_Learning_Frameworks.ipynb`
