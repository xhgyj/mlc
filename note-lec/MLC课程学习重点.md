# MLC 课程学习重点

## 📚 课程概览

MLC（Machine Learning Compilation）课程学习 TVM 框架，核心是理解**张量程序的抽象与变换**。

---

# L3: TensorIR 张量程序抽象

## 核心概念

### TensorIR 三要素
| 概念 | 说明 | 语法 |
|------|------|------|
| **Block** | 计算单元，表示一段循环体 | `with T.block("name"):` |
| **Axis** | 循环轴类型 | `T.axis.spatial()` / `T.axis.reduce()` |
| **Buffer** | 内存访问注解 | `T.reads()` / `T.writes()` |

### IRModule 结构
```python
@tvm.script.ir_module
class Module:
    @T.prim_func
    def func(...):
        # 低级张量函数
```

## Schedule 变换

### 常用操作
| 操作 | 作用 | 示例 |
|------|------|------|
| `split` | 分割循环 | `sch.split(loop, factors=[None, 4])` |
| `reorder` | 重排序循环 | `sch.reorder(j0, i, j1)` |
| `reverse_compute_at` | 融合计算 | `sch.reverse_compute_at(block, loop)` |
| `decompose_reduction` | 分离归约 | `sch.decompose_reduction(block)` |

### 编译流程
```
IRModule → Schedule → tvm.build → Runtime Module → 执行
```

## 🔍 Block 深入理解

### Block 的本质

**Block 是 TensorIR 中最基本的计算单元**，它封装了"一段循环体的核心计算逻辑"。

```
一个 block = 一个独立可优化的计算任务
```

### 为什么需要 Block？

#### 1. 显式化计算边界

没有 block 时，循环和计算混在一起；有了 block，计算边界清晰：

```python
# 传统写法 - 计算分散在循环中
for i in range(128):
    for j in range(128):
        for k in range(128):
            Y[i,j] += A[i,k] * B[k,j]  # 计算嵌在循环里

# TensorIR - 边界清晰
for i, j, k in ...:
    with T.block("Y"):  # ← 明确标记：这段循环在做什么
        # block 内部：纯粹的元素级计算逻辑
```

#### 2. 分离循环结构与计算逻辑

Block 让 TVM 能够区分：
- **外层循环** → 如何遍历（调度优化的重点）
- **Block 内部** → 计算什么（语义保证）

```python
with T.block("Y"):
    vi = T.axis.spatial(128, i)  # 声明：这是空间轴
    vj = T.axis.spatial(128, j)
    vk = T.axis.reduce(128, k)    # 声明：这是归约轴
    with T.init():
        Y[vi, vj] = T.float32(0)  # 初始化
    Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]  # 更新
```

#### 3. 语义注解让变换更安全

| 轴类型 | 含义 | Schedule 时可做的变换 |
|--------|------|----------------------|
| `spatial` | 每个输出元素独立计算 | 可以随意 split、reorder |
| `reduce` | 需要累加的维度 | split 时要小心 init/update |

```python
# TVM 知道 k 是 reduce 轴，所以 split 时会自动处理初始化
sch.split(k, factors=[32, 4])  # ✅ TVM 确保正确性
```

### Block 的结构剖析

```python
with T.block("Y"):           # 1. Block 名称（标识符）
    # 2. 轴绑定 - 建立循环变量与逻辑索引的映射
    vi = T.axis.spatial(128, i)  # i=外层循环变量, vi=逻辑位置
    vj = T.axis.spatial(128, j)
    vk = T.axis.reduce(128, k)

    # 3. 初始化块（归约操作必需）
    with T.init():
        Y[vi, vj] = T.float32(0)

    # 4. 实际计算 - 使用逻辑索引，不直接用循环变量
    Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
```

**结构可视化：**

```
┌─────────────────────────────────────────┐
│  外层循环                               │
│  ┌─────────────────────────────────┐   │
│  │  Block "Y"                      │   │
│  │  ┌───────────────────────────┐  │   │
│  │  │ 轴绑定 (axis bindings)   │  │   │
│  │  │ vi = T.axis.spatial(i)    │  │   │
│  │  │ vj = T.axis.spatial(j)    │  │   │
│  │  │ vk = T.axis.reduce(k)     │  │   │
│  │  └───────────────────────────┘  │   │
│  │  ┌───────────────────────────┐  │   │
│  │  │ init 块                   │  │   │
│  │  │ Y[vi, vj] = 0             │  │   │
│  │  └───────────────────────────┘  │   │
│  │  ┌───────────────────────────┐  │   │
│  │  │ 更新计算                   │  │   │
│  │  │ Y[vi,vj] += A[vi,vk]*B... │  │   │
│  │  └───────────────────────────┘  │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### Block 的关键价值：Schedule 能力

因为有了 block 的结构化表示，TVM 可以进行各种变换：

```python
sch = tvm.tir.Schedule(MyModule)

# 1. 获取 block
block = sch.get_block("Y")

# 2. 操作循环 - 不破坏 block 内部的计算逻辑
i, j, k = sch.get_loops(block)
sch.split(k, factors=[32, 4])      # 归约轴分块
sch.reorder(i, j, k.outer, k.inner)  # 重排序

# 3. GPU 映射
sch.gpu_bind(i, "blockIdx.x")     # 映射到 GPU block
sch.gpu_bind(j, "threadIdx.x")    # 映射到 GPU thread
```

### Block 解决的问题总结

| 问题 | 没有-block | 有-block |
|------|-----------|----------|
| 计算边界 | 隐式、分散 | 显式、封装 |
| 循环与计算 | 耦合 | 分离 |
| 轴的类型 | 推测 | 显式声明 |
| 优化变换 | 不安全、易出错 | 结构化、可验证 |

> **一句话总结**：Block 是 TensorIR 让"张量计算可编程优化"的核心抽象 —— 它把计算逻辑从循环结构中剥离出来，让编译器可以安全地重排、分块、并行化，而不破坏计算的正确性。

## 🔧 Block Axis 属性详解

### Axis 声明语法

```python
[block_axis] = T.axis.[axis_type]([axis_range], [mapped_value])
```

**示例：**
```python
vi = T.axis.spatial(128, i)
vj = T.axis.spatial(128, j)
vk = T.axis.reduce(128, k)
```

### Axis 声明包含的三类信息

| 信息 | 说明 | 示例中的值 |
|------|------|-----------|
| **绑定关系** | block_axis 绑定到哪个循环变量 | `vi` → `i` |
| **值域范围** | axis 的有效范围 | `128` 表示 `range(0, 128)` |
| **轴属性** | spatial 或 reduce | `spatial` / `reduce` |

### 两种轴属性对比

| 属性 | 含义 | 计算特性 | 可并行性 | 典型操作 |
|------|------|----------|----------|----------|
| **`spatial`** | 空间轴 | 每个输出元素独立计算 | ✅ 可并行 | split, reorder, fuse |
| **`reduce`** | 归约轴 | 需要跨维度累加 | ⚠️ 需特殊策略 | 归约优化（树形归约等） |

### 直观理解

```
矩阵乘法 C = A × B:
       A (128×128)    B (128×128)    C (128×128)
        ┌────┬────┐    ┌────┬────┐    ┌────┬────┐
        │    │    │    │    │    │    │    │    │
    i ─→│    │    │ × k→│    │    │ →  │    │    │← j
        │    │    │    │    │    │    │    │    │
        └────┴────┘    └────┴────┘    └────┴────┘

vi = T.axis.spatial(128, i)  # i 是空间轴：对应 C 的行
vj = T.axis.spatial(128, j)  # j 是空间轴：对应 C 的列
vk = T.axis.reduce(128, k)   # k 是归约轴：需要累加中间结果
```

### 为什么需要这些额外信息？

#### 1. **Block 自包含性**

Block 的 axis 信息使其**独立于外层循环**，成为可重用的计算单元：

```python
# Block 内部声明了自己的迭代需求
with T.block("Y"):
    vi = T.axis.spatial(128, i)  # 声明：我需要 128 次迭代
    ...
```

即使外层循环变化，Block 的语义不变。

#### 2. **正确性验证**

TVM 可以检查循环与 block 是否匹配：

```python
# ❌ 错误：循环范围 (127) 与 axis 范围 (128) 不匹配
for i in range(127):
    with T.block("C"):
        vi = T.axis.spatial(128, i)  # 编译错误！
```

#### 3. **指导并行化策略**

```python
# Spatial 轴 → 直接并行
sch.parallel(vi)  # ✅ 安全

# Reduce 轴 → 需要归约策略
sch.parallel(vk)  # ⚠️ 需要特殊处理
```

### 语法糖：`T.axis.remap`

当每个 axis 直接对应外层循环时，可以用简化语法：

```python
# SSR = "Spatial", "Spatial", "Reduce"
vi, vj, vk = T.axis.remap("SSR", [i, j, k])
```

**等价于：**
```python
vi = T.axis.spatial(128, i)
vj = T.axis.spatial(128, j)
vk = T.axis.reduce(128, k)
```

### 常用映射字符串

| 字符串 | 含义 | 示例 |
|--------|------|------|
| `"S"` | Spatial | 单个空间轴 |
| `"R"` | Reduce | 单个归约轴 |
| `"SS"` | 两个 Spatial | `vi, vj = T.axis.remap("SS", [i, j])` |
| `"SSR"` | Spatial, Spatial, Reduce | 矩阵乘法典型模式 |
| `"SRS"` | Spatial, Reduce, Spatial | 其他归约模式 |

### 完整示例对比

```python
# 标准写法
with T.block("Y"):
    vi = T.axis.spatial(128, i)
    vj = T.axis.spatial(128, j)
    vk = T.axis.reduce(128, k)
    with T.init():
        Y[vi, vj] = T.float32(0)
    Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]

# 简化写法
with T.block("Y"):
    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
    with T.init():
        Y[vi, vj] = T.float32(0)
    Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
```

### 核心要点

> **Block Axis 属性 = 迭代器的"类型标签"**
>
> - 告诉 TVM 这个轴在计算中的角色
> - 让编译器知道哪些变换是安全的
> - 实现计算逻辑与循环结构的解耦

---

# L4: 端到端模型构建

## 两层抽象

| 抽象层次 | 作用 | 装饰器 |
|----------|------|--------|
| **TensorIR** | 低级张量函数实现 | `@T.prim_func` |
| **Relax** | 高级计算图/神经网络 | `@R.function` |

## 关键构造

### 1. `call_tir` - 桥接高低层
```python
lv0 = R.call_tir(linear0, (x, w0, b0), out_sinfo=R.Tensor((1, 128), "float32"))
```
- 调用 `prim_func` 实现底层计算
- 使用**目标传递风格** (DPS)
- 显式指定输出类型 `out_sinfo`

### 2. Dataflow 块
```python
with R.dataflow():
    lv0 = R.call_tir(...)
    lv1 = R.call_tir(...)
    out = R.call_tir(...)
    R.output(out)
```
- 标记计算图作用域(可以表示哪些可以做计算图的优化，哪些地方有坑)
- 便于编译器优化分析
### Dataflow Block

Another important element in a relax function is the `R.dataflow()` scope annotation.

```python
with R.dataflow():
    lv0 = R.call_tir(cls.linear0, (x, w0, b0), out_sinfo=R.Tensor((1, 128), dtype="float32"))
    lv1 = R.call_tir(cls.relu0, (lv0,), out_sinfo=R.Tensor((1, 128), dtype="float32"))
    out = R.call_tir(cls.linear1, (lv1, w1, b1), out_sinfo=R.Tensor((1, 10), dtype="float32"))
    R.output(out)
```

This connects back to the **computational graph** discussion we had in the last section. Recall that ideally, each computational graph operation should be side effect free.

What if we still want to introduce operations that contains side effect? A dataflow block is a way for us to mark the computational graph regions of the program. Specifically, within a dataflow block, all the operations need to be side-effect free. Outside a dataflow block, the operations can contain side-effect. The program below is an example program that contains two dataflow blocks.

```python
@R.function
def main(x: Tensor((1, 784), "float32"),
         w0: Tensor((128, 784), "float32"),
         b0: Tensor((128,), "float32"),
         w1: Tensor((10, 128), "float32"),
         b1: Tensor((10,), "float32")):


    // 表示可以用于计算图优化
    with R.dataflow():
        lv0 = R.call_tir(cls.linear0, (x, w0, b0), out_sinfo=R.Tensor((1, 128), dtype="float32"))
        gv0 = R.call_tir(cls.relu0, (lv0,), out_sinfo=R.Tensor((1, 128), dtype="float32"))
        R.output(gv0)
    // 不可以
    gv1 = R.alloc_tensor((1, 128), dtype="float32")

    with R.dataflow():
        out = R.call_tir(cls.linear1, (gv0, gv1, b0), out_sinfo=R.Tensor((1, 128), dtype="float32"))
        R.output(out)
    return out
```

Most of our lectures will only deal with computational graphs (dataflow blocks). But it is good to keep the reason behind in mind.
### 3. 计算图视图
```
Input → Linear → ReLU → Linear → Output
```
- 每个节点 = 一个计算操作
- 每条边 = 张量数据流

## 三种实现方式

| 方式 | 特点 | 适用场景 |
|------|------|----------|
| **纯 TensorIR** | 所有操作用 TVM 实现 | 自定义算子、性能调优 |
| **外部库调用** | `call_dps_packed` 调用 PyTorch 等 | 复用已有实现 |
| **混合模式** | TensorIR + 外部库 | 灵活组合 |

### 外部库调用示例
```python
# 注册运行时函数
@tvm.register_func("env.linear", override=True)
def torch_linear(x, w, b, out):
    x_t = torch.from_dlpack(x)
    # ... 使用 PyTorch 实现
```

## 参数绑定
```python
MyModule = relax.transform.BindParams("main", nd_params)(MyModule)
```
- 将权重绑定为常量
- 减少运行时参数传递

## 编译运行流程
```python
# 1. 构建
ex = relax.build(MyModule, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

# 2. 准备数据
data_nd = tvm.nd.array(data_np)

# 3. 执行
res = vm["main"](data_nd)
res_np = res.numpy()
```

---

## 🔑 核心思想

### 抽象与实现分离
```
高级抽象 (Relax) → 中间表示 (TensorIR) → 底层实现 (LLVM/CUDA)
```

### 目标传递风格 (DPS)
- 调用者预先分配输出缓冲区
- 函数直接写入结果，避免额外拷贝

### DLPack 零拷贝
- 不同框架间交换张量无需拷贝
- `torch.from_dlpack()` / `tvm.nd.array()`

---

## 📝 学习检查清单

- [ ] 理解 TensorIR 的 Block/Axis/Buffer 概念
- [ ] 掌握基本 Schedule 操作 (split/reorder/...)
- [ ] 理解 Relax 函数与计算图的关系
- [ ] 掌握 `call_tir` 和 Dataflow 块的用法
- [ ] 了解如何集成外部库函数
- [ ] 能够构建并运行端到端模型

---

## 🔗 相关资源

### 官方资源
- TVM 官方文档: https://tvm.apache.org/
- MLC 课程: https://mlc.ai/
- 安装命令: `pip install mlc-ai-nightly -f https://mlc.ai/wheels`

### 本项目文档
- [MetaSchedule 搜索原理分析](./metaschedule_search_principles.md) - 深入解释 TVM 自动调优如何搜索最优实现
- [TVM 三层抽象详解](./tvm_three_layer_abstraction.md) - Relax、TE、TIR 三层的区别和优化作用
- [Relax 变换指南](../relax_transformation_guide.md) - 内置变换和自定义 Pass 完整指南
- [TensorIR 学习笔记](./3_TensorIR_Tensor_Program_Abstraction_Case_Study_Action_summary.md) - TensorIR 核心概念总结
