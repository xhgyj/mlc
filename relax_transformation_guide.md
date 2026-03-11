# Relax 变换指南

## 目录

- [1. 变换概述](#1-变换概述)
- [2. 内置变换](#2-内置变换)
  - [2.1 LegalizeOps](#21-legalizeops)
  - [2.2 融合优化流水线](#22-融合优化流水线)
- [3. 自定义 Pass](#3-自定义-pass)
  - [3.1 Mutator 层](#31-mutator-层)
  - [3.2 Pass 层](#32-pass-层)
  - [3.3 应用 Pass](#33-应用-pass)
- [4. TVM 编译流程](#4-tvm-编译流程)
- [5. 完整示例](#5-完整示例)

---

## 1. 变换概述

在 TVM/Relax 中，**变换（Transformation）** 是优化程序的核心机制。通过一系列 Pass，可以将高层神经网络定义转换为高效可执行的机器码。

### 为什么要变换？

| 层级 | 表示形式 | 目的 |
|------|----------|------|
| 高层 | `relax.op.nn.relu` | 用户友好，语义清晰 |
| 中层 | `call_tir("fused_relax_nn_relu")` | 编译器可分析，可优化 |
| 低层 | TIR PrimFunc | 可生成机器码 |

### 变换的类型

- **降低变换**：高级算子 → 低级算子（如 LegalizeOps）
- **优化变换**：融合、常量折叠、死代码消除
- **目标相关变换**：针对特定硬件的后端优化

---

## 2. 内置变换

### 2.1 LegalizeOps

**作用**：将高级算子降低为低级 `call_tir` 调用。

```python
mod = tvm.relax.transform.LegalizeOps()(origin_mod)
```

#### 变换前后对比

**变换前**（高级算子）：
```python
@R.function
def main(x: R.Tensor((784,))) -> R.Tensor((10,)):
    with R.dataflow():
        lv1 = R.nn.relu(x)                    # 高级算子
        lv2 = R.nn.linear(lv1, weight, bias)  # 高级算子（封装）
        R.output(lv2)
    return lv2
```

**变换后**（低级算子）：
```python
@R.function
def main(x: R.Tensor((784,))) -> R.Tensor((10,)):
    with R.dataflow():
        lv1 = R.call_tir("fused_relax_nn_relu", x)      # 低级调用
        lv2 = R.call_tir("fused_relax_nn_matmul", ...)  # 分解后的操作
        lv3 = R.call_tir("fused_relax_nn_add", ...)
        R.output(lv3)
    return lv3
```

#### 为什么需要 Legalize？

1. **高级算子没有 TIR 实现**：只是抽象符号，缺乏计算细节
2. **后续优化需要分析计算模式**：必须先有 TIR 代码才能分析
3. **统一接口**：所有算子最终都通过 `call_tir` 调用

---

### 2.2 融合优化流水线

融合是 ML 编译器中最重要的优化技术之一，通过将多个操作合并为一个 kernel 来减少内存访问和 kernel 启动开销。

```python
mod = tvm.ir.transform.Sequential(
    [
        tvm.relax.transform.AnnotateTIROpPattern(),
        tvm.relax.transform.FuseOps(),
        tvm.relax.transform.FuseTIR(),
    ]
)(mod)
```

#### 三个步骤详解

| 步骤 | 作用 | 输入 → 输出 |
|------|------|-------------|
| **AnnotateTIROpPattern** | 标注每个 TIR 操作的计算模式 | TIR 函数 → 模式注解（element-wise/reduction 等） |
| **FuseOps** | 在图级别识别可融合的操作序列 | 带模式注解的图 → 融合分组 |
| **FuseTIR** | 为每个融合组生成融合后的 TIR 函数 | 融合分组 → `fused_*` TIR 函数 |

#### 融合效果示例

**融合前**（三个独立的 kernel）：
```
[x] ──► MatMul ──► [temp1] ──► Add ──► [temp2] ──► ReLU ──► [y]
       kernel1        (写内存)       kernel2        (写内存)       kernel3
```

**融合后**（一个融合 kernel）：
```
[x] ──► [MatMul + Add + ReLU] ──► [y]
       fused_kernel (所有中间结果在寄存器/共享内存中)
```

#### 性能收益

| 指标 | 融合前 | 融合后 |
|------|--------|--------|
| Kernel 启动次数 | 3 次 | 1 次 |
| 全局内存访问 | 5 次（读+写中间结果） | 2 次（读输入，写输出） |
| 带宽利用率 | 低（频繁读写） | 高（复用中间结果） |

#### 计算模式

融合决策基于算子的计算模式：

| 模式 | 特征 | 示例 |
|------|------|------|
| **Element-wise** | 每个输出元素独立计算 | ReLU, Add, Mul |
| **Broadcast** | 支持形状广播 | Add, Sub |
| **Reduction** | 需要归约操作 | Sum, Max, Mean |
| **Injective** | 可逆映射 | Reshape, Transpose |

**融合规则**（简化）：
- Element-wise + Element-wise → 可融合
- Element-wise + Reduction → 谨慎融合
- Reduction + Reduction → 不可直接融合

---

## 3. 自定义 Pass

TVM 提供了两层架构来实现自定义变换。

### 3.1 Mutator 层

**Mutator** 负责表达式树的遍历和节点替换。

```python
from tvm.relax.expr_functor import PyExprMutator, mutator

@mutator
class ReluRewriter(PyExprMutator):
    def __init__(self, mod: IRModule):
        super().__init__(mod)

    def visit_call_(self, call: relax.Call) -> relax.Expr:
        """当遍历到 Call 节点时调用"""
        # 只处理 relu 算子
        if call.op.name == "relax.nn.relu":
            # 替换为 gelu，保持输入参数不变
            return relax.op.nn.gelu(call.args[0])

        # 其他情况递归处理子节点
        return super().visit_call_(call)
```

#### Visitor 模式

`PyExprMutator` 使用 Visitor 模式遍历表达式树：

```python
        Call
       /    \
    op    args[0]
           /   \
       Call   Tensor
       /   \
    op   args
```

| 钩子方法 | 触发时机 |
|----------|----------|
| `visit_call_()` | 遍历到 `relax.Call` 节点 |
| `visit_tuple_()` | 遍历到 `relax.Tuple` 节点 |
| `visit_tuple_getitem_()` | 遍历到 `relax.TupleGetItem` 节点 |
| `visit_dataflow_var_()` | 遍历到 `relax.DataflowVar` 节点 |

#### Mutator 工作流程

```
1. visit_expr(func) 开始遍历
       │
2. 递归遍历表达式树（自动调用 visit_*）
       │
3. 在 visit_call_ 中检测并替换节点
       │
4. builder_ 记录所有修改
       │
5. builder_.get() 返回新模块
```

---

### 3.2 Pass 层

**Pass** 负责模块级别的协调和管理。

```python
@tvm.transform.module_pass(opt_level=0, name="ReluToGelu")
class ReluToGelu:
    def transform_module(
        self,
        mod: IRModule,
        _ctx: tvm.transform.PassContext
    ) -> IRModule:
        """模块级变换入口"""
        rewriter = ReluRewriter(mod)                      # 创建 mutator

        # 遍历模块中的所有全局函数
        for g_var, func in mod.functions_items():
            # 只处理 Relax 函数（跳过 TIR PrimFunc）
            if isinstance(func, relax.Function):
                # 应用变换，返回新函数
                new_func = rewriter.visit_expr(func)
                # 更新模块中的函数
                rewriter.builder_.update_func(g_var, new_func)

        # 返回变换后的完整模块
        return rewriter.builder_.get()
```

#### Pass 装饰器参数

| 参数 | 说明 |
|------|------|
| `opt_level` | 优化级别（0-3），用于 Pass 管线调度 |
| `name` | Pass 名称，用于调试和日志 |

#### Pass 与 Mutator 的关系

```
┌─────────────────────────────────────────┐
│  Pass (ReluToGelu)                       │
│                                         │
│  职责：                                  │
│  - 遍历所有函数                         │
│  - 管理变换上下文                       │
│  - 可组合到 Pass 管线中                 │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  Mutator (ReluRewriter)          │   │
│  │                                 │   │
│  │  职责：                          │   │
│  │  - 遍历表达式树                 │   │
│  │  - 检测和替换节点               │   │
│  │  - 递归处理子节点               │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

**为什么要两层？**

- **Pass 可组合**：多个 Pass 可以串成流水线
- **Mutator 可复用**：同一个 Mutator 可在不同 Pass 中使用
- **关注点分离**：Pass 管协调，Mutator 管变换

---

### 3.3 应用 Pass

```python
# 方式一：直接应用
mod = ReluToGelu()(origin_mod)

# 方式二：加入 Sequential 流水线
mod = tvm.ir.transform.Sequential([
    tvm.relax.transform.LegalizeOps(),
    ReluToGelu(),  # 自定义 Pass
    tvm.relax.transform.FuseOps(),
])(mod)

# 方式三：使用 PassContext 配置
with tvm.transform.PassContext(opt_level=3):
    mod = ReluToGelu()(mod)
```

---

## 4. TVM 编译流程

完整的 Relax 编译流程展示了各个变换如何协同工作。

```
┌─────────────────────────────────────────────────────────────┐
│  1. 用户代码 (PyTorch/nn.Module)                            │
└─────────────────────┬───────────────────────────────────────┘
                      │ export_tvm()
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 高层 Relax IR (relax.op.*)                              │
│     - relax.nn.relu                                         │
│     - relax.nn.linear (封装)                                │
└─────────────────────┬───────────────────────────────────────┘
                      │ LegalizeOps
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  3. 低层 Relax IR (call_tir)                                │
│     - call_tir("fused_relax_nn_relu")                       │
│     - call_tir("fused_relax_nn_matmul")                     │
└─────────────────────┬───────────────────────────────────────┘
                      │ AnnotateTIROpPattern
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 带模式注解的 IR                                         │
│     - 每个算子标注计算模式                                   │
└─────────────────────┬───────────────────────────────────────┘
                      │ FuseOps + FuseTIR
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  5. 融合后的 IR                                             │
│     - call_tir("fused_matmul_add_relu")                     │
└─────────────────────┬───────────────────────────────────────┘
                      │ 后端变换 (如 CUDA)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  6. 机器码 (PTX/CUDA/SSE/AVX)                               │
└─────────────────────────────────────────────────────────────┘
```

### 为什么需要先 Legalize 再融合？

```
AnnotateTIROpPattern 需要分析 TIR 代码：
- 扫描循环结构
- 识别 reduction axis vs spatial axis
- 判断内存访问模式

只有 Legalize 后才有 TIR 代码可供分析！
```

---

## 5. 完整示例

以下是一个完整的自定义变换示例，演示如何将 `add + mul` 替换为 `fma`（融合乘加）。

```python
from tvm import IRModule, relax
from tvm.relax.expr_functor import PyExprMutator, mutator
import tvm

# ========== 1. 定义 Mutator ==========
@mutator
class AddMulToFmaMutator(PyExprMutator):
    """检测 add(mul(a, b), c) 模式并替换为 fma(a, b, c)"""

    def visit_call_(self, call: relax.Call) -> relax.Expr:
        # 递归处理子节点（确保先处理嵌套调用）
        call = super().visit_call_(call)

        # 检测是否是 add 操作
        if call.op.name != "relax.add":
            return call

        # add 的第一个参数是否是 mul？
        arg0 = call.args[0]
        if isinstance(arg0, relax.Call) and arg0.op.name == "relax.multiply":
            # 检测到模式：add(mul(a, b), c)
            a = arg0.args[0]
            b = arg0.args[1]
            c = call.args[1]
            return relax.op.call_dps_packed("fma", [a, b, c], call.scaled_args[2].struct_info)

        return call


# ========== 2. 定义 Pass ==========
@tvm.transform.module_pass(opt_level=2, name="AddMulToFma")
class AddMulToFma:
    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        mutator = AddMulToFmaMutator(mod)

        for g_var, func in mod.functions_items():
            if isinstance(func, relax.Function):
                new_func = mutator.visit_expr(func)
                mutator.builder_.update_func(g_var, new_func)

        return mutator.builder_.get()


# ========== 3. 应用 Pass ==========
# 假设有一个模块 mod
optimized_mod = AddMulToFma()(mod)
```

### 常见自定义变换场景

| 场景 | 技术要点 |
|------|----------|
| 算子替换 | 检测 `call.op.name`，返回新的 Call |
| 常量折叠 | 在 `visit_call_` 中计算常量表达式 |
| 死代码消除 | 使用 `DataflowBlock` 分析依赖关系 |
| 布局转换 | 在 `call_tir` 前后插入 transpose |
| 量化替换 | float32 算子 → int8 算子 + dequant |

---

## 参考资料

- [TVM Relax 文档](https://tvm.apache.org/docs/reference/api/python/relax.html)
- [TVM 变换 Pass 列表](https://tvm.apache.org/docs/reference/api/python/relax.transform.html)
- [TensorIR 介绍](https://tvm.apache.org/docs/arch/tir.html)
