# TVM Relax Transform API 完全指南

> 本文档基于 TVM 0.24.dev0 版本官方文档整理

## 目录

- [1. API 分类概览](#1-api-分类概览)
- [2. 降低类变换 (Lowering)](#2-降低类变换-lowering)
- [3. 融合优化类 (Fusion)](#3-融合优化类-fusion)
- [4. 内存优化类 (Memory)](#4-内存优化类-memory)
- [5. 布局优化类 (Layout)](#5-布局优化类-layout)
- [6. 死代码消除类 (DCE)](#6-死代码消除类-dce)
- [7. 精度优化类 (Precision)](#7-精度优化类-precision)
- [8. 数据流转换类 (Dataflow)](#8-数据流转换类-dataflow)
- [9. 参数管理类 (Params)](#9-参数管理类-params)
- [10. 图优化典型流程](#10-图优化典型流程)
- [11. Pass 顺序指南](#11-pass-顺序指南)
- [12. 调试技巧](#12-调试技巧)

---

## 1. API 分类概览

```
tvm.relax.transform
│
├── 【降低类】将高级算子转为低级算子
│   ├── LegalizeOps                # 主要降低 Pass
│   ├── DecomposeOpsForInference   # 推理模式算子分解
│   ├── DecomposeOpsForTraining    # 训练模式算子分解
│   └── CanonicalizeBindings       # 规范化变量绑定
│
├── 【融合类】操作融合优化
│   ├── AnnotateTIROpPattern  # 标注 TIR 操作模式
│   ├── FuseOps               # 图级操作融合
│   ├── FuseTIR               # 生成融合后的 TIR 代码
│   ├── FuseOpsByPattern      # 基于模式的融合
│   ├── CombineParallelMatmul # 合并并行矩阵乘法
│   ├── FuseTransposeMatmul   # 融合转置和矩阵乘法
│   └── MergeCompositeFunctions  # 合并复合函数
│
├── 【布局类】数据布局转换
│   ├── ConvertLayout         # 自动布局转换
│   ├── AlterOpImpl           # 替换算子实现 (不同布局)
│   └── FoldConstant          # 常量折叠 (配合布局优化)
│
├── 【内存类】内存管理优化
│   ├── StaticPlanBlockMemory # 静态规划 Block 内存 (替代 PlanBuffer)
│   ├── AllocateWorkspace     # 工作空间分配
│   ├── KillAfterLastUse      # 在最后使用后立即释放
│   ├── LiftTransformParams   # 提升变换参数为全局权重
│   ├── LazyTransformParams   # 懒加载变换参数
│   └── MergeCompositeFunctions  # 合并复合函数
│
├── 【死代码消除类】
│   ├── DeadCodeElimination   # 删除未使用的代码
│   ├── EliminateCommonSubexpr # 公共子表达式消除
│   ├── SimplifyInference     # 简化推理模式 (删除 training 专用操作)
│   ├── InlinePrivateFunctions  # 内联私有函数
│   ├── RemoveUnusedParameters  # 移除未使用的参数
│   └── RemoveUnusedOutputs     # 移除未使用的输出
│
├── 【精度优化类】
│   ├── ToMixedPrecision      # 混合精度转换 (FP32 → FP16)
│   └── FastMathTransform     # 快速数学变换优化
│
├── 【数据流转换类】
│   ├── ConvertToDataflow     # 转换为 dataflow 块
│   ├── ToNonDataflow         # 移除 dataflow 注解
│   └── DataflowUseInplaceCalls  # 原地调用优化
│
└── 【参数管理类】
    ├── BindParams            # 绑定参数
    ├── BundleModelParams     # 打包参数
    ├── LiftTransformParams   # 提升变换参数
    └── LazyTransformParams   # 懒加载变换参数
```

---

## 2. 降低类变换 (Lowering)

### 2.1 LegalizeOps

**作用**：将高级 `relax.op.*` 算子降低为 `call_tir` 调用。

```python
mod = tvm.relax.transform.LegalizeOps()(mod)
```

| 高级算子 | 降低后 |
|----------|--------|
| `relax.op.nn.relu` | `call_tir("fused_relax_nn_relu")` |
| `relax.op.nn.linear` | `call_tir("fused_relax_nn_matmul")` + `call_tir("fused_relax_nn_add")` |
| `relax.op.nn.conv2d` | `call_tir("fused_relax_nn_conv2d")` |

**何时使用**：图优化的**第一步**，后续融合依赖于降低后的 TIR 代码。

---

### 2.2 DecomposeOpsForInference

**作用**：将推理模式下的复杂算子分解为基础算子。

```python
mod = tvm.relax.transform.DecomposeOpsForInference()(mod)
```

**典型分解**：

| 复杂算子 | 分解为 |
|----------|--------|
| `BatchNorm` | `mean`, `variance`, `reshape`, `mul`, `add` |
| `LayerNorm` | `mean`, `sub`, `pow`, `sum`, `sqrt`, `div`, `mul`, `add` |
| `Dropout` (inference) | 被完全移除 |

**相关 API**：
- `DecomposeOpsForTraining()` - 训练模式的算子分解

**应用场景**：
- 硬件不支持复杂算子的原生实现
- 需要将复杂算子融入融合 kernel

---

### 2.3 CanonicalizeBindings

**作用**：规范化变量绑定，消除冗余变量。

```python
mod = tvm.relax.transform.CanonicalizeBindings()(mod)
```

**变换示例**：

```python
# 变换前
lv1 = x
lv2 = lv1
lv3 = lv2
return lv3

# 变换后
return x
```

---

## 3. 融合优化类 (Fusion)

### 3.1 融合三件套

```python
mod = tvm.ir.transform.Sequential([
    tvm.relax.transform.AnnotateTIROpPattern(),
    tvm.relax.transform.FuseOps(),
    tvm.relax.transform.FuseTIR(),
])(mod)
```

| Pass | 作用 |
|------|------|
| `AnnotateTIROpPattern` | 为每个 TIR 操作标注计算模式 (element-wise / reduction / injective) |
| `FuseOps` | 根据模式信息识别可融合的操作序列 |
| `FuseTIR` | 生成融合后的 TIR 函数 |

---

### 3.2 FuseOpsByPattern

**作用**：使用自定义模式进行融合。

```python
from tvm.relax.transform import FusionPattern, PatternCheckContext

# 定义融合模式
pattern = FusionPattern()
pattern.with_kind("ewise")
pattern.with_block("add", "relu")

mod = tvm.relax.transform.FuseOpsByPattern([pattern])(mod)
```

---

### 3.3 CombineParallelMatmul

**作用**：合并并行矩阵乘法操作。

```python
mod = tvm.relax.transform.CombineParallelMatmul()(mod)
```

**效果**：

```python
# 变换前
y1 = x @ w1
y2 = x @ w2
y3 = x @ w3

# 变换后 (合并为一个更大的 matmul)
w_combined = concat([w1, w2, w3], axis=1)
y_combined = x @ w_combined
y1, y2, y3 = split(y_combined, ...)
```

---

### 3.4 FuseTransposeMatmul

**作用**：融合转置和矩阵乘法操作。

```python
mod = tvm.relax.transform.FuseTransposeMatmul()(mod)
```

---

### 3.5 MergeCompositeFunctions

**作用**：将多个复合函数合并为一个。

```python
mod = tvm.relax.transform.MergeCompositeFunctions()(mod)
```

---

## 4. 内存优化类 (Memory)

### 4.1 StaticPlanBlockMemory

**作用**：静态规划内存分配，实现 Buffer 复用。

```python
mod = tvm.relax.transform.StaticPlanBlockMemory()(mod)
```

**效果**：

```python
# 变换前
temp1 = allocate(shape1)
temp2 = allocate(shape2)
temp3 = allocate(shape3)

# 变换后 (复用内存)
buffer = allocate(max(shape1, shape2, shape3))
temp1 = buffer[0:size1]
temp2 = buffer[0:size2]  # 复用
temp3 = buffer[0:size3]  # 复用
```

---

### 4.2 AllocateWorkspace

**作用**：为需要额外内存的外部函数分配统一的工作空间。

```python
mod = tvm.relax.transform.AllocateWorkspace()(mod)
```

**应用场景**：
- CUDA kernel 需要 shared memory
- 某些算法需要临时 buffer (如 FFT)

---

### 4.3 KillAfterLastUse

**作用**：在变量最后一次使用后立即释放内存。

```python
mod = tvm.relax.transform.KillAfterLastUse()(mod)
```

**效果**：

```python
# 变换前
temp1 = allocate(shape1)
temp2 = allocate(shape2)
result = compute(temp1, temp2)
return result  # temp1, temp2 在这里才释放

# 变换后
temp1 = allocate(shape1)
temp2 = allocate(shape2)
result = compute(temp1, temp2)
kill temp2  # 立即释放
kill temp1  # 立即释放
return result
```

---

### 4.4 LiftTransformParams

**作用**：将变换参数从函数参数提升为全局权重。

```python
mod = tvm.relax.transform.LiftTransformParams()(mod)
```

**效果**：

```python
# 变换前
def main(x, scale, zero_point):
    y = x * scale + zero_point
    return y

# 变换后 (scale 和 zero_point 变为权重)
def main(x):
    y = x * params.scale + params.zero_point
    return y
```

---

## 5. 布局优化类 (Layout)

### 5.1 ConvertLayout

**作用**：自动检测并转换布局。

```python
# 指定目标布局
desired_layouts = {
    "relax.nn.conv2d": ["NHWC", "OHWI"],
}

mod = tvm.relax.transform.ConvertLayout(desired_layouts)(mod)
```

**应用场景**：
- CPU 上运行 GPU 训练的模型 (NCHW → NHWC)
- 特定后端要求特定数据布局

---

### 5.2 AlterOpImpl

**作用**：替换算子的底层实现，支持不同布局。

```python
# 定义新的算子实现 (NHWC 布局)
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def conv2d_nhwc(...):
        # NHWC 优化的实现
        ...

# 创建替换映射
op_impl_map = {
    "relax.nn.conv2d": MyModule["conv2d_nhwc"]
}

# 定义布局转换
op_buffer_transforms = {
    "relax.nn.conv2d": [
        IndexMap.from_func(lambda n, h, w, c: (n, c, h, w))  # NHWC → NCHW
    ]
}

mod = tvm.relax.transform.AlterOpImpl(
    op_impl_map=op_impl_map,
    op_buffer_transforms=op_buffer_transforms
)(mod)
```

---

### 5.3 FoldConstant

**作用**：编译期计算常量表达式。

```python
mod = tvm.relax.transform.FoldConstant()(mod)
```

**变换示例**：

```python
# 变换前
y = x * 2.0 / 4.0 + 1.0

# 变换后
y = x * 0.5 + 1.0  # 2.0 / 4.0 在编译期计算为 0.5
```

---

## 6. 死代码消除类 (DCE)

### 6.1 DeadCodeElimination

**作用**：删除未使用的变量和函数。

```python
mod = tvm.relax.transform.DeadCodeElimination()(mod)
```

**消除示例**：

```python
# 变换前
lv1 = add(x, y)      # 未使用
lv2 = mul(x, 2)
lv3 = add(lv2, 1)    # 未使用
return lv2

# 变换后
lv2 = mul(x, 2)
return lv2
```

---

### 6.2 EliminateCommonSubexpr

**作用**：消除公共子表达式。

```python
mod = tvm.relax.transform.EliminateCommonSubexpr()(mod)
```

**变换示例**：

```python
# 变换前
y1 = add(mul(x, x), 1)
y2 = add(mul(x, x), 2)  # mul(x, x) 重复计算

# 变换后
temp = mul(x, x)
y1 = add(temp, 1)
y2 = add(temp, 2)
```

---

### 6.3 SimplifyInference

**作用**：简化推理模式，删除 training 专用操作。

```python
mod = tvm.relax.transform.SimplifyInference()(mod)
```

**效果**：

| 操作 | 训练模式 | 推理模式 |
|------|----------|----------|
| Dropout | 保留 | 删除 |
| BatchNorm | 保留 | 简化为固定 scale/shift |
| LayerNorm | 保留 | 简化 |

---

### 6.4 InlinePrivateFunctions

**作用**：内联私有函数，减少函数调用开销。

```python
mod = tvm.relax.transform.InlinePrivateFunctions()(mod)
```

---

### 6.5 RemoveUnusedParameters

**作用**：移除未使用的函数参数。

```python
mod = tvm.relax.transform.RemoveUnusedParameters()(mod)
```

---

### 6.6 RemoveUnusedOutputs

**作用**：移除未使用的输出。

```python
mod = tvm.relax.transform.RemoveUnusedOutputs()(mod)
```

---

## 7. 精度优化类 (Precision)

### 7.1 ToMixedPrecision

**作用**：将模型转换为混合精度（FP32 → FP16/BF16）。

```python
mod = tvm.relax.transform.ToMixedPrecision()(mod)
```

**变换效果**：

```python
# 变换前 (FP32)
y = conv2d(x, w)  # x, w, y 都是 FP32

# 变换后 (混合精度)
y_fp16 = conv2d(x_fp16, w_fp16)  # 计算使用 FP16
y = cast(y_fp16, "float32")       # 结果转回 FP32
```

---

### 7.2 FastMathTransform

**作用**：使用快速数学函数替代标准实现。

```python
mod = tvm.relax.transform.FastMathTransform()(mod)
```

**优化示例**：

| 标准函数 | 快速版本 | 说明 |
|---------|---------|------|
| `exp` | `fast_exp` | 降低精度的指数函数 |
| `tanh` | `fast_tanh` | 降低精度的 tanh |
| `sqrt` | `rsqrt` | 倒数平方根 |

---

## 8. 数据流转换类 (Dataflow)

### 8.1 ConvertToDataflow

**作用**：将连续的数据流操作转换为 dataflow 块。

```python
mod = tvm.relax.transform.ConvertToDataflow()(mod)
```

**效果**：

```python
# 变换前
lv1 = add(x, y)
lv2 = mul(lv1, z)
lv3 = relu(lv2)

# 变换后
with R.dataflow():
    lv1 = add(x, y)
    lv2 = mul(lv1, z)
    lv3 = relu(lv2)
    R.output(lv3)
```

---

### 8.2 ToNonDataflow

**作用**：移除 dataflow 注解，转换为普通绑定块。

```python
mod = tvm.relax.transform.ToNonDataflow()(mod)
```

---

### 8.3 DataflowUseInplaceCalls

**作用**：在 dataflow 块中使用原地调用优化。

```python
mod = tvm.relax.transform.DataflowUseInplaceCalls()(mod)
```

---

## 9. 参数管理类 (Params)

### 9.1 BindParams

**作用**：将参数绑定到模块中。

```python
mod = tvm.relax.transform.BindParams("main", params_dict)(mod)
```

---

### 9.2 BundleModelParams

**作用**：将模型参数打包成单一结构。

```python
mod = tvm.relax.transform.BundleModelParams()(mod)
```

---

### 9.3 LiftTransformParams

**作用**：将变换参数从函数参数提升为全局权重。

```python
mod = tvm.relax.transform.LiftTransformParams()(mod)
```

---

### 9.4 LazyTransformParams

**作用**：懒加载变换参数，仅在需要时计算。

```python
mod = tvm.relax.transform.LazyTransformParams()(mod)
```

**相关 API**：
- `LazySetOutput()` - 懒加载设置输出
- `LazyGetInput()` - 懒加载获取输入

---

## 10. 图优化典型流程

### 10.1 标准优化流水线

```python
def standard_optimization_pipeline(mod):
    """标准图优化流水线"""
    return tvm.ir.transform.Sequential([
        # 1. 降低高级算子
        tvm.relax.transform.LegalizeOps(),

        # 2. 简化和规范化
        tvm.relax.transform.CanonicalizeBindings(),
        tvm.relax.transform.FoldConstant(),
        tvm.relax.transform.SimplifyInference(),

        # 3. 融合优化
        tvm.relax.transform.AnnotateTIROpPattern(),
        tvm.relax.transform.FuseOps(),
        tvm.relax.transform.FuseTIR(),

        # 4. 死代码消除
        tvm.relax.transform.DeadCodeElimination(),

        # 5. 内存优化
        tvm.relax.transform.StaticPlanBlockMemory(),
    ])(mod)
```

---

### 10.2 CPU 优化流水线

```python
def cpu_optimization_pipeline(mod):
    """针对 CPU 的优化流水线"""
    return tvm.ir.transform.Sequential([
        # 布局转换 (NCHW → NHWC)
        tvm.relax.transform.ConvertLayout({
            "relax.nn.conv2d": ["NHWC", "OHWI"],
        }),

        # 降低
        tvm.relax.transform.LegalizeOps(),

        # 融合
        tvm.relax.transform.AnnotateTIROpPattern(),
        tvm.relax.transform.FuseOps(),
        tvm.relax.transform.FuseTIR(),

        # 内存规划
        tvm.relax.transform.StaticPlanBlockMemory(),
    ])(mod)
```

---

### 10.3 LoRA 优化流水线

```python
def lora_optimization_pipeline(mod):
    """LoRA (Low-Rank Adaptation) 优化流水线"""
    return tvm.ir.transform.Sequential([
        # 调整矩阵乘法顺序以减少内存
        # 原: x @ (A @ B)  →  新: (x @ A) @ B
        tvm.relax.transform.AdjustMatmulOrder(),

        # 标准优化
        tvm.relax.transform.LegalizeOps(),
        tvm.relax.transform.FoldConstant(),

        # 融合
        tvm.relax.transform.AnnotateTIROpPattern(),
        tvm.relax.transform.FuseOps(),
        tvm.relax.transform.FuseTIR(),

        # 内存优化 (LoRA 特别需要)
        tvm.relax.transform.StaticPlanBlockMemory(),
    ])(mod)
```

---

### 10.4 混合精度优化流水线

```python
def mixed_precision_pipeline(mod):
    """混合精度优化流水线"""
    return tvm.ir.transform.Sequential([
        # 1. 转换为混合精度
        tvm.relax.transform.ToMixedPrecision(),

        # 2. 标准优化流程
        tvm.relax.transform.LegalizeOps(),
        tvm.relax.transform.AnnotateTIROpPattern(),
        tvm.relax.transform.FuseOps(),
        tvm.relax.transform.FuseTIR(),

        # 3. 死代码消除
        tvm.relax.transform.DeadCodeElimination(),
    ])(mod)
```

---

## 11. Pass 顺序指南

### 正确的 Pass 顺序

```
1. LegalizeOps          # 先降低，让所有算子变为 call_tir
       │
2. Simplify*            # 简化图结构 (FoldConstant, SimplifyInference)
       │
3. Transform Layout     # 布局转换 (在融合前做，避免破坏融合)
       │
4. Annotate + Fusion    # 融合优化
       │
5. Memory Optimization  # 内存规划 (在融合后做)
       │
6. DCE                  # 最后清理死代码
```

### 常见错误顺序

| 错误顺序 | 问题 |
|----------|------|
| 先 Fuse 后 Legalize | Fuse 无法识别高级算子 |
| 先 Memory 后 Fusion | 融合会破坏内存规划 |
| 先 DCE 后 Fusion | 融合可能引入新的死代码 |

---

## 12. 调试技巧

### 12.1 查看 Pass 效果

```python
def debug_pass(mod, pass_func, name):
    """调试单个 Pass 的效果"""
    print(f"\n=== Before: {name} ===")
    mod.show()

    mod = pass_func(mod)

    print(f"\n=== After: {name} ===")
    mod.show()

    return mod

# 使用
mod = debug_pass(mod, tvm.relax.transform.LegalizeOps(), "LegalizeOps")
mod = debug_pass(mod, tvm.relax.transform.FuseOps(), "FuseOps")
```

---

### 12.2 PassContext 配置

```python
# 配置 Pass 上下文
with tvm.transform.PassContext(
    opt_level=3,                    # 优化级别
    required_pass=["FuseOps"],      # 必须运行的 Pass
    disabled_pass=["LegalizeOps"],  # 禁用的 Pass
):
    mod = tvm.relax.transform.LegalizeOps()(mod)
```

---

### 12.3 打印 Pass 统计

```python
# 启用性能分析
with tvm.transform.PassContext(config={"relax.FuseOps": {"max_depth": 8}}):
    mod = tvm.relax.transform.FuseOps()(mod)

# 打印融合统计
print("Fused functions:")
for gvar, func in mod.functions.items():
    if "fused_" in gvar.name_hint:
        print(f"  - {gvar.name_hint}")
```

---

## 参考资料

- [TVM Relax Transform API 官方文档](https://tvm.apache.org/docs/reference/api/python/relax/transform.html)
- [TVM Pass 基础设施](https://tvm.apache.org/docs/arch/pass_infra.html)
- [Relax 设计文档](https://github.com/apache/tvm/blob/main/src/relax/README.md)

