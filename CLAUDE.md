# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 提供在此仓库中工作的指导。

## 仓库概述

这是 **MLC（机器学习编译）课程** 的学习工作区，来自 mlc-ai。仓库包含教学用的 Jupyter 笔记本，涵盖 TVM（Tensor Virtual Machine）概念，包括 TensorIR、Relax 和端到端模型编译。

## 环境配置

所有笔记本都需要 `mlc-ai-nightly` 包（**不是**稳定的 `tvm` 包）：

```bash
pip install mlc-ai-nightly -f https://mlc.ai/wheels
```

此 nightly 版本包含课程材料中使用的最新 TensorIR/Relax API 变更。使用稳定的 `tvm` 可能会导致与笔记本代码不兼容。

## 主要依赖

- `tvm` / `mlc-ai-nightly` - 核心机器学习编译框架
- `numpy` - 数组操作和参考实现
- `torch`（可选） - 用于外部库集成示例
- `torchvision` - 用于数据集加载（Fashion MNIST 示例）

## 常用代码模式

### 运行笔记本

这些笔记本设计为按顺序逐个单元格运行。关键操作：

```python
# 构建 TensorIR 模块
ex = relax.build(MyModule, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

# 从 numpy 创建 TVM NDArray
data_nd = tvm.nd.array(data_np)

# 执行模型
res = vm["main"](data_nd, w0_nd, b0_nd, w1_nd, b1_nd)

# 转换回 numpy
res_np = res.numpy()
```

### IRModule 结构

```python
@tvm.script.ir_module
class ModuleName:
    # 低级张量函数（TensorIR）
    @T.prim_func
    def func_name(...):
        # T.axis.spatial() - 空间循环轴
        # T.axis.reduce() - 归约循环轴
        ...

    # 高级计算图（Relax）
    @R.function
    def main(x: R.Tensor(...)) -> R.Tensor(...):
        with R.dataflow():
            # R.call_tir() - 调用原始函数
            # R.call_dps_packed() - 调用外部库
            ...
```

### Schedule 变换

```python
# 创建 schedule
sch = tvm.tir.Schedule(MyModule)

# 常用变换
sch.split(loop, factors=[None, 4])  # 分割循环
sch.reorder(...)                     # 重排序循环
sch.reverse_compute_at(block, loop)  # 融合计算
sch.decompose_reduction(block)       # 分离初始化/更新
```

## TVM/MLC 核心概念

- **TensorIR (`T.prim_func`)**: 低级张量程序抽象，具有显式循环、块注解和缓冲区语义
- **Relax (`R.function`)**: 高级神经网络执行抽象
- **`call_tir`**: Relax 和 TensorIR 之间的桥梁，使用目标传递风格
- **Dataflow 块**: 计算图区域的作用域注解
- **IRModule**: 同时包含 TensorIR 和 Relax 函数的容器

## API 变更说明

笔记本可能使用已弃用的语法，会触发警告（例如 `T.Buffer[...]` 语法）。这些警告不会阻止执行，但表明 API 正在演进。
