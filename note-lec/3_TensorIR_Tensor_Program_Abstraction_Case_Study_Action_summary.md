# Tensor Program Abstraction Case Study (TensorIR) — 深度笔记

## 1. 环境准备（单元 1-4）
开篇先通过 Colab 徽章提醒我们：这份教材依赖 TVM 的 mlc-ai nightly 版本。`mlc-ai-nightly` 将打包最新的 TensorIR / 调度改动，不一定与稳定版 TVM 一致，但它能保证后续示例（尤其是 schedule API）可直接运行。安装命令 `pip install mlc-ai-nightly -f https://mlc.ai/wheels` 会拉取 LLVM 支持的 CPU 编译后端；如果未安装，后续 `tvm.build`、`tir.Schedule` 等步骤可能直接报错或给出缺失算子提示。因此学习时要先确认 Python 环境隔离（虚拟环境/conda）并安装上述 wheel，再继续阅读。

## 2. 课前导入与动机示例（单元 5-16）
教材先回顾 “ML Compilation（MLC）= 构造可转换的张量程序 + 不断验证 + 迭代优化” 这一原则，然后给我们一个 128×128 的矩阵乘加 ReLU 的经典算子。这里特意对同一算子写了两个版本：普通 NumPy 一行搞定，与“低层 NumPy”（显式 for 循环 + 手动分配 `np.empty`）版本。第二种写法虽然笨重，但它逼迫我们思考真实硬件执行会经历的循环、缓存、写回等细节。通过单元 15 的 `np.testing.assert_allclose`，我们验证低层实现正确，从而建立一个“地面真相”——后续所有 TensorIR 代码只要能与它对齐，就说明语义无误。

## 3. TensorIR 基础（单元 17-36）
TensorIR 使用 `@tvm.script.ir_module` 和 `@T.prim_func` 在 Python AST 中定义算子。需要注意的是，`T.Buffer[...]` 语法在新版中已被弃用，官方鼓励写成调用式 `T.Buffer((m, n), "float32")`，否则虽能运行但会反复收到 DeprecationWarning。教材随后用一系列图示解释 block 的含义：一个 block 代表一段循环体，`T.axis.spatial` 对应纯迭代轴，`T.axis.reduce` 则表示归约轴。通过 block metadata，TensorIR 能把“当前循环在计算哪个逻辑张量位置”与“这段循环在做 init 还是 update”显式写出来，为后续调度提供足够的语义信息。此外，IRModule 不止能存一个函数，示例展示了如何把 `mm` 和 `relu` 切成两个独立的 `PrimFunc` 并放在同一 module 里——这对于描述后端联编或组合图模式非常有用。到这一节末尾，读者应理解：**TensorIR = 指令级循环 + block 标签 + Buffer 读写注解**，它比普通 Python/NumPy 更接近硬件。

## 4. 变换流程示例（单元 37-58）
这一节的目标是让你真正掌握 `tir.Schedule` 的基本操作。首先通过 `lnumpy_mm_relu_v2` 展示“手动做 loop splitting/tiling”长什么样：将 `j` 轴拆成 `j0`(32) 和 `j1`(4) 两层循环，改变迭代顺序以改善访问局部性。接着切换到 TensorIR：
1. `sch = tvm.tir.Schedule(MyModule)` 复制出可变 IR；
2. `get_block`、`get_loops` 获取 block Y 及其循环句柄；
3. `split(j, factors=[None, 4])` 表示“外层长度自动推导，内层固定为 4”；
4. `reorder` 调整循环次序；
5. `reverse_compute_at` 把输出 block 往内层推，从而贴近数据生产者；
6. `decompose_reduction` 将 `with T.init():` 的初始化块剥离，显式化“init/update”两个阶段。

这些操作并不是为了炫技，而是告诉你：**任何 loop-level 转换都可以程序化复现**，而且顺序敏感——Schedule 是命令式的，重复调用会报“找不到原循环”的错，需要从头重来。完成这些步骤后，我们得到的 `sch.mod.show()` 与 `lnumpy_mm_relu_v3` 在结构上对齐，证明 schedule 的语义就是对 IR 进行等价重写。

## 5. 构建与基准测试（单元 59-71）
完成变换后要落地到真实可执行模块。`tvm.build(MyModule, target="llvm")` 会调用 TVM 后端（默认 CPU），把 TensorIR 降成 LLVM IR，再生成可调用的 runtime 模块。输入输出使用 `tvm.nd.array` 封装 NumPy 数组，运行 `rt_lib["mm_relu"](a_nd, b_nd, c_nd)` 时，TVM 会自动处理 dtype、stride。重要的是：再次用 `np.testing.assert_allclose` 验证结果，确保 schedule 没破坏语义。

`time_evaluator` 是 TVM 自带基准工具，模式是“多次 warmup + 多次测量，返回均值和方差”。通过比较 `rt_lib`（原版）与 `rt_lib_after`（变换版）的耗时，可以直观看到 loop 变换的价值。即使在 CPU 上，合理的 tiling / reorder 也能带来显著速度提升，尤其当原始写法缓存友好性差时。

## 6. 性能直觉（单元 72-78）
光看到数字还不够，教材进一步用图示解释**缓存层次**与**循环顺序**的关系：内层循环如果连续访问内存，会充分利用 cache line；如果频繁跨行跨列，则会触发更多 miss。`reverse_compute_at` 可以让 ReLU 紧跟着矩阵乘的局部块执行，避免把巨大中间张量写回内存。`transform(mod, jfactor)` 练习函数则鼓励我们：改变分块因子，观察 `time_evaluator` 输出，就能形成“如何为特定硬件挑选分块大小”的直觉。

## 7. 程序化生成 TensorIR（单元 79-83）
手写 TensorIR 适合教学和小规模算子，但大型网络往往先用更高层的 Tensor Expression (TE) 来描述计算，再通过 `te.create_prim_func` 自动生成 TensorIR。TE 里 `te.compute` 接收形状与 `fcompute` 回调（即如何根据索引计算元素），然后 `with_attr({"global_symbol": "mm_relu"})` 指定导出名。生成后的 PrimFunc 与手写版一样可以喂给 `tir.Schedule`，这就是 TVM AutoTVM/MetaSchedule 的基础：先自动生成合法 IR，再自动搜索调度空间。

## 8. 讨论与最终总结（单元 84-89）
最后的讨论强调 MLC 的闭环：**提出实现 → 应用调度 → 构建 & 测试 → 结合硬件反馈继续调度或换实现**。教材用流程图对比“直接写 TensorIR”与“借助 schedule/TE 生成”，说明两者互补：
- 当你需要研究新的硬件特性或发明新变换时，手写 TensorIR 可精准控制每个细节；
- 当你想在大模型上批量优化时，自动调度器会利用同样的 schedule 原语搜索最优组合。

总结段再次提醒：掌握 block / axis 语义，是理解 TVM 如何保障正确性的关键；而 schedule API 则是把“手动 loop 优化”程序化、可复用的途径。只要能熟练把 TensorIR 编译到 runtime module，并用 `time_evaluator` 反馈性能，我们就真正跨出了学习 TVM 的第一步。
