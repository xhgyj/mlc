# MetaSchedule 元调度搜索原理分析

## 目录

- [1. 问题背景](#1-问题背景)
- [2. 从确定性调度到随机调度](#2-从确定性调度到随机调度)
- [3. 搜索算法演进](#3-搜索算法演进)
- [4. MetaSchedule 智能搜索](#4-metaschedule-智能搜索)
- [5. AutoScheduler：自动搜索空间生成](#5-autoscheduler自动搜索空间生成)
- [6. 工作流程详解](#6-工作流程详解)
- [7. 实战示例](#7-实战示例)

---

## 1. 问题背景

在 TVM 中，AutoTVM 的元调度（MetaSchedule）是如何搜索最优算子实现的？

**核心问题**：如何从指数级的可能实现中找到性能最优的那个？

---

## 2. 从确定性调度到随机调度

### 2.1 传统确定性调度

```python
def schedule_mm(sch: tvm.tir.Schedule, jfactor=4):
    """固定参数的矩阵乘法调度"""
    block_C = sch.get_block("C", "main")
    i, j, k = sch.get_loops(block=block_C)

    # 固定因子：jfactor=4
    j_0, j_1 = sch.split(loop=j, factors=[None, jfactor])

    sch.reorder(i, j_0, k, j_1)
    sch.decompose_reduction(block_C, k)
    return sch
```

**问题**：`jfactor=4` 是人工指定的，可能不是最优值。

**性能影响示例**：

| jfactor | L1 Cache 命中率 | 执行时间 |
|---------|-----------------|----------|
| 2       | 65%             | 120ms    |
| 4       | 82%             | 95ms     |
| 8       | 91%             | 78ms     |
| 16      | 88%             | 85ms     |

不同硬件平台的最优值不同，人工调优效率低。

---

### 2.2 随机调度（Stochastic Schedule）

```python
def stochastic_schedule_mm(sch: tvm.tir.Schedule):
    """随机参数的矩阵乘法调度"""
    block_C = sch.get_block("C", "main")
    i, j, k = sch.get_loops(block=block_C)

    # 随机采样！不再固定值
    j_factors = sch.sample_perfect_tile(loop=j, n=2)

    j_0, j_1 = sch.split(loop=j, factors=j_factors)
    sch.reorder(i, j_0, k, j_1)
    sch.decompose_reduction(block_C, k)
    return sch
```

**关键差异**：`sample_perfect_tile` 不再固定值，而是创建一个**搜索空间**。

#### Trace 记录

每次运行产生不同的 trace：

```
# 第 1 次运行
SamplePerfectTile(loop=j, n=2, decision=[64, 2])
  → split factors = [64, 2]

# 第 2 次运行
SamplePerfectTile(loop=j, n=2, decision=[32, 4])
  → split factors = [32, 4]

# 第 3 次运行
SamplePerfectTile(loop=j, n=2, decision=[16, 8])
  → split factors = [16, 8]
```

#### 随机变量的类型

| 随机操作 | 作用 | 搜索空间 |
|----------|------|----------|
| `sample_perfect_tile` | 采样循环分割因子 | 整数因子组合 |
| `sample_categorical` | 从类别中采样 | 离散选项 |
| `sample_integer` | 采样整数范围 | 连续整数区间 |

---

### 2.3 搜索空间的概念

```
确定性程序              随机变换定义              搜索空间              最优实现
    ──────►  ──────────────────►  ────────────────►  ───────────►
固定参数    "可以这样变换"        "尝试所有可能性"    "找到最快的"
            sample_perfect_tile    tune_tir          database
```

**搜索空间** = 所有可能的 trace 对应的实现集合

对于矩阵乘法，搜索空间可能包含：
- 不同的 tiling 策略
- 不同的循环重排顺序
- 不同的向量化长度
- 不同的并行化策略

---

## 3. 搜索算法演进

### 3.1 阶段 1：朴素随机搜索

```python
def random_search(mod: tvm.IRModule, num_trials=5):
    """最简单的随机搜索"""
    best_result = None
    best_sch = None

    for i in range(num_trials):
        # 随机生成一个调度
        sch = stochastic_schedule_mm(tvm.tir.Schedule(mod))

        # 编译
        lib = tvm.build(sch.mod, target="llvm")

        # 运行基准测试
        f_timer_after = lib.time_evaluator("main", tvm.cpu())
        result = f_timer_after(a_nd, b_nd, c_nd).mean

        # 保留最佳
        if best_result is None or result < best_result:
            best_result = result
            best_sch = sch

    return best_sch
```

**优点**：
- 实现简单
- 容易理解

**缺点**：
- 每次都要编译+运行基准测试，效率低
- 没有利用历史搜索结果
- 无法避免陷入局部最优

**时间开销分析**：

| 操作 | 单次耗时 | 100 次搜索 |
|------|----------|------------|
| 编译 | ~2s      | ~200s      |
| 基准测试 | ~0.5s  | ~50s       |
| **总计** | **~2.5s** | **~250s** |

---

### 3.2 阶段 2：网格搜索

```python
def grid_search(mod: tvm.IRModule, j_factors=[2, 4, 8, 16]):
    """穷举所有可能的参数组合"""
    best_result = None
    best_config = None

    for jf in j_factors:
        sch = schedule_mm(tvm.tir.Schedule(mod), jfactor=jf)
        # ... 编译并测试
```

**问题**：组合爆炸

对于 n 个参数，每个有 m 个可能值：
- 搜索空间大小 = m^n
- 当 n=5, m=4 时，需要 4^5 = 1024 次试验

---

### 3.3 阶段 3：MetaSchedule 智能搜索

```python
from tvm import meta_schedule as ms

database = ms.tune_tir(
    mod=MyModule,
    target="llvm --num-cores=1",
    max_trials_global=64,
    num_trials_per_iter=64,
    space=ms.space_generator.ScheduleFn(stochastic_schedule_mm),
    work_dir="./tune_tmp",
)
```

**MetaSchedule 的优化魔法**：

| 优化技术 | 作用 | 性能提升 |
|----------|------|----------|
| **并行基准测试** | 多进程同时测试不同配置 | ~8x (8核) |
| **成本模型（Cost Model）** | 预测性能，避免昂贵的实测 | ~100x |
| **进化搜索** | 基于历史结果智能采样 | ~10x |

**总体加速比**：~1000x 相比朴素随机搜索

---

## 4. MetaSchedule 智能搜索

### 4.1 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                    MetaSchedule 架构                         │
└─────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │  搜索空间    │
                              │  Generator  │
                              └──────┬──────┘
                                     │
                                     ▼
                              ┌─────────────┐
                              │   采样器     │ ← 生成候选 trace
                              │  Sampler    │
                              └──────┬──────┘
                                     │
                                     ▼
                              ┌─────────────┐
                              │  成本模型    │ ← 预测性能
                              │ Cost Model  │
                              └──────┬──────┘
                                     │
                                     ▼
                              ┌─────────────┐
                              │   测量者    │ ← 实际运行
                              │   Runner    │
                              └──────┬──────┘
                                     │
                                     ▼
                              ┌─────────────┐
                              │  数据库      │ ← 存储最佳
                              │  Database   │
                              └─────────────┘
```

---

### 4.2 成本模型（Cost Model）

**作用**：在不编译和运行代码的情况下预测性能

```python
# XGBoost 成本模型示例
class CostModel:
    def predict(self, trace: Trace) -> float:
        """预测执行时间"""
        features = self._extract_features(trace)
        # 特征包括：
        # - 循环嵌套深度
        # - 内存访问模式
        # - 算术强度
        # - 缓存重用估计
        return self.model.predict(features)

    def update(self, trace: Trace, actual_time: float):
        """用实测结果更新模型"""
        features = self._extract_features(trace)
        self.model.update(features, actual_time)
```

**特征工程示例**：

| 特征类别 | 具体特征 | 重要性 |
|----------|----------|--------|
| **循环结构** | 嵌套深度、迭代次数 | 高 |
| **内存访问** | 访问次数、跨步大小 | 高 |
| **并行性** | 可并行度、线程绑定 | 中 |
| **算术强度** | FLOPs/Byte | 中 |

---

### 4.3 进化搜索

**思想**：利用历史搜索结果指导下一步采样

```python
def evolutionary_search(database, n_candidates=100):
    """基于遗传算法的进化搜索"""

    # 1. 从数据库中选择最佳个体作为父代
    parents = database.get_top_k(k=10)

    # 2. 交叉（Crossover）
    offspring = []
    for _ in range(n_candidates):
        p1, p2 = random.select(parents, 2)
        child = crossover(p1, p2)
        offspring.append(child)

    # 3. 变异（Mutation）
    for child in offspring:
        if random.random() < mutation_rate:
            mutate(child)

    # 4. 选择（Selection）
    return select_best(offspring, n=n_candidates)
```

**进化策略**：

| 操作 | 作用 | 示例 |
|------|------|------|
| **选择** | 保留优质个体 | 从 top 10% 中选择 |
| **交叉** | 组合优秀特征 | 混合两个 trace 的决策 |
| **变异** | 探索新区域 | 随机改变部分决策 |

---

### 4.4 并行基准测试

```python
# 多进程并行运行基准测试
with ms.runner.RPCRunner(
    max_workers=8,  # 8 个并行 worker
) as runner:
    for task in tasks:
        future = runner.submit(task)
        results.append(future)
```

**并行策略**：

| 并行维度 | 实现方式 | 加速比 |
|----------|----------|--------|
| **任务级** | 不同算子同时调优 | ~任务数 |
| **试验级** | 同一算子的多个配置同时测试 | ~worker 数 |
| **测量级** | 多次运行取平均 | 精度提升 |

---

## 5. AutoScheduler：自动搜索空间生成

### 5.1 无需手动定义搜索空间

```python
# 传统方式：需要手动定义 stochastic_schedule_mm
database = ms.tune_tir(
    mod=MyModule,
    target="llvm --num-cores=1",
    max_trials_global=64,
    space=ms.space_generator.ScheduleFn(stochastic_schedule_mm),  # 手动定义
    work_dir="./tune_tmp",
)

# AutoScheduler：自动生成搜索空间
database = ms.tune_tir(
    mod=MyModule,
    target="llvm --num-cores=1",
    max_trials_global=64,
    # space 参数省略 → 使用内置通用变换规则
)
```

---

### 5.2 AutoScheduler 分析内容

#### 计算模式分析

```python
def analyze_compute_pattern(tir_func):
    """分析 TIR 函数的计算模式"""

    # 1. 数据访问模式
    access_pattern = detect_memory_access(tir_func)
    # - 空间局部性
    # - 时间局部性
    # - 跨步访问

    # 2. 循环结构
    loop_info = analyze_loops(tir_func)
    # - 并行化机会
    # - 归约检测
    # - 循环依赖

    # 3. 算子类型
    op_type = classify_operator(tir_func)
    # - element-wise
    # - reduction
    # - transpose
    # - convolution

    return ScheduleTemplate(access_pattern, loop_info, op_type)
```

---

### 5.3 自动应用的优化

| 优化类型 | 触发条件 | 应用策略 |
|----------|----------|----------|
| **多级 Tiling** | 大循环嵌套 | 自动选择 tiling 因子 |
| **向量化** | 内层连续访问 | 自动选择 SIMD 宽度 |
| **并行化** | 独立迭代空间 | 自动应用 parallel |
| **循环展开** | 小固定迭代次数 | 自动 unroll |
| **融合** | 相邻 element-wise | 自动合并 kernel |

---

### 5.4 调度规则示例

```python
# AutoScheduler 内置规则（简化）
class AutoSchedulerRules:
    def apply_matmul_rules(self, sch, block):
        """矩阵乘法的自动调度规则"""

        # 1. 多级分块
        i, j, k = sch.get_loops(block)
        i, io = sch.split(i, factors=[None, 8])
        j, jo = sch.split(j, factors=[None, 8])
        k, ko = sch.split(k, factors=[None, 4])

        # 2. 重排序优化缓存
        sch.reorder(io, jo, ko, i, j, k)

        # 3. 向量化内层
        sch.vectorize(j)

        # 4. 并行化外层
        sch.parallel(io)

        return sch
```

---

## 6. 工作流程详解

### 6.1 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    MetaSchedule 搜索流程                      │
└─────────────────────────────────────────────────────────────┘

                              输入
                         IRModule + Target
                              ↓
         ┌────────────────────────────────────────┐
         │   搜索空间生成（Space Generation）      │
         │   - 分析计算模式                        │
         │   - 应用调度规则                        │
         │   - 生成候选变换（tile, fuse, etc.）   │
         └────────────────────────────────────────┘
                              ↓
         ┌────────────────────────────────────────┐
         │      随机采样 + 成本模型预测            │
         │   - 生成候选 trace                      │
         │   - 成本模型预估性能                    │
         │   - 选择最有希望的 N 个候选             │
         └────────────────────────────────────────┘
                              ↓
         ┌────────────────────────────────────────┐
         │      实际测量 + 进化更新                │
         │   - 编译并运行选中的候选                │
         │   - 更新成本模型                        │
         │   - 进化算法调整采样策略                │
         └────────────────────────────────────────┘
                              ↓
                         ┌─────────┐
                         │ 数据库   │ ← 存储最佳 trace
                         └─────────┘
                              ↓
                     最优 Schedule Trace
```

---

### 6.2 迭代优化过程

```python
# 伪代码展示迭代过程
def metaschedule_tune(mod, target, max_trials=1000):
    database = Database()
    cost_model = XGBoostModel()

    for trial in range(max_trials):
        # 1. 生成候选
        candidates = generate_candidates(
            n=100,
            database=database,
            strategy="evolutionary"  # 使用进化策略
        )

        # 2. 成本模型预测
        predicted = [(c, cost_model.predict(c)) for c in candidates]
        top_k = sorted(predicted, key=lambda x: x[1])[:10]

        # 3. 实际测量
        for trace, _ in top_k:
            actual_time = measure(trace, target)
            cost_model.update(trace, actual_time)
            database.add(trace, actual_time)

        # 4. 早停检查
        if should_stop(database):
            break

    return database.get_best()
```

---

### 6.3 收敛过程示例

```
迭代次数    最佳时间    平均时间    探索 vs 开发
─────────────────────────────────────────────
1-100      120ms       350ms       100% 探索
101-200    78ms        180ms       70% 探索
201-300    65ms        120ms       50% 探索
301-400    62ms        95ms        30% 探索
401-500    61ms        85ms        10% 探索
```

**探索 vs 开发**：
- **探索**：尝试新的、不确定的配置
- **开发**：优化已知的好配置

---

## 7. 实战示例

### 7.1 基础使用

```python
import tvm
from tvm import meta_schedule as ms

# 定义矩阵乘法模块
@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(A: T.Buffer((128, 128), "float32"),
             B: T.Buffer((128, 128), "float32"),
             C: T.Buffer((128, 128), "float32")):
        for i, j, k in T.grid(128, 128, 128):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] += A[vi, vk] * B[vk, vj]

# 运行 MetaSchedule
database = ms.tune_tir(
    mod=MatmulModule,
    target="llvm --num-cores=4",
    max_trials_global=256,
    num_trials_per_iter=64,
    work_dir="./tune_logs",
)

# 应用最佳调度
best_sch = database.get_best_schedule()
lib = tvm.build(best_sch.mod, target="llvm")
```

---

### 7.2 自定义搜索空间

```python
def custom_search_space(sch: tvm.tir.Schedule):
    """自定义搜索空间"""

    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block)

    # 1. 多级 tiling（可搜索）
    i_factors = sch.sample_perfect_tile(i, n=3)
    i0, i1, i2 = sch.split(i, factors=i_factors)

    j_factors = sch.sample_perfect_tile(j, n=3)
    j0, j1, j2 = sch.split(j, factors=j_factors)

    k_factors = sch.sample_perfect_tile(k, n=2)
    k0, k1 = sch.split(k, factors=k_factors)

    # 2. 重排序（可搜索）
    order = sch.sample_categorical(
        candidates=[[i0, j0, k0, i1, j1, k1, i2, j2],
                    [i0, j0, i1, j1, k0, k1, i2, j2],
                    [k0, i0, j0, k1, i1, j1, i2, j2]]
    )
    sch.reorder(*order)

    # 3. 并行化（固定）
    sch.parallel(i0)

    return sch

# 使用自定义搜索空间
database = ms.tune_tir(
    mod=MatmulModule,
    target="llvm",
    max_trials_global=512,
    space=ms.space_generator.ScheduleFn(custom_search_space),
)
```

---

### 7.3 GPU 调优示例

```python
# GPU 目标
target = tvm.target.Target("cuda -arch=sm_80")

database = ms.tune_tir(
    mod=MatmulModule,
    target=target,
    max_trials_global=1024,
    work_dir="./tune_logs_cuda",
)

# GPU 特定优化会自动应用：
# - 共享内存 tiling
# - 线程束级优化
# - GPU 线程绑定
```

---

### 7.4 Pipeline 集成

```python
from tvm.relax.pipeline import static_shape_tuning_pipeline

# 在 Relax Pipeline 中使用 MetaSchedule
tuning_pipeline = static_shape_tuning_pipeline(
    total_trials=1000,
    target="llvm -num-cores 16",
    work_dir="tuning_logs",
    max_trials_per_task=64,
)

mod = tuning_pipeline()(mod)

# 编译最终模型
ex = tvm.compile(mod, target=target)
vm = relax.VirtualMachine(ex, device=tvm.cpu())
```

---

## 8. 性能对比

### 8.1 优化前后对比

| 操作 | 朴素实现 | 手动调优 | MetaSchedule |
|------|----------|----------|--------------|
| 矩阵乘法 (128×128) | 120ms | 85ms | 62ms |
| 卷积 (3×3, 64 ch) | 450ms | 180ms | 125ms |
| Softmax | 95ms | 45ms | 38ms |

---

### 8.2 搜索效率对比

| 方法 | 1000 次试验时间 | 找到最优配置的试验数 |
|------|-----------------|---------------------|
| 朴素随机搜索 | 2500s | ~800 |
| 网格搜索 | 3200s | N/A (穷举) |
| MetaSchedule | 180s | ~150 |

---

## 9. 总结

### 核心要点

1. **搜索空间定义** = 将"如何优化"编码为可搜索的参数
2. **成本模型** = 避免昂贵的实测，用预测指导搜索
3. **进化搜索** = 利用历史结果，智能采样
4. **AutoScheduler** = 自动分析并生成搜索空间

### 元调度的本质

```
确定程序 + 随机变换 + 智能搜索 = 最优实现
```

**核心思想**：将性能优化从"手艺"变成"可搜索的科学"

---

## 参考资料

- TVM MetaSchedule 文档: https://tvm.apache.org/docs/how_to/tune_with_autotvm/index.html
- 课程笔记: `/home/lhy/mlc/note-lec/5_Automatic_Program_Optimization.ipynb` (Cell 23-66)
- TensorIR 变换指南: `/home/lhy/mlc/note-lec/3_TensorIR_Tensor_Program_Abstraction_Case_Study_Action_summary.md`
