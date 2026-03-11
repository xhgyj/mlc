# ============================================================
# Section 1: 环境设置与导入
# ============================================================

import sys
import os
import time
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# 设置环境变量
os.environ['TVM_HOME'] = '/home/lhy/tvm'
sys.path.insert(0, '/home/lhy/tvm/python')
sys.path.insert(0, '/home/lhy/tvm/ffi/python')

# 导入 TVM 和相关模块
import tvm
from tvm import IRModule, relax
from tvm.relax.frontend import nn
import tvm.relax.transform as transform

# 数据处理和可视化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 设置可视化样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 6)

# 验证 TVM 导入
print(f"TVM version: {tvm.__version__}")
print(f"TVM imported from: {tvm.__file__}")

# ============================================================
# Section 2: 工具函数定义
# ============================================================

def apply_passes(mod: IRModule, passes: List, verbose: bool = False) -> IRModule:
    """
    应用一系列 Pass 到模块
    
    Parameters
    ----------
    mod : IRModule
        输入模块
    passes : List
        Pass 列表（可以是 Pass 对象或函数）
    verbose : bool
        是否打印详细信息
    
    Returns
    -------
    IRModule
        变换后的模块
    """
    result = mod
    for i, pass_func in enumerate(passes):
        if verbose:
            pass_name = getattr(pass_func, '__name__', str(pass_func))
            print(f"  Applying pass {i+1}/{len(passes)}: {pass_name}")
        try:
            result = pass_func(result)
        except Exception as e:
            if verbose:
                print(f"    Warning: Pass failed with {type(e).__name__}: {e}")
            # 继续执行，某些 Pass 可能不适用于当前模块
    return result


def count_fused_kernels(mod: IRModule) -> Dict[str, int]:
    """
    统计模块中融合 kernel 的数量和类型
    
    Returns
    -------
    Dict[str, int]
        包含各种统计信息的字典
    """
    stats = {
        "total_functions": 0,
        "fused_functions": 0,
        "prim_functions": 0,
        "relax_functions": 0,
        "fused_names": []
    }
    
    for gvar, func in mod.functions_items():
        stats["total_functions"] += 1
        
        if hasattr(func, "attrs") and func.attrs is not None:
            if "prim_func_name" in func.attrs:
                stats["prim_functions"] += 1
        
        func_name = gvar.name_hint
        if "fused_" in func_name:
            stats["fused_functions"] += 1
            stats["fused_names"].append(func_name)
        
        if isinstance(func, relax.Function):
            stats["relax_functions"] += 1
    
    return stats


def print_module_stats(mod: IRModule, name: str = "Module"):
    """
    打印模块的统计信息
    """
    stats = count_fused_kernels(mod)
    print(f"\n{name} Statistics:")
    print(f"  Total functions: {stats['total_functions']}")
    print(f"  Relax functions: {stats['relax_functions']}")
    print(f"  TIR PrimFuncs: {stats['prim_functions']}")
    print(f"  Fused functions: {stats['fused_functions']}")
    if stats['fused_names']:
        print(f"  Fused kernel names: {', '.join(stats['fused_names'][:5])}")
        if len(stats['fused_names']) > 5:
            print(f"    ... and {len(stats['fused_names']) - 5} more")
    return stats


def benchmark_module(
    mod: IRModule,
    params: Dict[str, np.ndarray],
    data_shape: Tuple[int, ...],
    target: str = "llvm",
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: tvm.runtime.Device = None
) -> Dict[str, Any]:
    """
    对模块进行性能测试
    
    Parameters
    ----------
    mod : IRModule
        要测试的模块
    params : Dict[str, np.ndarray]
        模型参数
    data_shape : Tuple[int, ...]
        输入数据形状
    target : str
        编译目标 (如 "llvm", "cuda")
    num_iterations : int
        测试迭代次数
    warmup_iterations : int
        预热迭代次数
    device : tvm.runtime.Device
        运行设备（默认使用 CPU）
    
    Returns
    -------
    Dict[str, Any]
        包含性能指标的字典
    """
    if device is None:
        device = tvm.cpu()
    
    # 生成随机测试数据
    data_np = np.random.randn(*data_shape).astype(np.float32)
    
    # 构建模块
    build_start = time.time()
    try:
        ex = relax.build(mod, target=target)
        build_time = time.time() - build_start
    except Exception as e:
        return {
            "error": str(e),
            "build_failed": True
        }
    
    # 创建虚拟机
    vm = relax.VirtualMachine(ex, device)
    
    # 转换数据为 TVM NDArray
    data_nd = tvm.nd.array(data_np, device=device)
    params_nd = {k: tvm.nd.array(v, device=device) for k, v in params.items()}
    
    # 预热
    for _ in range(warmup_iterations):
        _ = vm["main"](data_nd, **params_nd)
    
    device.sync()
    
    # 性能测试
    times = []
    for _ in range(num_iterations):
        start = time.time()
        result = vm["main"](data_nd, **params_nd)
        device.sync()
        end = time.time()
        times.append(end - start)
    
    times = np.array(times)
    
    return {
        "avg_time_ms": np.mean(times) * 1000,
        "std_time_ms": np.std(times) * 1000,
        "min_time_ms": np.min(times) * 1000,
        "max_time_ms": np.max(times) * 1000,
        "median_time_ms": np.median(times) * 1000,
        "throughput": num_iterations / np.sum(times),
        "build_time": build_time,
        "build_failed": False
    }


print("Tool functions defined successfully!")
print("Available functions:")
print("  - apply_passes(): Apply a series of passes to a module")
print("  - count_fused_kernels(): Count fused kernels in a module")
print("  - print_module_stats(): Print module statistics")
print("  - benchmark_module(): Benchmark a module's performance")# ============================================================                                      
       # Section 3: 优化策略配置                                                                           
       # ============================================================                                      
                                                                                                           
       # 定义优化策略配置                                                                                  
       PASS_CONFIGS = {                                     
           "baseline": {
               "passes": [],
               "description": "No optimization (baseline)",
               "category": "Baseline"
           },

           "legalize": {
               "passes": [
                   transform.LegalizeOps()
               ],
               "description": "LegalizeOps only",
               "category": "Single Pass"
           },

           "canonicalize": {
               "passes": [
                   transform.LegalizeOps(),
                   transform.CanonicalizeBindings()
               ],
               "description": "LegalizeOps + CanonicalizeBindings",
               "category": "Single Pass"
           },

           "fold_constant": {
               "passes": [
                   transform.LegalizeOps(),
                   transform.FoldConstant()
               ],
               "description": "LegalizeOps + FoldConstant",
               "category": "Single Pass"
           },

           "dce": {
               "passes": [
                   transform.LegalizeOps(),
                   transform.DeadCodeElimination()
               ],
               "description": "LegalizeOps + DeadCodeElimination",
               "category": "Single Pass"
           },

           "fuse": {
               "passes": [
                   transform.LegalizeOps(),
                   transform.AnnotateTIROpPattern(),
                   transform.FuseOps(),
                   transform.FuseTIR()
               ],
               "description": "LegalizeOps + Fusion pipeline",
               "category": "Fusion"
           },

           "fuse_deep": {
               "passes": [
                   transform.LegalizeOps(),
                   transform.AnnotateTIROpPattern(),
                   transform.FuseOps(),
                   transform.FuseTIR()
               ],
               "description": "LegalizeOps + Fusion pipeline (deep)",
               "category": "Fusion"
           },

           "memory": {
               "passes": [
                   transform.LegalizeOps(),
               ],
               "description": "LegalizeOps only",
               "category": "Memory"
           },

           "simplify": {
               "passes": [
                   transform.LegalizeOps(),
                   transform.CanonicalizeBindings(),
                   transform.FoldConstant(),
                   #transform.SimplifyInference(),
               ],
               "description": "LegalizeOps + Simplify passes",
               "category": "Simplify"
           },

           "standard": {
               "passes": tvm.ir.transform.Sequential([
                   # 1. 降低高级算子
                   transform.LegalizeOps(),
                   # 2. 简化和规范化
                   transform.CanonicalizeBindings(),
                   transform.FoldConstant(),
                   #transform.SimplifyInference(),
                   # 3. 融合优化
                   transform.AnnotateTIROpPattern(),
                   transform.FuseOps(),
                   transform.FuseTIR(),
                   # 4. 死代码消除
                   transform.DeadCodeElimination(),
               ]),
               "description": "Standard optimization pipeline",
               "category": "Pipeline"
           },

           "cpu_optimized": {
               "passes": tvm.ir.transform.Sequential([
                   # 降低
                   transform.LegalizeOps(),
                   # 简化
                   transform.CanonicalizeBindings(),
                   transform.FoldConstant(),
                   # 融合
                   transform.AnnotateTIROpPattern(),
                   transform.FuseOps(),
                   transform.FuseTIR(),
               ]),
               "description": "CPU-optimized pipeline",
               "category": "Pipeline"
           },
       }

       # 打印配置摘要
       print(f"Defined {len(PASS_CONFIGS)} optimization strategies:\n")
       for name, config in PASS_CONFIGS.items():
           print(f"  [{name:15s}] {config['description']:40s} ({config['category']})")
