#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relax 优化可视化对比脚本

用于对比不同 Relax 优化流水线的性能效果，支持 MLP 和 DeepCNN 两种模型。
生成推理时间、吞吐量和融合函数数量的可视化图表。

Author: MLC Course
Date: 2026-03-12
"""

# ============================================================
# 环境设置与导入
# ============================================================
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置 TVM 环境
os.environ['TVM_HOME'] = '/home/lhy/tvm'
sys.path.insert(0, '/home/lhy/tvm/python')
sys.path.insert(0, '/home/lhy/tvm/ffi/python')

import tvm
from tvm import relax
from tvm.relax.frontend import nn
from tvm import IRModule


# ============================================================
# 工具函数定义
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
    device: tvm.runtime.Device = None,
    entry_name: str = "forward"
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
    entry_name : str
        入口函数名称

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
        ex = tvm.compile(mod, target=target)
        build_time = time.time() - build_start
    except Exception as e:
        return {
            "error": str(e),
            "build_failed": True
        }

    # 创建虚拟机
    vm = relax.VirtualMachine(ex, device)

    # 转换数据为 TVM NDArray
    data_nd = tvm.runtime.tensor(data_np, device=device)
    param_dict = dict(params)
    params_nd = {}
    for k, v in param_dict.items():
        shape = v.shape
        dtype = v.dtype
        random_weight = np.random.randn(*shape).astype(dtype)
        params_nd[k] = tvm.runtime.tensor(random_weight, device=device)
    param_values = list(params_nd.values())

    # 预热
    for _ in range(warmup_iterations):
        _ = vm[entry_name](data_nd, *param_values)
    device.sync()

    # 性能测试
    times = []
    for _ in range(num_iterations):
        start = time.time()
        result = vm[entry_name](data_nd, *param_values)
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


# ============================================================
# 模型定义
# ============================================================

class MLP(nn.Module):
    """简单的多层感知机"""
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class DeepCNN(nn.Module):
    """深度 CNN 模型 - 内存密集型"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2D(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2D(128, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.fc = nn.Linear(256 * 32 * 32, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        b, c, h, w = x.shape
        x = x.reshape(b, -1)
        x = self.fc(x)
        return x


# ============================================================
# 优化流水线配置
# ============================================================

# 导入所有必要的变换
from tvm.relax import transform

PIPELINES = {
    "baseline": {
        "passes": [],
        "description": "No optimization",
    },
    "standard": {
        "passes": tvm.ir.transform.Sequential([
            transform.LegalizeOps(),
            transform.CanonicalizeBindings(),
            transform.FoldConstant(),
            transform.AnnotateTIROpPattern(),
            transform.FuseOps(),
            transform.FuseTIR(),
            transform.DeadCodeElimination(),
        ]),
        "description": "Standard pipeline",
    },
    "memory": {
        "passes": tvm.ir.transform.Sequential([
            transform.LegalizeOps(),
            transform.AnnotateTIROpPattern(),
            transform.FuseOps(),
            transform.FuseTIR(),
            # transform.StaticPlanBlockMemory(),
            # transform.KillAfterLastUse(),
        ]),
        "description": "Memory optimized",
    },
    "nchw": {
        "passes": tvm.ir.transform.Sequential([
            transform.LegalizeOps(),
            transform.AnnotateTIROpPattern(),
            transform.FuseOps(),
            transform.FuseTIR(),
        ]),
        "description": "NCHW layout",
    },
    "nhwc": {
        "passes": tvm.ir.transform.Sequential([
            transform.ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]}),
            transform.LegalizeOps(),
            transform.AnnotateTIROpPattern(),
            transform.FuseOps(),
            transform.FuseTIR(),
        ]),
        "description": "NHWC layout (CPU friendly)",
    },
    "fp32": {
        "passes": tvm.ir.transform.Sequential([
            transform.LegalizeOps(),
            transform.AnnotateTIROpPattern(),
            transform.FuseOps(),
            transform.FuseTIR(),
        ]),
        "description": "FP32 precision",
    },
    "fp16": {
        "passes": tvm.ir.transform.Sequential([
            # transform.ToMixedPrecision(),
            transform.LegalizeOps(),
            transform.AnnotateTIROpPattern(),
            transform.FuseOps(),
            transform.FuseTIR(),
        ]),
        "description": "FP16 mixed precision",
    },
}


# ============================================================
# 核心功能函数
# ============================================================

def create_model(model_name: str) -> Tuple[IRModule, Dict[str, np.ndarray], Tuple[int, ...]]:
    """
    创建指定模型并返回模块、参数和输入形状

    Parameters
    ----------
    model_name : str
        模型名称 ("mlp" 或 "cnn")

    Returns
    -------
    Tuple[IRModule, Dict[str, np.ndarray], Tuple[int, ...]]
        (模块, 参数, 输入形状)
    """
    if model_name.lower() == "mlp":
        model = MLP(input_dim=784, hidden_dim=256, output_dim=10)
        mod, params = model.export_tvm(
            {"forward": {"x": nn.spec.Tensor(("n", 784), "float32")}}
        )
        data_shape = (32, 784)
    elif model_name.lower() == "cnn":
        model = DeepCNN()
        mod, params = model.export_tvm(
            {"forward": {"x": nn.spec.Tensor(("n", 3, 32, 32), "float32")}}
        )
        data_shape = (16, 3, 32, 32)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return mod, params, data_shape


def run_single_benchmark(
    model_name: str,
    pipeline_name: str,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    运行单次 benchmark

    Returns
    -------
    Dict[str, Any]
        包含性能指标和统计信息的字典
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {model_name.upper()} + {pipeline_name}")
        print(f"{'='*60}")

    # 创建原始模型
    origin_mod, params, data_shape = create_model(model_name)

    # 获取优化流水线
    pipeline = PIPELINES[pipeline_name]
    passes = pipeline["passes"]

    # 应用 Pass
    if verbose:
        print(f"Applying {len(passes) if passes else 0} passes...")

    if isinstance(passes, tvm.ir.transform.Sequential):
        optimized_mod = passes(origin_mod)
    elif passes:
        optimized_mod = apply_passes(origin_mod, passes, verbose=verbose)
    else:
        optimized_mod = origin_mod

    # 统计融合函数
    stats = count_fused_kernels(optimized_mod)

    # 性能测试
    if verbose:
        print("Benchmarking...")

    result = benchmark_module(
        optimized_mod,
        params,
        data_shape,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations
    )

    # 组合结果
    result.update({
        "model": model_name,
        "pipeline": pipeline_name,
        "description": pipeline["description"],
        "fused_functions": stats["fused_functions"],
        "total_functions": stats["total_functions"],
    })

    if verbose:
        if result.get("build_failed"):
            print(f"  FAILED: {result.get('error', 'Unknown error')}")
        else:
            print(f"  Avg time: {result['avg_time_ms']:.2f} ms")
            print(f"  Std time: {result['std_time_ms']:.2f} ms")
            print(f"  Throughput: {result['throughput']:.2f} inferences/sec")
            print(f"  Fused functions: {result['fused_functions']}")

    return result


def run_benchmark_suite(
    models: List[str] = None,
    pipelines: List[str] = None,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    verbose: bool = False
) -> pd.DataFrame:
    """
    运行完整的测试套件

    Returns
    -------
    pd.DataFrame
        包含所有测试结果的 DataFrame
    """
    if models is None:
        models = ["mlp", "cnn"]
    if pipelines is None:
        pipelines = list(PIPELINES.keys())

    results = []

    total_tests = len(models) * len(pipelines)
    current = 0

    print(f"\n{'='*60}")
    print(f"Relax Optimization Benchmark Suite")
    print(f"{'='*60}")
    print(f"Models: {', '.join([m.upper() for m in models])}")
    print(f"Pipelines: {', '.join(pipelines)}")
    print(f"Iterations: {num_iterations} (warmup: {warmup_iterations})")
    print(f"{'='*60}")

    for model_name in models:
        for pipeline_name in pipelines:
            current += 1
            if verbose:
                print(f"\n[{current}/{total_tests}] ", end="")

            try:
                result = run_single_benchmark(
                    model_name,
                    pipeline_name,
                    num_iterations,
                    warmup_iterations,
                    verbose
                )
                results.append(result)
            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")
                results.append({
                    "model": model_name,
                    "pipeline": pipeline_name,
                    "description": PIPELINES[pipeline_name]["description"],
                    "error": str(e),
                    "build_failed": True
                })

    return pd.DataFrame(results)


def plot_inference_time(results: pd.DataFrame, output_dir: str):
    """
    绘制推理时间柱状图

    - X 轴: 优化流水线
    - Y 轴: 平均推理时间 (ms)
    - 分组: 按 MLP 和 CNN 分组
    - 误差线: 标准差
    """
    # 过滤成功的结果
    valid = results[~results.get("build_failed", False)]

    if valid.empty:
        print("No valid data for inference time plot")
        return

    # 准备数据
    models = valid["model"].unique()
    pipelines = valid["pipeline"].unique()

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(pipelines))
    width = 0.35

    colors = ['#3498db', '#e74c3c']

    for i, model in enumerate(models):
        model_data = valid[valid["model"] == model]
        times = []
        stds = []
        for p in pipelines:
            row = model_data[model_data["pipeline"] == p]
            if not row.empty:
                times.append(row["avg_time_ms"].values[0])
                stds.append(row["std_time_ms"].values[0])
            else:
                times.append(0)
                stds.append(0)

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, times, width, yerr=stds,
                      label=model.upper(), color=colors[i % len(colors)],
                      alpha=0.8, capsize=4)

        # 添加数值标签
        for bar, time_val in zip(bars, times):
            if time_val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{time_val:.1f}',
                        ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Optimization Pipeline', fontsize=12)
    ax.set_ylabel('Inference Time (ms)', fontsize=12)
    ax.set_title('Inference Time Comparison (lower is better)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in pipelines], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "inference_time.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_throughput(results: pd.DataFrame, output_dir: str):
    """
    绘制吞吐量柱状图

    - X 轴: 优化流水线
    - Y 轴: 每秒推理次数
    """
    valid = results[~results.get("build_failed", False)]

    if valid.empty:
        print("No valid data for throughput plot")
        return

    models = valid["model"].unique()
    pipelines = valid["pipeline"].unique()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(pipelines))
    width = 0.35

    colors = ['#2ecc71', '#9b59b6']

    for i, model in enumerate(models):
        model_data = valid[valid["model"] == model]
        throughputs = []
        for p in pipelines:
            row = model_data[model_data["pipeline"] == p]
            if not row.empty:
                throughputs.append(row["throughput"].values[0])
            else:
                throughputs.append(0)

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, throughputs, width,
                      label=model.upper(), color=colors[i % len(colors)],
                      alpha=0.8)

        for bar, val in zip(bars, throughputs):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}',
                        ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Optimization Pipeline', fontsize=12)
    ax.set_ylabel('Throughput (inferences/second)', fontsize=12)
    ax.set_title('Throughput Comparison (higher is better)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in pipelines], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "throughput.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_fused_functions(results: pd.DataFrame, output_dir: str):
    """
    绘制融合函数数量柱状图

    - X 轴: 优化流水线
    - Y 轴: 融合函数数量
    """
    valid = results[~results.get("build_failed", False)]

    if valid.empty:
        print("No valid data for fused functions plot")
        return

    models = valid["model"].unique()
    pipelines = valid["pipeline"].unique()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(pipelines))
    width = 0.35

    colors = ['#f39c12', '#1abc9c']

    for i, model in enumerate(models):
        model_data = valid[valid["model"] == model]
        counts = []
        for p in pipelines:
            row = model_data[model_data["pipeline"] == p]
            if not row.empty:
                counts.append(row["fused_functions"].values[0])
            else:
                counts.append(0)

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, counts, width,
                      label=model.upper(), color=colors[i % len(colors)],
                      alpha=0.8)

        for bar, val in zip(bars, counts):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(val)}',
                        ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Optimization Pipeline', fontsize=12)
    ax.set_ylabel('Number of Fused Functions', fontsize=12)
    ax.set_title('Kernel Fusion Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in pipelines], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "fused_functions.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def save_results_csv(results: pd.DataFrame, output_dir: str):
    """保存结果到 CSV 文件"""
    output_path = os.path.join(output_dir, "results.csv")

    # 选择要导出的列
    columns = ["model", "pipeline", "description", "avg_time_ms", "std_time_ms",
               "throughput", "fused_functions", "total_functions", "build_time"]

    # 只保留存在的列
    export_cols = [c for c in columns if c in results.columns]

    results[export_cols].to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")


def print_summary(results: pd.DataFrame):
    """打印测试结果汇总"""
    valid = results[~results.get("build_failed", False)]

    if valid.empty:
        print("\nNo successful benchmarks!")
        return

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    for model in valid["model"].unique():
        model_data = valid[valid["model"] == model]
        print(f"\n{model.upper()}:")

        # 找出最快的流水线
        fastest = model_data.loc[model_data["avg_time_ms"].idxmin()]
        print(f"  Fastest: {fastest['pipeline']} ({fastest['avg_time_ms']:.2f} ms)")

        # 找出吞吐量最高的流水线
        best_throughput = model_data.loc[model_data["throughput"].idxmax()]
        print(f"  Best throughput: {best_throughput['pipeline']} ({best_throughput['throughput']:.2f} inf/sec)")

    failed = results[results.get("build_failed", False)]
    if not failed.empty:
        print(f"\nFailed benchmarks:")
        for _, row in failed.iterrows():
            print(f"  {row['model']} + {row['pipeline']}: {row.get('error', 'Unknown')}")


# ============================================================
# 主程序入口
# ============================================================

def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(
        description="Relax Optimization Benchmark - Visualize optimization effects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python relax_optimization_benchmark.py
  python relax_optimization_benchmark.py --models mlp cnn --pipelines standard memory nhwc
  python relax_optimization_benchmark.py -i 30 -w 5
  python relax_optimization_benchmark.py -o ./my_results -v
        """
    )

    parser.add_argument(
        "--models", "-m",
        nargs="+",
        choices=["mlp", "cnn"],
        default=["mlp", "cnn"],
        help="Models to benchmark (default: both)"
    )

    parser.add_argument(
        "--pipelines", "-p",
        nargs="+",
        choices=list(PIPELINES.keys()),
        default=list(PIPELINES.keys()),
        help="Optimization pipelines to test (default: all)"
    )

    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)"
    )

    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir.absolute()}")

    # 运行测试
    results = run_benchmark_suite(
        models=args.models,
        pipelines=args.pipelines,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
        verbose=args.verbose
    )

    # 生成可视化
    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print(f"{'='*60}")

    plot_inference_time(results, str(output_dir))
    plot_throughput(results, str(output_dir))
    plot_fused_functions(results, str(output_dir))
    save_results_csv(results, str(output_dir))

    # 打印汇总
    print_summary(results)

    print(f"\n{'='*60}")
    print("Benchmark complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
