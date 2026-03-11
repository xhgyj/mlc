# 环境设置
import sys
import os
import time
import numpy as np
from typing import Dict, List, Any, Tuple

# 设置环境变量
os.environ['TVM_HOME'] = '/home/lhy/tvm'
sys.path.insert(0, '/home/lhy/tvm/python')
sys.path.insert(0, '/home/lhy/tvm/ffi/python')

# 导入 TVM 和相关模块
import tvm
from tvm import IRModule, relax
from tvm.relax.frontend import nn
import tvm.relax.transform as transform
#from tvm import ndarray as nd

# 验证导入
print(f"TVM version: {tvm.__version__}")
print("Environment setup complete!")
print(f"NumPy: {np.__version__}")
import tvm
import torch



def apply_passes(mod: IRModule, passes: List, verbose: bool = False) -> IRModule:
    """应用一系列 Pass 到模块"""
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


def count_fused_kernels(mod: IRModule) -> Dict[str, Any]:
    """统计模块中融合 kernel 的数量和类型"""
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
    """打印模块的统计信息"""
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
    num_iterations: int = 10,
    warmup_iterations: int = 10
) -> Dict[str, Any]:
    """对模块进行性能测试"""
    device = tvm.cpu()
    
    # 生成随机测试数据
    data_np = np.random.randn(*data_shape).astype(np.float32)
    
    # 构建模块
    build_start = time.time()
    try:
        ex = tvm.compile(mod, target=target)
        build_time = time.time() - build_start
    except Exception as e:
        return {"error": str(e), "build_failed": True}
    
    # 创建虚拟机
    vm = relax.VirtualMachine(ex, device)
    
    # 转换数据为 TVM NDArray
    data_nd = tvm.nd.array(data_np, device=device)
    params_nd = {k: tvm.nd.array(v, device=device) for k, v in params.items()}
    
    # 预热
    for _ in range(warmup_iterations):
        _ = vm["forward"](data_nd, **params_nd)
    device.sync()
    
    # 性能测试
    times = []
    for _ in range(num_iterations):
        start = time.time()
        result = vm["forward"](data_nd, **params_nd)
        device.sync()
        end = time.time()
        times.append(end - start)
    
    times = np.array(times)
    
    return {
        "avg_time_ms": np.mean(times) * 1000,
        "std_time_ms": np.std(times) * 1000,
        "min_time_ms": np.min(times) * 1000,
        "max_time_ms": np.max(times) * 1000,
        "throughput": num_iterations / np.sum(times),
        "build_time": build_time,
        "build_failed": False
    }


print("Tool functions defined successfully!")

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

# 创建并导出模型
model = MLP()
origin_mod, params = model.export_tvm(
    {"forward": {"x": nn.spec.Tensor(("n", 784), "float32")}}
)
#print(params)

print("Original MLP Model:")
#origin_mod.show()
print_module_stats(origin_mod, "Original")

# 定义标准优化流水线
standard_pipeline = tvm.ir.transform.Sequential([
    # 1. 降低高级算子
    transform.LegalizeOps(),
    # 2. 简化和规范化
    transform.CanonicalizeBindings(),
    transform.FoldConstant(),
    # 3. 融合优化
    transform.AnnotateTIROpPattern(),
    transform.FuseOps(),
    transform.FuseTIR(),
    # 4. 死代码消除
    transform.DeadCodeElimination(),
])

# 应用优化流水线
optimized_mod = standard_pipeline(origin_mod)

print("\nOptimized Module:")
#optimized_mod.show()
print_module_stats(optimized_mod, "Optimized")

#单次执行对比
device = tvm.cpu()
    
# 生成随机测试数据
data_np = np.random.randn(32, 784).astype(np.float32)
    
# 构建模块
build_start = time.time()
    
ex = tvm.compile(origin_mod, target="llvm")
build_time = time.time() - build_start
    
    
# 创建虚拟机
vm = relax.VirtualMachine(ex, device)
    
# 转换数据为 TVM NDArray
data_nd = tvm.runtime.tensor(data_np,device)
param_dict = dict(params)
params_nd = {}
for k, v in param_dict.items():
    # 获取原始 Tensor 的形状和数据类型
    shape = v.shape
    dtype = v.dtype
    
    # 使用 numpy 生成正态分布的随机权重，并确保类型一致
    random_weight = np.random.randn(*shape).astype(dtype)
    
    # 将生成的随机 numpy 数组转换为 TVM 的 NDArray，并放进设备中
    params_nd[k] = tvm.runtime.tensor(random_weight, device=device)
    
    #print(type(k),type(v))
print(params_nd)
#params_nd = {k: tvm.runtime.tensor(v, device=device) for k, v in param_dict.items()}
param_values = list(params_nd.values())
# 预热
for _ in range(10):
    _ = vm["forward"](data_nd, *param_values) 
device.sync()
start = time.time()
result = vm["forward"](data_nd, *param_values)
device.sync()
end = time.time()
#times.append(end - start)
print(result)