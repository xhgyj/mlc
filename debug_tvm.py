
#!/usr/bin/env python3
"""TVM Debug Script - 诊断 Relax 运行时问题"""

import sys
import os
import traceback

# 设置环境
os.environ['TVM_HOME'] = '/home/lhy/tvm'
sys.path.insert(0, '/home/lhy/tvm/python')
sys.path.insert(0, '/home/lhy/tvm/ffi/python')

import numpy as np
import tvm
from tvm import relax
from tvm.relax.frontend import nn
import tvm.relax.transform as transform

print("=" * 60)
print("TVM 诊断脚本")
print("=" * 60)
print(f"TVM 版本: {tvm.__version__}")
print(f"TVM 路径: {tvm.__file__}")
print(f"LLVM 已启用: {tvm.runtime.enabled('llvm')}")
print()

# ========== 测试 1: 简单的 TVM 操作 ==========
print("[测试 1] 基础 TVM 数组操作...")
try:
    arr = tvm.nd.array(np.array([1, 2, 3], dtype="float32"))
    print("  ✓ TVM NDArray 创建成功")
    print(f"    值: {arr.numpy()}")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    traceback.print_exc()

print()

# ========== 测试 2: 创建简单的 Relax 模块 ==========
print("[测试 2] 创建简单的 Relax 模块...")
try:
    @tvm.script.ir_module
    class SimpleModule:
        @T.prim_func
        def main(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
            for i in range(4):
                B[i] = A[i] + 1.0

    print("  ✓ 简单 TensorIR 模块创建成功")
    # SimpleModule.show()
except Exception as e:
    print(f"  ✗ 失败: {e}")
    traceback.print_exc()

print()

# ========== 测试 3: 使用 nn.Module API ==========
print("[测试 3] 使用 nn.Module API 创建模型...")
try:
    class SimpleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)

        def forward(self, x):
            return self.fc(x)

    model = SimpleLinear()
    mod, params = model.export_tvm(
        {"forward": {"x": nn.spec.Tensor(("n", 10), "float32")}}
    )
    print("  ✓ 模型导出成功")
    print(f"    参数数量: {len(params)}")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    traceback.print_exc()
    mod, params = None, None

print()

# ========== 测试 4: LegalizeOps Pass ==========
if mod is not None:
    print("[测试 4] 应用 LegalizeOps Pass...")
    try:
        legalized = transform.LegalizeOps()(mod)
        print("  ✓ LegalizeOps 成功")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        traceback.print_exc()
        legalized = None

print()

# ========== 测试 5: 构建 Module (关键步骤) ==========
if mod is not None:
    print("[测试 5] 构建 Relax Module (使用 VirtualMachine)...")
    try:
        print("  步骤 1: 调用 relax.build()...")
        # 使用更简单的 target
        target = "llvm"
        print(f"    Target: {target}")

        ex = relax.build(mod, target=target)
        print("  ✓ relax.build() 成功")

        print("  步骤 2: 创建 VirtualMachine...")
        device = tvm.cpu()
        vm = relax.VirtualMachine(ex, device)
        print("  ✓ VirtualMachine 创建成功")

        print("  步骤 3: 准备输入数据...")
        data = np.random.randn(2, 10).astype("float32")
        data_nd = tvm.nd.array(data, device=device)
        params_nd = {k: tvm.nd.array(v, device=device) for k, v in params.items()}
        print(f"    输入形状: {data.shape}")
        print(f"    参数数量: {len(params_nd)}")

        print("  步骤 4: 执行推理...")
        result = vm["main"](data_nd, **params_nd)
        print("  ✓ 推理成功")
        print(f"    输出形状: {result.shape}")
        print(f"    输出值: {result.numpy()}")

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        traceback.print_exc()

print()

# ========== 测试 6: 测试优化的模块 ==========
if 'legalized' is not None and legalized is not None:
    print("[测试 6] 构建优化后的模块...")
    try:
        ex = relax.build(legalized, target="llvm")
        device = tvm.cpu()
        vm = relax.VirtualMachine(ex, device)

        data = np.random.randn(2, 10).astype("float32")
        data_nd = tvm.nd.array(data, device=device)
        params_nd = {k: tvm.nd.array(v, device=device) for k, v in params.items()}

        result = vm["main"](data_nd, **params_nd)
        print("  ✓ 优化模块构建和执行成功")
        print(f"    输出形状: {result.shape}")

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        traceback.print_exc()

print()

# ========== 诊断信息 ==========
print("=" * 60)
print("诊断建议")
print("=" * 60)
print()
print("如果遇到段错误 (segfault)，可能的原因：")
print()
print("1. LLVM 版本不兼容")
print("   解决: 重新编译 TVM 或使用预编译的 wheels")
print()
print("2. 内存不足")
print("   解决: 减小模型大小或 batch size")
print()
print("3. TVM 构建问题")
print("   解决: 重新安装 mlc-ai-nightly:")
print("       pip install mlc-ai-nightly -f https://mlc.ai/wheels --force-reinstall")
print()
print("4. Python/numpy 版本问题")
print("   解决: 确保 numpy 已正确安装")
print()

# 检查关键依赖
print("当前环境:")
print(f"  Python: {sys.version}")
try:
    import numpy as np
    print(f"  NumPy: {np.__version__}")
except:
    print("  NumPy: 未安装!")

try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
except:
    print("  PyTorch: 未安装")

print()
print("=" * 60)
