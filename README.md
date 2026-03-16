# 🥇 Custom DL Operator Profiler & Auto-Optimizer

**Custom-dl-optimizer** is a production-grade deep learning runtime optimization library. It bridges the gap between high-level model research and low-level hardware execution by performing automated **Graph Surgery** and **Custom Kernel Injection**.

By analyzing the "Arithmetic Intensity" of a model, this tool identifies bottlenecks and automatically optimizes them for NVIDIA GPUs, achieving significant speedups (1.5x - 4x) without requiring the developer to write a single line of CUDA or Triton.

---

## 🏗 Project Architecture & Roles

The system is divided into four distinct "System Engineer" roles:

| Component | File | Role |
| :--- | :--- | :--- |
| **The Profiler** | `profiler.py` | Uses `torch.profiler` to programmatically extract CUDA event times. It identifies whether a layer is **Compute-bound** (limited by math units) or **Memory-bound** (limited by VRAM bandwidth). |
| **The Surgeon** | `graph_surgeon.py` | Uses **PyTorch FX** to trace the model into an Intermediate Representation (IR). It performs pattern matching to find inefficient layers and replaces them with optimized alternatives. |
| **The Engine** | `triton_kernels.py` | Contains custom-written **OpenAI Triton** kernels. These bypass the standard PyTorch C++ overhead to execute math directly on the GPU SRAM using optimized block-level memory access. |
| **The Orchestrator**| `engine.py` | The user-facing API. It handles **Memory Layout Change (NHWC)** and **Auto-Mixed Precision (AMP)** to maximize Tensor Core utilization. |

---

## 📦 Installation

```bash
pip install custom-dl-optimizer
```

## Benchmarks
### Standard Resnet Test
This test compares a native PyTorch ResNet-18 against the Auto-Optimized version.
```python
import torch
import time
from torchvision.models import resnet18
from custom_dl_optimizer import AutoOptimizer

# Setup
device = "cuda"
inputs = torch.randn(128, 3, 224, 224).to(device)
model = resnet18().to(device).eval()

def benchmark(m, inp, label):
    # Warmup
    for _ in range(10): m(inp)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(50): 
        with torch.no_grad():
            m(inp)
    torch.cuda.synchronize()
    ms = ((time.time() - start) / 50) * 1000
    print(f"{label} Latency: {ms:.2f}ms")
    return ms

# Measure Baseline
baseline_ms = benchmark(model, inputs, "Standard PyTorch ResNet18")

# Optimize
optimizer = AutoOptimizer(model)
fast_model = optimizer.optimize(inputs)

# Measure Optimized
inputs_nhwc = inputs.to(memory_format=torch.channels_last)
fast_ms = benchmark(fast_model, inputs_nhwc, "Optimized ResNet18")

print(f"\n Total Improvement: {baseline_ms/fast_ms:.2f}x")
```

```text
Standard PyTorch ResNet18 Latency: 106.90ms

==================================================
🚀 STARTING DL-OPTIMIZER PIPELINE
==================================================
[Profiler] Running warm-up...
[Profiler] Tracing execution...
⚠️ [Profiler] Bottleneck Found: 'cudaDeviceSynchronize' is heavily taxing the hardware.
[Graph Surgeon] Tracing model into FX AST Graph...
✅ [Graph Surgeon] Injected custom Triton kernels in 9 locations.
[Memory Ops] Converting memory layout to Channels-Last (NHWC)...
[Precision] Wrapping model in FP16 Auto-Mixed Precision...
✅ PIPELINE COMPLETE. Model is ready for inference.

Optimized ResNet18 Latency: 52.02ms

🚀 Total Improvement: 2.05x
```