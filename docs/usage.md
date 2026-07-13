# Usage Guide

## Basic Optimization

```python
import torch
from torchvision.models import resnet50
from custom_dl_optimizer import AutoOptimizer, OptimizationConfig

model = resnet50(weights=None).eval()
inputs = torch.randn(32, 3, 224, 224)

config = OptimizationConfig(
    enable_compile=torch.cuda.is_available(),
    compile_mode="max-autotune",
)
optimizer = AutoOptimizer(model, config=config)
optimized, report = optimizer.optimize_with_report(inputs)

args, kwargs = optimizer.prepare_inputs(
    inputs,
    channels_last=report.channels_last,
)
with torch.inference_mode():
    output = optimized(*args, **kwargs)
```

`optimize()` keeps the original one-call behavior and stores the same report on `optimizer.last_report`.

## Multi-Input Models

Pass example positional and keyword inputs exactly as the model receives them:

```python
optimized = optimizer.optimize(left, right, attention_mask=mask)
output = optimized(left, right, attention_mask=mask)
```

Nested tuples, lists, and dictionaries are moved to the selected device recursively.

## Reading the Report

```python
print(report.selected_plan)
print(report.selection_reason)

for candidate in report.candidates:
    print(candidate.name, candidate.latency_ms, candidate.parity, candidate.error)

print(report.graph.conv_bn_fusions)
print(report.graph.total_rewrites)
```

The candidate measurements are short pilot timings used for plan selection. Use a separate benchmark protocol for reported paper numbers.

## Compilation

TorchInductor is opt-in because compilation can add substantial setup latency:

```python
config = OptimizationConfig(
    enable_compile=True,
    compile_mode="reduce-overhead",  # Or "default" / "max-autotune".
    dynamic_shapes=False,
)
```

User-defined Triton activation kernels are not nested inside the Inductor candidate. The compiler candidate receives Conv-BN folding but leaves pointwise fusion to Inductor.

## Correctness

Output parity is enabled by default. Candidates that fail `torch.allclose` with the configured tolerances are rejected.

```python
config = OptimizationConfig(rtol=5e-2, atol=5e-2, verify_outputs=True)
```

Tensor parity does not replace dataset-level accuracy evaluation.

## CPU Behavior

CPU execution disables CUDA AMP and channels-last selection automatically. FX rewrites remain available, and Triton modules fall back to native PyTorch operations.
