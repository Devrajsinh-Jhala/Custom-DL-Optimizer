# Custom-DL-Optimizer

[![PyPI](https://img.shields.io/pypi/v/custom-dl-optimizer.svg)](https://pypi.org/project/custom-dl-optimizer/)
[![Tests](https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/actions/workflows/tests.yml/badge.svg)](https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/actions/workflows/tests.yml)
[![Python](https://img.shields.io/pypi/pyversions/custom-dl-optimizer.svg)](https://pypi.org/project/custom-dl-optimizer/)
[![License: MIT](https://img.shields.io/badge/license-MIT-2f855a.svg)](LICENSE)

Custom-DL-Optimizer is a research-oriented PyTorch inference optimizer. It profiles candidate execution plans, rewrites supported graphs with PyTorch FX, optionally injects Triton kernels, and returns the fastest numerically valid plan for the supplied input signature.

[Documentation and project site](https://devrajsinh-jhala.github.io/Custom-DL-Optimizer/) | [Research notebook](Custom_DL_Optimizer_Research_Colab.ipynb) | [Benchmarks](docs/benchmarks.md)

> Status: alpha. The package is suitable for ML systems research and controlled inference experiments. It is not a replacement for TensorRT, Torch-TensorRT, TVM, XLA, or TorchInductor.

## What It Does

- Profiles representative operators with `torch.profiler`.
- Folds safe Conv2d-BatchNorm2d inference patterns.
- Rewrites supported module and functional ReLU operations with FX.
- Autotunes an optional Triton ReLU kernel when Triton is available.
- Applies CUDA FP16 autocast and channels-last layout when eligible.
- Optionally evaluates a TorchInductor candidate without nesting user Triton kernels.
- Checks output parity against eager FP32.
- Selects a custom plan only when it clears a configurable measured-speedup threshold.
- Supports positional and keyword model inputs, nested containers, CPU fallback, and structured reports.

## Installation

```bash
pip install custom-dl-optimizer
```

Install the vision examples separately:

```bash
pip install "custom-dl-optimizer[vision]"
```

Custom Triton kernels are enabled only when a compatible Triton installation is already available. The package does not independently upgrade Triton because PyTorch and Triton versions must remain compatible.

## Quick Start

```python
import torch
from torchvision.models import resnet50

from custom_dl_optimizer import AutoOptimizer, OptimizationConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
model = resnet50(weights=None).eval()
inputs = torch.randn(32, 3, 224, 224)

config = OptimizationConfig(
    enable_compile=torch.cuda.is_available(),
    compile_mode="max-autotune",
    min_speedup=1.02,
)
optimizer = AutoOptimizer(model, device=device, config=config)
optimized, report = optimizer.optimize_with_report(inputs)

# Preparing the layout once avoids conversion cost in a serving loop.
args, kwargs = optimizer.prepare_inputs(
    inputs,
    channels_last=report.channels_last,
)
with torch.inference_mode():
    outputs = optimized(*args, **kwargs)

print(report.selected_plan)
print(report.selection_reason)
print(report.as_dict())
```

The original one-call API remains supported:

```python
optimized = AutoOptimizer(model, device=device).optimize(inputs)
outputs = optimized(inputs)
```

## Selection Policy

For each input signature, the optimizer can evaluate:

1. Native PyTorch with eligible layout and precision settings.
2. FX graph surgery with Conv-BatchNorm folding and optional Triton replacement.
3. FX graph surgery plus TorchInductor when compilation is enabled.

Each candidate is checked against eager FP32. A custom candidate is selected only if it is valid and at least `min_speedup` faster than the native candidate during pilot measurement. Compilation and selection happen during `optimize()` and are not included in steady-state inference latency.

## API

### `OptimizationConfig`

Important fields include:

| Field | Default | Purpose |
| --- | ---: | --- |
| `enable_profiling` | `True` | Record representative operator timings |
| `enable_conv_bn_folding` | `True` | Fold safe evaluation-mode Conv-BN patterns |
| `enable_triton` | `True` | Use custom kernels when Triton is compatible |
| `enable_amp` | `True` | Enable CUDA FP16 autocast |
| `channels_last` | `True` | Evaluate channels-last for 4D CUDA tensors |
| `enable_compile` | `False` | Add a TorchInductor candidate |
| `compile_mode` | `"default"` | TorchInductor compilation mode |
| `verify_outputs` | `True` | Reject candidates that fail parity |
| `min_speedup` | `1.02` | Required gain over the native plan |
| `copy_model` | `True` | Avoid mutating the supplied module |

### `OptimizationReport`

The report records the selected plan, selection reason, operator profile, graph rewrite counts, candidate latency, numerical error, compilation setup time, and fallback warnings.

## Research Benchmark Snapshot

An earlier fixed-path research run on an NVIDIA Tesla T4 produced the following results. These values describe that notebook environment, not guaranteed package performance.

| Model | Batch | Eager FP32 | AMP/NHWC + Inductor | Experimental path | vs Eager | vs Inductor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-50 | 128 | 366.997 ms | 99.254 ms | 89.307 ms | 4.11x | 1.11x |
| MobileNet-V2 | 128 | 112.703 ms | 54.312 ms | 64.372 ms | 1.75x | 0.84x |
| VGG-16 | 128 | 639.399 ms | 273.326 ms | 273.402 ms | 2.34x | 1.00x |
| EfficientNet-B0 | 128 | 146.788 ms | 70.784 ms | 66.409 ms | 2.21x | 1.07x |
| DenseNet-121 | 128 | 360.750 ms | 194.997 ms | 193.169 ms | 1.87x | 1.01x |

MobileNet-V2 regressed under the fixed path. The package's new selection guard is designed to retain the native plan when a custom candidate does not demonstrate a sufficient pilot gain. Rerun the research notebook before using any number in a paper.

## Repository Layout

```text
custom_dl_optimizer/
  config.py              Optimization configuration
  report.py              Structured optimization reports
  core/
    engine.py            Candidate construction and selection
    profiler.py          Operator profiling
    graph_surgeon.py     FX rewrites and Conv-BN folding
    triton_kernels.py    Optional autotuned kernels
docs/                    Usage, benchmark, and research notes
examples/                Runnable package examples
tests/                   CPU-safe correctness tests
site/                    GitHub Pages landing page
```

## Development

```bash
git clone https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer.git
cd Custom-DL-Optimizer
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m build
python -m twine check dist/*
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [CHANGELOG.md](CHANGELOG.md) before submitting changes or publishing a release.

## Limitations

- FX symbolic tracing cannot capture every data-dependent Python program.
- Selection is specific to the supplied shapes, dtypes, device, and software stack.
- The package currently targets inference, not training or backward compilation.
- Standalone pointwise kernels may lose to fused native/compiler kernels.
- Dataset-level accuracy must be evaluated separately from tensor-output parity.

## Citation

Citation metadata is available in [CITATION.cff](CITATION.cff). Report the exact package version, GPU, CUDA, PyTorch, Triton, shapes, precision, warmup, iterations, and parity tolerance.

## License

Released under the [MIT License](LICENSE).
