# Custom-DL-Optimizer

[![PyPI](https://img.shields.io/pypi/v/custom-dl-optimizer.svg)](https://pypi.org/project/custom-dl-optimizer/)
[![CI](https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/actions/workflows/tests.yml/badge.svg)](https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/actions/workflows/tests.yml)
[![Python](https://img.shields.io/pypi/pyversions/custom-dl-optimizer.svg)](https://pypi.org/project/custom-dl-optimizer/)
[![License: MIT](https://img.shields.io/badge/license-MIT-137a55.svg)](LICENSE)

Custom-DL-Optimizer is an auditable PyTorch inference plan selector. Give it a model and representative inputs; it profiles eligible execution plans, validates every candidate against eager FP32, and returns the fastest plan that clears your configured gain threshold.

[Project site](https://devrajsinh-jhala.github.io/Custom-DL-Optimizer/) | [Usage](docs/usage.md) | [Provider API](docs/providers.md) | [Agent toolkit](docs/agents.md) | [Research notebook](Custom_DL_Optimizer_Research_Colab.ipynb)

> Status: research alpha. Use it for controlled inference experiments and deployment feasibility checks. It complements rather than replaces TorchInductor, Torch-TensorRT, TensorRT, ONNX Runtime, TVM, or vendor profilers.

## Why It Exists

An optimization is useful only when it wins on the actual model, input signature, device, and software stack. A transformation that helps ResNet can regress MobileNet; compilation can reduce steady-state latency while adding material setup cost; reduced precision can be fast but numerically unacceptable.

Custom-DL-Optimizer makes those tradeoffs explicit:

- measures eager FP32, native AMP/layout, FX, and optional compiler candidates;
- supports external candidate providers for TensorRT, ONNX Runtime wrappers, private compilers, or research passes;
- rejects candidates that fail output structure or tolerance checks;
- records construction and lazy first-call time separately from steady-state latency;
- reports repeat samples, median, repeat-sample P90, standard deviation, and break-even calls;
- optionally selects by projected total cost for an expected request volume;
- reports speedup against eager FP32 and the native optimized path;
- falls back when custom work does not clear `min_speedup`;
- exports runtime provenance and results as JSON;
- exposes a closed, dependency-neutral tool surface for in-process agents.

## Installation

```bash
pip install custom-dl-optimizer
```

Vision examples require:

```bash
pip install "custom-dl-optimizer[vision]"
```

Triton and third-party compiler packages are runtime-detected. They are not force-installed because their versions must match the installed PyTorch/CUDA stack.

## Quick Start

Version 2 uses a result-oriented API:

```python
import torch
from torchvision.models import resnet50

from custom_dl_optimizer import OptimizationConfig, Optimizer

model = resnet50(weights=None).eval()
sample = torch.randn(8, 3, 224, 224)

optimizer = Optimizer(
    device="cuda" if torch.cuda.is_available() else "cpu",
    config=OptimizationConfig(
        enable_compile=torch.cuda.is_available(),
        compile_mode="max-autotune",
        expected_calls=50_000,
        min_speedup=1.02,
    ),
)
result = optimizer.optimize(model, sample)

with torch.inference_mode():
    output = result(sample)

print(result.selected_plan)
print(result.report.selection_reason)
result.save_report("artifacts/resnet50-optimization.json")
```

`OptimizationResult` owns both the callable selected module and the evidence used to select it. The selected wrapper prepares nested positional and keyword tensors for the target device and memory layout.

## Candidate Evidence

```python
for candidate in result.report.candidates:
    print(
        candidate.name,
        candidate.latency_ms,
        candidate.latency_p90_ms,
        candidate.first_call_time_s,
        candidate.projected_total_ms,
        candidate.break_even_calls_vs_baseline,
        candidate.speedup_vs_eager,
        candidate.speedup_vs_native,
        candidate.parity,
        candidate.error,
    )
```

Built-in plans are:

| Candidate | Purpose |
| --- | --- |
| `eager_fp32` | Correctness and performance reference |
| `native` | Eligible AMP and channels-last execution |
| `fx` | Safe Conv-BN folding and supported FX rewrites |
| `fx_inductor` | FX preparation followed by TorchInductor |

Only valid candidates participate in selection. Pilot measurements are useful for plan choice; use a full benchmark protocol for publication claims.

## Cold-Start-Aware Selection

Steady-state latency is the default selection basis. For a known serving horizon, set `expected_calls` to include candidate construction, lazy compilation on first invocation, and the remaining steady-state calls:

```python
config = OptimizationConfig(
    enable_compile=True,
    expected_calls=10_000,
    min_speedup=1.02,
)
```

The report exposes `selection_basis="projected_total_time"`, projected total milliseconds for each valid candidate, and its break-even call count against the fastest steady-state built-in baseline. This prevents a short-lived job from choosing a compiler whose build cost cannot be recovered.

## Compare Another Compiler

External backends implement the small `CandidateProvider` protocol. The package handles warmup, timing, parity, reporting, and selection:

```python
import torch
import torch_tensorrt  # Registers the torch.compile backend.

from custom_dl_optimizer import (
    FunctionCandidateProvider,
    OptimizationConfig,
    Optimizer,
)


def build_torch_tensorrt(model, context):
    return torch.compile(
        model,
        backend="torch_tensorrt",
        dynamic=context.config.dynamic_shapes,
    )


optimizer = Optimizer(
    device="cuda",
    config=OptimizationConfig(min_speedup=1.02),
    providers=(
        FunctionCandidateProvider(
            name="torch_tensorrt",
            builder=build_torch_tensorrt,
            availability=lambda context: context.device.type == "cuda",
        ),
    ),
)
result = optimizer.optimize(model, sample)
```

See [docs/providers.md](docs/providers.md) for isolation, input, correctness, and dependency rules. Torch-TensorRT officially supports use as a `torch.compile` backend; ONNX Runtime uses ordered execution providers and may require a wrapper that converts between PyTorch tensors and the session API.

## Agent Toolkit

Agents cannot safely transmit live modules or tensors through JSON. The host application therefore registers workloads in process, then exposes four bounded tools: inspect runtime, list workloads, optimize one workload, and read its report.

```python
from custom_dl_optimizer import OptimizationAgentToolkit, Optimizer

toolkit = OptimizationAgentToolkit(Optimizer(device="cuda"))
toolkit.register_workload(
    "resnet50-b8",
    model,
    sample,
    description="ResNet-50 inference, batch 8",
)

schemas = toolkit.tool_schemas()
response = toolkit.invoke(
    "custom_dl_optimize",
    {"workload": "resnet50-b8"},
)
```

The toolkit has no network client and never evaluates caller-provided code. See [docs/agents.md](docs/agents.md).

## Runtime Provenance

```python
capabilities = optimizer.inspect_runtime()
print(capabilities.as_dict())
```

Reports include Python and PyTorch versions, device name and type, CUDA/cuDNN versions, compute capability, and the availability of AMP, channels-last, Triton, and `torch.compile`.

## v1 Compatibility

`AutoOptimizer` remains available for one transition release, but new code should use `Optimizer`. The primary v2 contract returns one `OptimizationResult` rather than a `(module, report)` tuple. See [docs/migration-v2.md](docs/migration-v2.md).

## Historical Research Snapshot

An earlier fixed-path notebook run on an NVIDIA Tesla T4 produced the following hardware-specific results. These are not guaranteed package performance and are not a state-of-the-art claim.

| Model | Batch | Eager FP32 | Inductor | Experimental | vs Eager | vs Inductor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-50 | 128 | 366.997 ms | 99.254 ms | 89.307 ms | 4.11x | 1.11x |
| MobileNet-V2 | 128 | 112.703 ms | 54.312 ms | 64.372 ms | 1.75x | 0.84x |
| VGG-16 | 128 | 639.399 ms | 273.326 ms | 273.402 ms | 2.34x | 1.00x |
| EfficientNet-B0 | 128 | 146.788 ms | 70.784 ms | 66.409 ms | 2.21x | 1.07x |
| DenseNet-121 | 128 | 360.750 ms | 194.997 ms | 193.169 ms | 1.87x | 1.01x |

MobileNet-V2 is the important result: the experimental path regressed against Inductor. Version 2 measures and exposes that failure instead of assuming every rewrite is beneficial. Rerun the research notebook before citing any value.

## Development

```bash
git clone https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer.git
cd Custom-DL-Optimizer
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check custom_dl_optimizer tests examples tools
python -m build
python -m twine check dist/*
```

Read [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [CHANGELOG.md](CHANGELOG.md) before contributing or publishing.

## Limitations

- Selection is specific to the supplied shapes, dtypes, device, and software versions.
- FX symbolic tracing cannot represent every data-dependent Python program.
- Provider setup may require third-party dependencies and substantial compilation time.
- Tensor parity is not a substitute for dataset-level quality evaluation.
- The current package targets inference; it does not optimize training or backward graphs.
- GPU energy, peak memory, and multi-stream throughput require separate measurement.

## Citation and License

Citation metadata is in [CITATION.cff](CITATION.cff). Report the package version, hardware, CUDA, PyTorch, candidate providers, shapes, precision, warmup, iterations, and tolerances.

Released under the [MIT License](LICENSE).
