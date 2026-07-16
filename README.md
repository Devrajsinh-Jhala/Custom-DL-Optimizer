# Custom-DL-Optimizer

[![PyPI](https://img.shields.io/pypi/v/custom-dl-optimizer.svg)](https://pypi.org/project/custom-dl-optimizer/)
[![CI](https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/actions/workflows/tests.yml/badge.svg)](https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/actions/workflows/tests.yml)
[![Python](https://img.shields.io/pypi/pyversions/custom-dl-optimizer.svg)](https://pypi.org/project/custom-dl-optimizer/)
[![License: MIT](https://img.shields.io/badge/license-MIT-137a55.svg)](https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/blob/master/LICENSE)

Custom-DL-Optimizer v3 is an auditable, workload-aware PyTorch inference plan selector. Give it a model and representative serving inputs; it builds eligible execution plans, validates every candidate against eager FP32, and replaces the native baseline only when a confidence-bounded speedup clears the deployment policy.

[Project site](https://devrajsinh-jhala.github.io/Custom-DL-Optimizer/) | [Usage](https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/blob/master/docs/usage.md) | [Provider API](https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/blob/master/docs/providers.md) | [Research protocol](https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/blob/master/docs/paper-launch.md) | [Research notebook](https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/blob/master/Custom_DL_Optimizer_Research_Colab.ipynb)

> Status: research beta. Use it for controlled inference experiments and deployment qualification. It orchestrates and evaluates PyTorch and provider plans; it does not replace their compiler or vendor profiling stacks.

## Why It Exists

An optimization is useful only when it wins on the actual model, input signature, device, and software stack. A transformation that helps ResNet can regress MobileNet; compilation can reduce steady-state latency while adding material setup cost; reduced precision can be fast but numerically unacceptable.

Custom-DL-Optimizer makes those tradeoffs explicit:

- measures eager FP32, native AMP/layout, FX, and optional compiler candidates;
- supports external candidate providers for TensorRT, ONNX Runtime wrappers, private compilers, or research passes;
- rejects candidates that fail output structure or tolerance checks;
- validates every candidate across weighted input shapes, batches, and keyword signatures;
- records construction and lazy first-call time separately from steady-state latency;
- reports request samples, median, P90/P95/P99, bootstrap mean bounds, incremental peak CUDA memory, and break-even calls;
- selects by steady-state mean or projected lifecycle cost for an expected request volume;
- rejects plans that exceed configured setup, first-call, or memory limits;
- persists content-addressed decisions and revalidates cached winners before reuse;
- includes optional Torch-TensorRT, ONNX Runtime, and TorchAO providers;
- reports speedup against eager FP32 and the native optimized path;
- retains the native baseline when a challenger's upper confidence bound does not clear `min_speedup` against the baseline's lower bound;
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

Optional integration dependencies are deliberately separate:

```bash
pip install "custom-dl-optimizer[onnxruntime-gpu]"
pip install "custom-dl-optimizer[quantization]"
```

Triton and third-party compiler packages are runtime-detected. They are not force-installed because their versions must match the installed PyTorch/CUDA stack.

## Quick Start

Version 3 separates target, measurement, validation, and deployment policy:

```python
import torch
from torchvision.models import resnet50

from custom_dl_optimizer import (
    DeploymentConstraints,
    ExecutionTarget,
    InferenceOptimizer,
    MeasurementPolicy,
    OptimizationPolicy,
)

model = resnet50(weights=None).eval()
sample = torch.randn(8, 3, 224, 224)

optimizer = InferenceOptimizer(
    target=ExecutionTarget("cuda" if torch.cuda.is_available() else "cpu"),
    policy=OptimizationPolicy(
        objective="lifecycle_latency",
        enable_compile=torch.cuda.is_available(),
        compile_mode="max-autotune",
        measurement=MeasurementPolicy(iterations=20, repeats=3),
        constraints=DeploymentConstraints(
            expected_calls=50_000,
            min_speedup=1.02,
        ),
    ),
)
decision = optimizer.select_signature(model, sample)

with torch.inference_mode():
    output = decision(sample)

print(decision.selected_plan)
print(decision.report.selection_reason)
decision.save_report("artifacts/resnet50-optimization.json")
```

`OptimizationDecision` owns the selected `BuiltPlan` and the evidence used to select it. Its runner can be a PyTorch module or another callable runtime adapter. The selected wrapper prepares nested positional and keyword tensors for the target device and memory layout.

## Workload-Aware Selection

Use a weighted profile when production traffic contains multiple signatures:

```python
from custom_dl_optimizer import WorkloadCase, WorkloadProfile

profile = WorkloadProfile(
    name="image-serving",
    expected_calls=100_000,
    cases=(
        WorkloadCase("batch-1", args=(batch_1,), weight=70),
        WorkloadCase("batch-8", args=(batch_8,), weight=25),
        WorkloadCase("batch-32", args=(batch_32,), weight=5),
    ),
)
decision = optimizer.select(model, profile)
```

Each candidate must pass parity for every case. Selection uses the normalized traffic weights, so a backend cannot win by optimizing an unrepresentative shape.

## Candidate Evidence

```python
for candidate in decision.report.candidates:
    print(
        candidate.name,
        candidate.latency_ms,
        candidate.latency_p90_ms,
        candidate.latency_p99_ms,
        candidate.latency_ci95_low_ms,
        candidate.latency_ci95_high_ms,
        candidate.selection_cost_ms,
        candidate.selection_cost_ci_low_ms,
        candidate.selection_cost_ci_high_ms,
        candidate.confidence_gate_passed,
        candidate.peak_memory_mb,
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

Resource policies are optional:

```python
policy = OptimizationPolicy(
    constraints=DeploymentConstraints(
        max_setup_time_s=30,
        max_first_call_time_s=10,
        max_peak_memory_mb=2048,
    ),
)
```

## Cold-Start-Aware Selection

Steady-state latency is the default selection basis. For a known serving horizon, set `expected_calls` to include candidate construction, lazy compilation on first invocation, and the remaining steady-state calls:

```python
policy = OptimizationPolicy(
    objective="lifecycle_latency",
    enable_compile=True,
    constraints=DeploymentConstraints(expected_calls=10_000, min_speedup=1.02),
)
```

The report exposes `selection_basis="projected_total_time"`, lifecycle confidence bounds, projected total milliseconds, and break-even calls. This prevents a short-lived job from choosing a compiler whose build cost cannot be recovered.

## Persistent Plan Cache

Enable a content-addressed cache to avoid re-running the full candidate race at every process start:

```python
policy = OptimizationPolicy(
    plan_cache_dir=".custom-dl-cache",
    reuse_cached_plan=True,
)
```

The key includes model weights, workload signatures and weights, selection policy, provider settings, and runtime provenance. A cache hit still runs output parity, constraint checks, and a short latency-regression probe before the plan is accepted.

## Compare Other Runtimes

First-party optional providers use the same policy as built-in plans:

```python
from custom_dl_optimizer import (
    ExecutionTarget,
    ONNXRuntimeProvider,
    InferenceOptimizer,
    TorchAOQuantizationProvider,
    TorchTensorRTProvider,
)

optimizer = InferenceOptimizer(
    target=ExecutionTarget("cuda"),
    providers=(
        TorchTensorRTProvider(
            compile_options={"optimization_level": 3},
        ),
        ONNXRuntimeProvider(),
        TorchAOQuantizationProvider(scheme="int8_weight_only"),
    ),
)
decision = optimizer.select_signature(model, sample)
```

Unavailable optional dependencies are reported without breaking the remaining race. Custom, mobile, NPU, and private backends implement `ExecutionProvider` and return a `BuiltPlan`. Availability is explicit, so an absent vendor SDK cannot masquerade as a valid candidate. See the [provider documentation](https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/blob/master/docs/providers.md).

## CLI and Decision Bundles

```bash
custom-dl-optimizer inspect --device cuda
custom-dl-optimizer report artifacts/report.json
custom-dl-optimizer cache list --cache-dir .custom-dl-cache
custom-dl-optimizer paper-export artifacts/run-*/report.json \
  --output-dir artifacts/paper
```

`decision.save_bundle("artifacts/run")` writes a schema-v3 report and portable decision manifest. Executable engine serialization remains provider-owned so the package never pretends a generic Python module is a deployable engine artifact.

`paper-export` preserves raw samples in candidate-level and workload-case CSV files, writes a LaTeX results table and provenance manifest, and produces median/P99 figures when the `research` extra is installed. These are reporting artifacts, not a substitute for the controlled protocol in [paper-launch.md](https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/blob/master/docs/paper-launch.md).

## Agent Toolkit

Agents cannot safely transmit live modules or tensors through JSON. The host application therefore registers workloads in process, then exposes four bounded tools: inspect runtime, list workloads, optimize one workload, and read its report.

```python
from custom_dl_optimizer import ExecutionTarget, InferenceOptimizer, OptimizationAgentToolkit

toolkit = OptimizationAgentToolkit(
    InferenceOptimizer(target=ExecutionTarget("cuda"))
)
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

The toolkit has no network client and never evaluates caller-provided code. See the [agent documentation](https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/blob/master/docs/agents.md).

## Runtime Provenance

```python
capabilities = optimizer.inspect_runtime()
print(capabilities.as_dict())
```

Reports include Python and PyTorch versions, device name and type, CUDA/cuDNN versions, compute capability, and the availability of AMP, channels-last, Triton, and `torch.compile`.

## Migration

The v3 primary API is `InferenceOptimizer.select(...) -> OptimizationDecision`. The v2 `Optimizer`, `OptimizationResult`, `OptimizationConfig`, and provider import paths remain compatibility aliases for one migration cycle, but new code should not depend on them. Cache and decision schemas intentionally changed and v2 cache records are ignored. See the [v3 migration guide](https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/blob/master/docs/migration-v3.md).

## Historical Research Snapshot

An earlier fixed-path notebook run on an NVIDIA Tesla T4 produced the following hardware-specific results. These are not guaranteed package performance and are not a state-of-the-art claim.

| Model | Batch | Eager FP32 | Inductor | Experimental | vs Eager | vs Inductor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-50 | 128 | 366.997 ms | 99.254 ms | 89.307 ms | 4.11x | 1.11x |
| MobileNet-V2 | 128 | 112.703 ms | 54.312 ms | 64.372 ms | 1.75x | 0.84x |
| VGG-16 | 128 | 639.399 ms | 273.326 ms | 273.402 ms | 2.34x | 1.00x |
| EfficientNet-B0 | 128 | 146.788 ms | 70.784 ms | 66.409 ms | 2.21x | 1.07x |
| DenseNet-121 | 128 | 360.750 ms | 194.997 ms | 193.169 ms | 1.87x | 1.01x |

MobileNet-V2 is the important result: the experimental path regressed against Inductor. Version 3 measures and exposes that failure and retains a baseline when the evidence is insufficient. Rerun the research notebook before citing any value.

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

Read [CONTRIBUTING.md](https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/blob/master/CONTRIBUTING.md), [SECURITY.md](https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/blob/master/SECURITY.md), and [CHANGELOG.md](https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/blob/master/CHANGELOG.md) before contributing or publishing.

## Limitations

- Selection is specific to the supplied workload distribution, device, and software versions.
- FX symbolic tracing cannot represent every data-dependent Python program.
- Provider setup may require third-party dependencies and substantial compilation time.
- Tensor parity is not a substitute for dataset-level quality evaluation.
- The current package targets inference; it does not optimize training or backward graphs.
- `peak_memory_mb` is incremental allocated CUDA memory during a warmed serial invocation; total process VRAM, energy, and concurrent-service tails require separate measurement.

## Citation and License

Citation metadata is in [CITATION.cff](https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/blob/master/CITATION.cff). Report the package version, hardware, CUDA, PyTorch, candidate providers, shapes, precision, warmup, iterations, and tolerances.

Released under the [MIT License](https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/blob/master/LICENSE).
