# Candidate Provider Guide

Candidate providers let Custom-DL-Optimizer compare an external compiler or runtime under the same correctness and timing policy as its built-in plans.

## Protocol

```python
class CandidateProvider(Protocol):
    name: str

    def is_available(self, context: CandidateContext) -> bool: ...
    def build(self, model: nn.Module, context: CandidateContext) -> nn.Module: ...
```

`CandidateContext` contains the resolved device, configuration, prepared representative arguments, complete workload profile, content-addressed cache key, and an optional provider-owned artifact directory. `build` receives an isolated model copy and must return an evaluation-mode `nn.Module` compatible with every workload case.

## Function Provider

```python
from custom_dl_optimizer import FunctionCandidateProvider, Optimizer

provider = FunctionCandidateProvider(
    name="my_compiler",
    availability=lambda context: context.device.type == "cuda",
    builder=lambda model, context: compile_model(model, context.example_args),
)

optimizer = Optimizer(device="cuda", providers=(provider,))
result = optimizer.optimize(model, sample)
```

## Provider Responsibilities

- Import and validate its own optional dependencies.
- Return a callable module without mutating process-global compiler settings.
- Preserve the input/output structure used by eager PyTorch.
- Perform any required first-call compilation inside `build` or the returned module.
- Report unavailability through `is_available`; do not silently fall back to another backend under the same name.

The engine records provider build time and per-shape lazy first invocations separately, validates every workload output, measures serial request distributions, enforces resource constraints, and considers the provider only when it clears `min_speedup` against the fastest valid built-in candidate. When `expected_calls` is configured, the comparison uses projected total cost rather than steady-state latency alone.

## First-Party Torch-TensorRT Provider

```python
from custom_dl_optimizer import TorchTensorRTProvider

provider = TorchTensorRTProvider(
    compile_options={
        "optimization_level": 3,
        "min_block_size": 5,
    }
)
```

The provider uses the official `torch.compile(..., backend="torch_tensorrt")` path. When a plan cache is configured it enables backend engine reuse and places the TensorRT timing cache in the provider artifact directory. Precision, workspace, dynamic-shape, and compatibility settings remain explicit `compile_options` and are captured in the cache identity.

## First-Party ONNX Runtime Provider

```python
from custom_dl_optimizer import ONNXRuntimeProvider

provider = ONNXRuntimeProvider(
    execution_providers=(
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ),
)
```

The provider exports ONNX once per cache key and uses CUDA I/O binding for CUDA sessions. It intentionally accepts only flat positional tensor inputs; outputs may be nested tensor tuples, lists, or dictionaries. Unsupported signatures fail this candidate without affecting other plans. Install `custom-dl-optimizer[onnxruntime]` for CPU or `custom-dl-optimizer[onnxruntime-gpu]` for NVIDIA execution providers.

## First-Party TorchAO Provider

```python
from custom_dl_optimizer import TorchAOQuantizationProvider

providers = (
    TorchAOQuantizationProvider(scheme="int8_weight_only"),
    TorchAOQuantizationProvider(scheme="int4_weight_only", group_size=128),
)
```

Supported schemes are `int8_weight_only`, `int8_dynamic`, `int4_weight_only`, `float8_weight_only`, and `float8_dynamic`. Hardware and model compatibility are delegated to TorchAO; unsupported candidates fail visibly. Quantization is never enabled globally, and every quantized plan must clear the same per-case FP32 parity policy.

## Fair Comparison Checklist

- Use identical semantic inputs and output tolerances.
- Warm each backend after compilation and cache creation.
- Keep host/device transfer policy consistent.
- Separate conversion and engine-build cost from steady-state latency.
- Record precision, dynamic shape ranges, workspace, and cache settings.
- Evaluate task-level accuracy before deployment.
