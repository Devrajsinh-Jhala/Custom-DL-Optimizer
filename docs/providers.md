# Execution Provider Guide

Execution providers let v3 compare a complete external compiler, runtime, or accelerator plan under the same workload, correctness, lifecycle, and confidence policy as built-in PyTorch plans.

## Protocol

```python
class ExecutionProvider(Protocol):
    name: str

    def probe(self, context: ProviderContext) -> ProviderAvailability: ...
    def build(self, model: nn.Module, context: ProviderContext) -> BuiltPlan: ...
```

`ProviderContext` contains the resolved device, `ExecutionTarget`, complete `OptimizationPolicy`, representative inputs, weighted workload, cache key, and an optional provider-owned artifact directory.

`BuiltPlan` contains a callable `InferenceRunner`, artifact references, and JSON-serializable provenance metadata. A runner does not need to subclass `nn.Module`; this permits local RPC, ONNX Runtime, mobile delegate, and vendor SDK adapters.

## Function Provider

```python
from custom_dl_optimizer import (
    BuiltPlan,
    FunctionExecutionProvider,
    ProviderAvailability,
)

provider = FunctionExecutionProvider(
    name="my_accelerator",
    availability=lambda context: (
        ProviderAvailability.supported()
        if vendor_sdk_available()
        else ProviderAvailability.unsupported("vendor SDK is not installed")
    ),
    builder=lambda model, context: BuiltPlan(
        runner=compile_model(model, context.example_args),
        artifacts=(str(context.artifact_dir),),
        metadata={"backend": "my_accelerator", "sdk": sdk_version()},
    ),
)
```

## Provider Responsibilities

- Probe dependencies and target support without mutating the model.
- Build from the isolated model copy supplied by the engine.
- Preserve input and output structure for every workload case.
- Return explicit unavailability instead of silently falling back under the same name.
- Put serialized engines, timing caches, and conversion artifacts under `artifact_dir`.
- Record backend, precision, SDK, compiler flags, and delegate metadata.
- Release resources through an optional runner `close()` method.

The engine owns candidate ordering, first-call timing, serial steady-state timing, parity, constraints, confidence bounds, baseline comparison, and failure isolation.

## Mobile and NPU Providers

The protocol is deliberately runner-based so an ExecuTorch or vendor NPU backend can participate without pretending to be a PyTorch module. A provider may export the model, lower it to a delegate, load the artifact on a connected target, and expose a runner that accepts the same tensors.

Custom-DL-Optimizer v3 does not claim built-in MediaTek NPU performance or compatibility. Such a claim requires the matching ExecuTorch/MediaTek SDK, supported SoC, model coverage, power mode, and device-side measurements. The provider contract is ready for that integration and will report an unavailable reason when those prerequisites are absent.

## First-Party Providers

`TorchTensorRTProvider` uses `torch.compile(..., backend="torch_tensorrt")` and can reuse a backend timing cache.

`ONNXRuntimeProvider` exports one ONNX artifact per cache key and uses CUDA I/O binding for GPU sessions. It currently requires flat positional tensor inputs; outputs may use nested tensor containers.

`TorchAOQuantizationProvider` supports maintained TorchAO weight-only and dynamic schemes. Every quantized plan must still clear per-case FP32 parity and the same confidence gate.

## Fair Comparison Checklist

- Use identical semantic inputs and output tolerances.
- Pin software versions, power mode, clocks policy, and device temperature where possible.
- Separate conversion, setup, and first-call compilation from steady state.
- Record precision, dynamic ranges, workspace, cache state, and provider settings.
- Repeat randomized candidate races across independent process runs.
- Evaluate dataset-level accuracy, memory, energy, and concurrent service tails separately.
