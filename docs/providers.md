# Candidate Provider Guide

Candidate providers let Custom-DL-Optimizer compare an external compiler or runtime under the same correctness and timing policy as its built-in plans.

## Protocol

```python
class CandidateProvider(Protocol):
    name: str

    def is_available(self, context: CandidateContext) -> bool: ...
    def build(self, model: nn.Module, context: CandidateContext) -> nn.Module: ...
```

`CandidateContext` contains the resolved device, configuration, and prepared example arguments. `build` receives an isolated model copy and must return an evaluation-mode `nn.Module` compatible with those inputs.

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

The engine records provider build time, validates output, measures latency, and considers the provider only when it clears `min_speedup` against the fastest valid built-in candidate.

## Torch-TensorRT

Torch-TensorRT can register a `torch.compile` backend. Import `torch_tensorrt`, then return `torch.compile(model, backend="torch_tensorrt")` from the provider builder. Follow the installed Torch-TensorRT documentation for precision, dynamic shape, engine cache, and serialization settings.

## ONNX Runtime

ONNX Runtime does not return a PyTorch `nn.Module` directly. A provider can export the model and return a small module wrapper around an `InferenceSession`. The wrapper is responsible for tensor conversion or I/O binding. Register execution providers in priority order, such as TensorRT, CUDA, then CPU, and expose the exact provider configuration in your experiment metadata.

## Fair Comparison Checklist

- Use identical semantic inputs and output tolerances.
- Warm each backend after compilation and cache creation.
- Keep host/device transfer policy consistent.
- Separate conversion and engine-build cost from steady-state latency.
- Record precision, dynamic shape ranges, workspace, and cache settings.
- Evaluate task-level accuracy before deployment.
