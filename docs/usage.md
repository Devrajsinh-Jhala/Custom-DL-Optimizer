# Usage Guide

## Optimize a Model

```python
import torch

from custom_dl_optimizer import OptimizationConfig, Optimizer

optimizer = Optimizer(
    device="cuda" if torch.cuda.is_available() else "cpu",
    config=OptimizationConfig(
        enable_compile=torch.cuda.is_available(),
        compile_mode="reduce-overhead",
        min_speedup=1.02,
    ),
)
result = optimizer.optimize(model.eval(), example_inputs)
output = result(example_inputs)
```

`result.module` is the selected callable module. `result.report` contains the evidence used to select it.

## Multi-Input Models

Pass representative positional and keyword inputs exactly as the model receives them:

```python
result = optimizer.optimize(
    model,
    input_ids,
    attention_mask=attention_mask,
)
output = result(input_ids, attention_mask=attention_mask)
```

Nested tuples, lists, and dictionaries are prepared recursively.

## Read and Save Evidence

```python
print(result.selected_plan)
print(result.report.selection_reason)

for candidate in result.report.candidates:
    print(candidate.name, candidate.latency_ms, candidate.latency_p90_ms)
    print(candidate.setup_time_s, candidate.first_call_time_s)
    print(candidate.projected_total_ms, candidate.break_even_calls_vs_baseline)
    print(candidate.speedup_vs_eager, candidate.speedup_vs_native)

result.save_report("artifacts/optimization.json")
```

The JSON report includes runtime provenance, graph rewrite coverage, construction and first-call time, every repeat sample, median/repeat-sample-P90/standard-deviation latency, projected total time, break-even calls, throughput, relative speedups, numerical error, and failures. These samples are per-repeat averages, so `latency_p90_ms` is not a request-level service-tail percentile.

## Configuration

Important `OptimizationConfig` fields:

| Field | Default | Purpose |
| --- | ---: | --- |
| `enable_profiling` | `True` | Capture representative operator self time |
| `benchmark_eager` | `True` | Measure the eager FP32 reference |
| `enable_fx` | `True` | Evaluate the built-in FX candidate |
| `enable_conv_bn_folding` | `True` | Fold safe evaluation-mode Conv-BN patterns |
| `enable_triton` | `True` | Use eligible custom Triton operations |
| `enable_amp` | `True` | Evaluate CUDA FP16 autocast |
| `channels_last` | `True` | Evaluate channels-last for 4D CUDA tensors |
| `enable_compile` | `False` | Add the FX plus TorchInductor candidate |
| `dynamic_shapes` | `False` | Request dynamic TorchInductor guards |
| `verify_outputs` | `True` | Reject parity failures |
| `selection_repeats` | `3` | Independent averaged timing samples |
| `expected_calls` | `None` | Select by projected cold-start plus steady-state cost when set |
| `min_speedup` | `1.02` | Gain required over the fastest valid built-in candidate |

## Correctness

Candidates must match eager FP32 output structure and satisfy `torch.allclose` using the configured `rtol` and `atol`. This protects local tensor behavior, not end-to-end task quality.

## Compilation Cost

`CandidateReport.setup_time_s` measures candidate construction and `first_call_time_s` captures lazy compilation plus the first inference. Steady-state latency excludes both. With `expected_calls=N`, projected cost is:

```text
setup + first call + median latency * (N - 1)
```

The selector applies `min_speedup` to that projected total. Without `expected_calls`, it retains steady-state median selection. `break_even_calls_vs_baseline` reports when a faster candidate recovers its additional cold-start cost.

## CPU Behavior

CPU execution disables CUDA AMP, channels-last selection, and Triton execution. FX rewrites, external CPU providers, parity checks, reports, and agent tooling remain available.
