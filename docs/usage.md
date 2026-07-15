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

## Weighted Workload Profiles

One example input is rarely an honest serving distribution. Use `WorkloadProfile` to evaluate multiple batches, shapes, dtypes, or keyword signatures under one policy:

```python
from custom_dl_optimizer import WorkloadCase, WorkloadProfile

profile = WorkloadProfile(
    name="production-traffic",
    expected_calls=50_000,
    cases=(
        WorkloadCase("short", args=(short_tokens,), weight=0.65),
        WorkloadCase("medium", args=(medium_tokens,), weight=0.25),
        WorkloadCase("long", args=(long_tokens,), weight=0.10),
    ),
)
result = optimizer.optimize_workload(model, profile)
```

Weights are normalized automatically. Every candidate must pass parity on every case. Aggregate latency samples are weighted combinations of aligned per-case samples, and every case remains available in `candidate.workload_cases`.

## Read and Save Evidence

```python
print(result.selected_plan)
print(result.report.selection_reason)

for candidate in result.report.candidates:
    print(candidate.name, candidate.latency_ms, candidate.latency_p99_ms)
    print(candidate.latency_ci95_low_ms, candidate.latency_ci95_high_ms)
    print(candidate.peak_memory_mb, candidate.constraint_violations)
    print(candidate.setup_time_s, candidate.first_call_time_s)
    print(candidate.projected_total_ms, candidate.break_even_calls_vs_baseline)
    print(candidate.speedup_vs_eager, candidate.speedup_vs_native)

result.save_report("artifacts/optimization.json")
```

The JSON report includes runtime provenance, graph rewrite coverage, construction and first-call time, every serial request sample, median, mean, P90/P95/P99, approximate 95% confidence bounds for the mean, incremental peak CUDA allocation, projected total time, break-even calls, throughput, relative speedups, per-case numerical error, constraints, and failures. Percentiles describe serial benchmark invocations; they are not concurrent service-tail measurements.

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
| `selection_iterations` | `10` | Individually timed invocations per repeat |
| `selection_repeats` | `3` | Measurement passes |
| `expected_calls` | `None` | Select by projected cold-start plus steady-state cost when set |
| `min_speedup` | `1.02` | Gain required over the fastest valid built-in candidate |
| `max_setup_time_s` | `None` | Reject candidates whose construction exceeds the limit |
| `max_first_call_time_s` | `None` | Reject candidates whose aggregate per-shape first calls exceed the limit |
| `max_peak_memory_mb` | `None` | Reject candidates whose warmed incremental CUDA allocation exceeds the limit |
| `plan_cache_dir` | `None` | Persist content-addressed plan decisions and provider artifacts |
| `reuse_cached_plan` | `True` | Validate a cached winner before running a full race |
| `cache_max_latency_regression` | `1.25` | Invalidate a cached plan after excessive latency regression |

## Correctness

Candidates must match eager FP32 output structure and satisfy `torch.allclose` using the configured `rtol` and `atol`. This protects local tensor behavior, not end-to-end task quality.

## Compilation Cost

`CandidateReport.setup_time_s` measures candidate construction and `first_call_time_s` captures lazy compilation plus the first inference. Steady-state latency excludes both. With `expected_calls=N`, projected cost is:

```text
setup + first calls for K workload cases + weighted median latency * (N - K)
```

The selector applies `min_speedup` to that projected total. Without `expected_calls`, it retains steady-state median selection. `break_even_calls_vs_baseline` reports when a faster candidate recovers its additional cold-start cost.

## Persistent Cache

```python
config = OptimizationConfig(plan_cache_dir=".custom-dl-cache")
optimizer = Optimizer(device="cuda", config=config, providers=providers)
```

The SHA-256 cache key covers model structure and weights, workload signatures and weights, policy fields, provider identities, and runtime capabilities. On a hit, only the recorded winner is rebuilt. It must pass all workload parity checks, resource constraints, and a short latency probe bounded by `cache_max_latency_regression`; otherwise the engine automatically runs a full selection.

Provider-specific binary artifacts live under the same key, but providers own their serialization format. `OptimizationResult.save_bundle()` stores the decision manifest and evidence, not an unverified executable.

## CPU Behavior

CPU execution disables CUDA AMP, channels-last selection, and Triton execution. FX rewrites, external CPU providers, parity checks, reports, and agent tooling remain available.
