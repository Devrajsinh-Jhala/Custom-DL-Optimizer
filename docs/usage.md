# Usage Guide

## Select a Plan

```python
import torch

from custom_dl_optimizer import (
    ExecutionTarget,
    InferenceOptimizer,
    MeasurementPolicy,
    OptimizationPolicy,
)

policy = OptimizationPolicy(
    enable_compile=torch.cuda.is_available(),
    compile_mode="reduce-overhead",
    measurement=MeasurementPolicy(iterations=20, repeats=3),
)
optimizer = InferenceOptimizer(
    target=ExecutionTarget("cuda" if torch.cuda.is_available() else "cpu"),
    policy=policy,
)
decision = optimizer.select_signature(model.eval(), example_inputs)
output = decision(example_inputs)
```

`decision.runner` is the selected callable. `decision.plan` contains provider artifacts and metadata. `decision.report` contains the evidence used to retain the baseline or select a challenger.

## Weighted Workloads

Use a weighted profile for multiple batches, shapes, dtypes, or keyword signatures:

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
decision = optimizer.select(model, profile)
```

Every candidate must pass parity on every case. Traffic weights are normalized and applied to each candidate's cost samples.

## Lifecycle Selection

For a short-lived job, compilation can cost more than it saves. Select by complete lifecycle cost when the serving horizon is known:

```python
from custom_dl_optimizer import DeploymentConstraints, OptimizationPolicy

policy = OptimizationPolicy(
    objective="lifecycle_latency",
    enable_compile=True,
    constraints=DeploymentConstraints(
        expected_calls=10_000,
        min_speedup=1.03,
        max_setup_time_s=60,
        max_first_call_time_s=15,
        max_peak_memory_mb=4096,
    ),
)
```

For candidate `p`, v3 estimates:

```text
cost(p) = setup(p) + first_calls(p) + remaining_calls * weighted_mean_latency(p)
```

Bootstrap resampling produces a confidence interval for that cost. A challenger replaces the fastest valid native baseline only when:

```text
challenger_upper_bound <= baseline_lower_bound / min_speedup
```

If the intervals overlap the threshold, v3 retains the baseline and records the rejection reason.

## Measurement Policy

| Field | Default | Purpose |
| --- | ---: | --- |
| `warmup` | `5` | Untimed warmup calls per workload case |
| `iterations` | `20` | Individually timed calls per repeat |
| `repeats` | `3` | Measurement passes |
| `confidence_level` | `0.95` | Bootstrap interval coverage |
| `bootstrap_resamples` | `1000` | Deterministic bootstrap resamples |
| `random_seed` | `17` | Candidate order and bootstrap seed |
| `randomize_candidate_order` | `True` | Reduce fixed ordering bias |
| `measure_peak_memory` | `True` | Measure warmed incremental CUDA allocation |

The report preserves raw serial invocation samples, median, mean, P90/P95/P99, latency bounds, selection-cost bounds, candidate order, random seed, and runtime provenance. These are not concurrent service-tail measurements.

## Validation Policy

```python
from custom_dl_optimizer import ValidationPolicy

policy = OptimizationPolicy(
    validation=ValidationPolicy(
        verify_outputs=True,
        rtol=5e-2,
        atol=5e-2,
    )
)
```

Candidates must preserve eager FP32 output structure and satisfy `torch.allclose` for every workload case. Tensor parity does not replace dataset-level task accuracy.

## Evidence and Bundles

```python
print(decision.selected_plan)
print(decision.report.selection_reason)
print(decision.report.baseline_plan)
print(decision.report.confidence_gate_passed)

for candidate in decision.report.candidates:
    print(candidate.name, candidate.selection_cost_ms)
    print(candidate.selection_cost_ci_low_ms, candidate.selection_cost_ci_high_ms)
    print(candidate.confidence_gate_passed, candidate.rejection_reason)

decision.save_report("artifacts/decision.json")
decision.save_bundle("artifacts/decision-bundle")
```

Bundle and report schema version 3 record evidence and provider artifact references. Providers own executable serialization.

## Persistent Cache

```python
policy = OptimizationPolicy(plan_cache_dir=".custom-dl-cache")
```

The SHA-256 key covers model weights, workload signatures and weights, policy, provider identities, and runtime capabilities. A cache hit rebuilds and validates the prior winner with parity, constraints, and a short latency-regression probe. v2 cache records are intentionally ignored.

## CPU Behavior

CPU execution disables CUDA AMP, channels-last selection, and Triton execution. FX rewrites, external CPU providers, parity checks, uncertainty gating, reports, and agent tooling remain available.
