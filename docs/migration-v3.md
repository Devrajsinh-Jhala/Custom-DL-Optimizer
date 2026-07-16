# Migrating to v3

Version 3 changes the primary API from a flat optimization configuration to explicit target, policy, provider, plan, and decision contracts.

## Primary API

Before:

```python
result = Optimizer(device="cuda", config=OptimizationConfig(...)).optimize(model, x)
```

After:

```python
decision = InferenceOptimizer(
    target=ExecutionTarget("cuda"),
    policy=OptimizationPolicy(...),
).select_signature(model, x)
```

Use `decision.runner`, `decision.plan`, and `decision.report`. The decision remains callable.

## Policy Split

- `MeasurementPolicy` owns timing, uncertainty, seed, order, and memory measurement.
- `ValidationPolicy` owns parity and tensor tolerances.
- `DeploymentConstraints` owns expected calls, replacement gain, setup, first-call, and memory limits.
- `OptimizationPolicy` owns the objective and candidate-generation settings.

## Provider Contract

`CandidateProvider.is_available(...) -> bool` becomes `ExecutionProvider.probe(...) -> ProviderAvailability`.

`build(...) -> nn.Module` becomes `build(...) -> BuiltPlan`. The runner may be any callable and can expose `close()`.

## Selection Semantics

v2 compared point estimates. v3 computes deterministic bootstrap mean-cost bounds and selects a challenger only when its upper bound clears the configured gain against the baseline lower bound. Reports now expose the baseline, confidence gate, random seed, candidate order, selection bounds, and rejection reason.

## Schemas and Cache

Report, bundle, paper-export manifest, and cache schemas are version 3. Existing v2 cache records are ignored because their point-estimate decisions do not carry v3 evidence.

The v2 top-level names remain compatibility aliases for one migration cycle. New integrations should use only the v3 names.
