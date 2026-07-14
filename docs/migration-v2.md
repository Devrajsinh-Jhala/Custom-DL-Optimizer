# Migrating to Version 2

Version 2 makes the result object the primary API and adds eager-relative evidence, runtime provenance, provider plugins, and agent tooling.

## Primary API

Version 1:

```python
engine = AutoOptimizer(model, config=config)
module, report = engine.optimize_with_report(sample)
```

Version 2:

```python
optimizer = Optimizer(config=config)
result = optimizer.optimize(model, sample)
module = result.module
report = result.report
```

`result(sample)` delegates to the selected module, and `result.save_report(path)` writes reproducible JSON evidence.

## Candidate Reports

Every measured candidate can now include:

- `calls_per_second`
- `speedup_vs_eager`
- `speedup_vs_native`
- runtime provenance on the parent report

The eager FP32 reference is benchmarked by default. Set `benchmark_eager=False` only when optimization setup latency is more important than eager-relative evidence.

## Compatibility Window

`AutoOptimizer`, `optimize`, `optimize_with_report`, and `prepare_inputs` remain available in version 2 for migration. New integrations should not depend on that compatibility surface; it is scheduled for removal in the next major version.
