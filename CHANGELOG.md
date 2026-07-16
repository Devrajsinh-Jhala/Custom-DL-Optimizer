# Changelog

All notable changes to this project are documented here.

## [Unreleased]

## [3.0.0] - 2026-07-17

### Added

- Confidence-gated plan replacement using deterministic bootstrap mean-cost bounds.
- Explicit `ExecutionTarget`, `MeasurementPolicy`, `ValidationPolicy`, `DeploymentConstraints`, and `OptimizationPolicy` contracts.
- Callable `InferenceRunner`, `BuiltPlan`, `ProviderAvailability`, and `ExecutionProvider` contracts for heterogeneous runtimes.
- `InferenceOptimizer.select` and callable `OptimizationDecision` primary API.
- Versioned report, bundle, paper-export, and cache schemas with baseline, seed, candidate order, cost bounds, and rejection evidence.
- Contract tests for lifecycle amortization, uncertainty gating, provider failure isolation, generic runners, deterministic order, and cache migration.

### Changed

- Selection uses weighted mean lifecycle cost rather than a median point estimate.
- A challenger replaces the fastest valid eager/native baseline only when its upper confidence bound clears `min_speedup` against the baseline lower bound.
- Provider availability reports a reason and provider builds return plans with artifacts and provenance metadata.
- First-party Torch-TensorRT, ONNX Runtime, and TorchAO integrations implement the v3 provider contract.
- Candidate measurement order is randomized deterministically to reduce fixed ordering bias.

### Compatibility

- v2 top-level names remain aliases for one migration cycle.
- Cache schema v3 intentionally rejects v2 decisions that lack confidence evidence.

## [2.2.0] - 2026-07-15

### Added

- Weighted `WorkloadProfile` and `WorkloadCase` selection across multiple signatures.
- Content-addressed persistent plan cache with parity, constraint, and latency-regression validation.
- First-party optional Torch-TensorRT, ONNX Runtime, and TorchAO providers.
- Setup, first-call, and incremental CUDA allocation constraints.
- Per-case request samples, P95/P99, mean confidence bounds, and workload evidence.
- CLI commands for runtime inspection, report summaries, cache management, and paper exports.
- Portable decision bundles, CSV/LaTeX/figure exporters, and a paper-launch experimental protocol.

### Changed

- Latency samples now represent individually timed serial invocations rather than repeat averages.
- Agent workloads can register complete weighted profiles.
- Expected-call projection accounts for the first invocation of every workload case.

## [2.1.0] - 2026-07-14

### Added

- Optional expected-call amortization for cold-start-aware plan selection.
- Separate candidate construction and lazy first-call timing.
- Per-candidate latency samples, minimum, P90, standard deviation, and break-even calls.
- Total optimizer wall-clock timing and explicit report selection basis.

### Changed

- External provider first-call compilation is measured separately from provider construction.
- PyPI-facing README links use absolute repository URLs.

## [2.0.0] - 2026-07-14

### Added

- Result-oriented `Optimizer` and `OptimizationResult` API.
- Serializable runtime capability and hardware provenance records.
- Eager and native relative speedup metrics for every candidate.
- Pluggable `CandidateProvider` protocol for external compilers and runtimes.
- Dependency-neutral `OptimizationAgentToolkit` with registered-workload boundaries.
- JSON report persistence and v1-to-v2 migration documentation.

### Changed

- Eager FP32 is benchmarked by default as an explicit reference candidate.
- Candidate models are isolated before memory-format conversion and provider compilation.
- Version 2 documentation uses a single result object as the primary contract.

### Deprecated

- `AutoOptimizer` remains for one migration release and is no longer the primary API.

## [1.1.0] - 2026-07-14

### Added

- Regression-aware candidate benchmarking and native fallback.
- Structured `OptimizationConfig` and `OptimizationReport` APIs.
- Safe Conv2d-BatchNorm2d folding.
- Functional and nested module ReLU rewriting.
- Multi-input and keyword-input support.
- Optional TorchInductor candidate isolated from user Triton kernels.
- Numerical output validation before candidate selection.
- Modern package metadata, CI, Trusted Publishing, examples, and documentation.
- GitHub Pages project site.

### Changed

- Triton is runtime-detected instead of independently upgraded by the package.
- Channels-last and AMP are now guarded by device and input eligibility.
- Profiling returns structured operator statistics.

### Removed

- Generated wheels, egg metadata, and Python cache files from source control.

## [1.0.1] - 2026-03-16

- Initial PyPI package with FX ReLU replacement, Triton kernel injection, channels-last layout, and AMP.

[Unreleased]: https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/compare/v2.2.0...v3.0.0
[2.2.0]: https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/releases/tag/v1.0.1
