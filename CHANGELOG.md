# Changelog

All notable changes to this project are documented here.

## [Unreleased]

## [2.0.0] - 2026-07-14

### Added

- Result-oriented `Optimizer` and `OptimizationResult` API.
- Serializable runtime capability and hardware provenance records.
- Eager and native relative speedup metrics for every candidate.
- Pluggable `CandidateProvider` protocol for external compilers and runtimes.
- Dependency-neutral `OptimizationAgentToolkit` with registered-workload boundaries.
- JSON report persistence and v1-to-v2 migration documentation.
- Optional expected-call amortization for cold-start-aware plan selection.
- Separate candidate construction and lazy first-call timing.
- Per-candidate latency samples, minimum, P90, standard deviation, and break-even calls.
- Total optimizer wall-clock timing and explicit report selection basis.

### Changed

- Eager FP32 is benchmarked by default as an explicit reference candidate.
- Candidate models are isolated before memory-format conversion and provider compilation.
- External provider first-call compilation is measured separately from provider construction.
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

[Unreleased]: https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/releases/tag/v1.0.1
