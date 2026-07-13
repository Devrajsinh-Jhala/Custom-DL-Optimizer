# Changelog

All notable changes to this project are documented here.

## [Unreleased]

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

[Unreleased]: https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer/releases/tag/v1.0.1
