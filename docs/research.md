# Research Framing

Custom-DL-Optimizer v3 should be framed as a confidence-gated lifecycle selector for heterogeneous inference plans, with compiler passes and provider runtimes as candidate generators.

## Research Contribution

The core contribution is not that the package replaces production runtimes such as TensorRT. The contribution is a transparent, regression-aware lifecycle-selection layer that preserves the PyTorch `nn.Module` interface while comparing compiler and runtime plans:

- programmatic runtime profiling
- FX graph tracing
- safe Conv-BatchNorm folding and operator replacement
- GPU memory-layout optimization
- mixed-precision execution
- optional Triton and TorchInductor candidates
- weighted multi-signature workload selection
- optional Torch-TensorRT, ONNX Runtime, and TorchAO candidates
- cold-start amortization and persistent validated decisions
- deterministic randomized candidate order and bootstrap mean-cost bounds
- confidence-gated replacement, numerical validation, resource constraints, and measured fallback

## Strong Claim

Use this style:

```text
Custom-DL-Optimizer replaces the fastest valid native inference plan only when a challenger remains numerically valid across the declared workload, satisfies deployment constraints, and its upper lifecycle-cost confidence bound clears the configured gain against the baseline lower bound.
```

Historical claim for the fixed-path Tesla T4 research run:

```text
On a Tesla T4, Custom-DL-Optimizer achieved up to 4.11x speedup over PyTorch eager FP32 and 2.46x average speedup across five CNN architectures. Against stronger AMP/NHWC and AMP/NHWC+TorchInductor baselines, it achieved 1.09x and 1.01x average speedup, respectively, with the strongest compiler-baseline gain on ResNet-50 at 1.11x.
```

Do not describe this historical table as a version 2 package benchmark. Rerun the current notebook and report confidence intervals before publication.

## Claims to Avoid

Avoid these until a full literature review and controlled comparison are complete:

- "state of the art"
- "faster than TensorRT"
- "production-grade compiler"
- "general optimizer for all PyTorch graphs"

## Suggested Paper Title

```text
Custom-DL-Optimizer v3: Confidence-Gated Lifecycle Selection of Heterogeneous Inference Plans
```

Use [paper-launch.md](paper-launch.md) as the required evaluation and claim protocol. Treat the historical T4 table as pilot evidence until the v3 experiment matrix is rerun.
