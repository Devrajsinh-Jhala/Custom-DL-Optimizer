# Research Framing

Custom-DL-Optimizer should be framed as a PyTorch-native micro-compiler for inference optimization.

## Research Contribution

The core contribution is not that the package replaces production runtimes such as TensorRT. The contribution is a transparent, regression-aware plan-selection layer that preserves the PyTorch `nn.Module` interface while applying compiler-style transformations:

- programmatic runtime profiling
- FX graph tracing
- safe Conv-BatchNorm folding and operator replacement
- GPU memory-layout optimization
- mixed-precision execution
- optional Triton and TorchInductor candidates
- numerical validation and measured fallback

## Strong Claim

Use this style:

```text
Custom-DL-Optimizer profiles multiple PyTorch inference plans and selects a custom graph only when it remains numerically valid and clears a measured gain threshold over the native optimized path.
```

Historical claim for the fixed-path Tesla T4 research run:

```text
On a Tesla T4, Custom-DL-Optimizer achieved up to 4.11x speedup over PyTorch eager FP32 and 2.46x average speedup across five CNN architectures. Against stronger AMP/NHWC and AMP/NHWC+TorchInductor baselines, it achieved 1.09x and 1.01x average speedup, respectively, with the strongest compiler-baseline gain on ResNet-50 at 1.11x.
```

Do not describe this historical table as a v1.1 package benchmark. Rerun the current notebook and report confidence intervals before publication.

## Claims to Avoid

Avoid these until a full literature review and controlled comparison are complete:

- "state of the art"
- "faster than TensorRT"
- "production-grade compiler"
- "general optimizer for all PyTorch graphs"

## Suggested Paper Title

```text
Custom-DL-Optimizer: A Profile-Guided PyTorch Micro-Compiler for Hardware-Aware NVIDIA GPU Inference
```
