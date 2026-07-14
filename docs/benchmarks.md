# Benchmarking Notes

Benchmark results are highly dependent on:

- GPU architecture
- CUDA and driver versions
- PyTorch and Triton versions
- batch size
- input resolution
- warmup count
- whether TensorFloat-32, AMP, channels-last, and TorchInductor are enabled

## Research Notebook

Use `Custom_DL_Optimizer_Research_Colab.ipynb` for paper-quality measurements. The notebook exports:

- CSV results
- JSON results
- LaTeX table
- latency comparison figure
- speedup figure
- compiler pass coverage figure
- output parity figure

## Historical T4 Snapshot

These measurements came from the earlier fixed-path research notebook. They are retained for reproducibility and must not be presented as version 2 package-level results.

| Model | Speedup vs Eager FP32 | Speedup vs AMP/NHWC | Speedup vs AMP/NHWC + Inductor |
| --- | ---: | ---: | ---: |
| ResNet-50 | 4.11x | 1.56x | 1.11x |
| MobileNet-V2 | 1.75x | 0.84x | 0.84x |
| VGG-16 | 2.34x | 1.00x | 1.00x |
| EfficientNet-B0 | 2.21x | 1.05x | 1.07x |
| DenseNet-121 | 1.87x | 1.01x | 1.01x |

Average CNN speedups:

- 2.46x over PyTorch eager FP32
- 1.09x over AMP/NHWC
- 1.01x over AMP/NHWC + TorchInductor

Interpretation: most of the gain over eager FP32 comes from known precision, layout, and compiler optimizations. The fixed path roughly matched TorchInductor across the suite and regressed on MobileNet-V2. Version 2 records eager-relative and native-relative evidence, and a custom or provider path must clear a guard threshold before replacing the native path.

## Reporting Template

When reporting results, include:

```text
GPU:
Driver:
CUDA:
PyTorch:
Triton:
Batch size:
Input shape:
Warmup iterations:
Measured iterations:
Repeats:
Timing method:
Output parity tolerance:
```

## Baselines

For a credible systems paper, compare against progressively stronger baselines:

1. PyTorch eager FP32
2. PyTorch AMP
3. PyTorch AMP + channels-last
4. PyTorch AMP + channels-last + TorchInductor
5. External provider candidates under the same measurement policy
6. Custom-DL-Optimizer adaptive selection

Do not claim SOTA unless the same hardware, model, precision, batch size, and input shape are compared against TensorRT, Torch-TensorRT, TVM, XLA, and TorchInductor.
