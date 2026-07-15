# Paper and Launch Protocol

## Working Title

**Custom-DL-Optimizer: Constraint-Aware and Cold-Start-Amortized Selection of PyTorch Inference Plans**

## Research Question

Can a workload-aware selector choose among native PyTorch, graph rewrites, TorchInductor, external runtimes, and quantized plans while preserving numerical correctness and reducing total deployment cost across heterogeneous input distributions?

The paper should claim a selection and measurement contribution, not a universally faster kernel compiler. Backend compilers remain responsible for lowering and executable serialization.

## System Contributions

1. Weighted multi-signature workload profiles rather than one dummy input.
2. Per-case FP32 parity gates and deployment constraints for setup, first-call, and incremental CUDA allocation.
3. Cold-start-amortized plan selection using an explicit production call horizon.
4. Content-addressed persistent decisions keyed by model weights, workload, policy, providers, and runtime, with parity and latency-regression validation on reuse.
5. One evidence schema across eager, AMP/layout, FX, TorchInductor, Torch-TensorRT, ONNX Runtime, TorchAO, and private providers.

## Experimental Matrix

### Models

- Vision: ResNet-50, MobileNet-V2, EfficientNet-B0, ConvNeXt-Tiny.
- Encoder transformers: BERT-base and DistilBERT sequence classification.
- Decoder workload: one small causal language model with prefill and decode reported separately.
- At least one model with a known compiler regression or graph break.

Use public pretrained checkpoints only when their licenses permit redistribution. Record the exact checkpoint revision.

### Workload Profiles

- Vision batches: 1, 8, 32, and 128 where memory permits.
- Transformer sequence lengths: 32, 128, 512, with at least two batch sizes.
- Report both a balanced synthetic distribution and one declared production-like skew.
- Use separate static-shape and dynamic-shape experiments.

### Hardware

- Minimum: NVIDIA T4 plus one Ampere-or-newer GPU.
- Preferred: T4, L4, A10/A100, and H100 where available.
- Repeat the complete baseline set on every GPU; do not transfer a winner between architectures.

### Baselines

1. PyTorch eager FP32.
2. PyTorch AMP and eligible channels-last.
3. PyTorch `torch.compile` with `default`, `reduce-overhead`, and one declared autotuning mode.
4. Torch-TensorRT with exact compilation and cache options.
5. ONNX Runtime CUDA and TensorRT execution providers with device I/O binding.
6. Applicable TorchAO INT8/INT4/FP8 configurations.
7. Custom-DL-Optimizer selection with and without each proposed mechanism.

## Metrics

- Candidate construction and each shape's first invocation.
- Serial median, mean, P90, P95, P99, standard deviation, and 95% mean interval.
- Weighted workload latency and projected total time at declared call horizons.
- Break-even calls versus the fastest valid built-in plan.
- Incremental warmed CUDA allocation and total process VRAM measured separately with an external harness.
- Throughput under fixed concurrency and request-level service P50/P95/P99 from an external load generator.
- Maximum and mean tensor error plus a task-level metric on a calibration dataset.
- Cache-key construction time, hit validation time, invalidation rate, and full-selection time avoided.

Package percentiles are serial-invocation distributions. They must not be described as service-tail latency.

## Required Ablations

| Ablation | Question |
| --- | --- |
| Single example vs weighted profile | Does representative traffic change the selected backend? |
| Steady state vs expected-call amortization | When does compilation recover its cold start? |
| Cache disabled vs miss vs validated hit | How much selection and compilation cost is avoided? |
| Parity gate disabled | How often would an invalid fast plan otherwise win? |
| Constraints disabled | Which winners violate startup or memory policy? |
| Built-ins only vs each provider | Which runtime contributes the selected gain? |
| Static vs dynamic shapes | What is the cost of generality and recompilation? |

## Statistical Protocol

1. Lock application clocks where permitted and record power/thermal policy.
2. Randomize candidate execution order across independent trials.
3. Run at least five independent process-level trials; do not treat loop iterations as independent process trials.
4. Report raw JSON and CSV, median across trials, bootstrap confidence intervals, and all failures.
5. Separate cold-cache and warm-cache trials.
6. Synchronize CUDA around timing boundaries and keep host/device transfer policy identical.
7. Pre-register tolerances and task-level quality thresholds before examining speed results.

## Claim Matrix

| Claim | Evidence required |
| --- | --- |
| Faster than eager | Same hardware, input distribution, precision disclosure, parity pass |
| Faster than TorchInductor | Same compile mode, cache state, shapes, and process protocol |
| Lower lifecycle cost | Construction plus first calls plus declared request horizon |
| Robust across workloads | Multiple model families, shapes, and at least two GPU generations |
| Production-ready | External concurrency, total VRAM, failure recovery, and task-quality evidence |

Do not use "state of the art" unless every relevant baseline is reproduced under the same protocol and the improvement is statistically supported. A defensible primary claim is that the system avoids backend regressions and minimizes constrained lifecycle cost more reliably than any fixed plan.

## Release Artifacts

- Tagged source release and immutable PyPI distributions.
- Raw reports and cache-state metadata for every trial.
- Environment lock files or container image digest.
- Scripts that regenerate every table and figure from raw results.
- Model/checkpoint manifest with revisions and licenses.
- A limitations section covering unsupported signatures, serialization ownership, serial percentile semantics, and hardware specificity.
