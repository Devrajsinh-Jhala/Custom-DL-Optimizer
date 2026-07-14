"""Build the standalone Custom-DL-Optimizer research Colab notebook.

The generated notebook is intentionally self-contained so experiments can be
reproduced in a clean Google Colab GPU runtime without importing local code.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Custom_DL_Optimizer_Research_Colab.ipynb"


def source(text: str) -> list[str]:
    text = dedent(text).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


cells = [
    markdown(
        r"""
        # Custom-DL-Optimizer: Reproducible Research Benchmark

        This notebook evaluates a profile-guided, regression-aware PyTorch inference optimizer on NVIDIA GPUs. It compares eager FP32, native AMP/layout optimization, TorchInductor, and an adaptive Custom-DL execution-plan portfolio. The reported path is selected with short pilot measurements and then evaluated with separate timing samples.

        **Research protocol.** Run all cells in a fresh GPU runtime. The notebook records the software/hardware environment, raw repeat timings, bootstrap 95% confidence intervals, numerical error, peak inference memory, compiler-pass coverage, plan-selection trials, and paper-ready tables/figures. Performance values are always generated from the current runtime; no result is hardcoded.

        **Claim boundary.** These experiments can support claims about the tested GPU, software versions, shapes, and precision. They do not establish state of the art without controlled TensorRT, Torch-TensorRT, TVM, XLA, and other relevant comparisons on identical hardware and workloads.
        """
    ),
    code(
        r"""
        # Keep the PyTorch-bundled Triton build. Upgrading Triton independently can break Inductor in Colab.
        !pip -q install pandas matplotlib seaborn tabulate
        """
    ),
    code(
        r"""
        import os
        import gc
        import json
        import math
        import random
        import platform
        import subprocess
        import sys
        import time
        import warnings
        from dataclasses import dataclass, asdict
        from datetime import datetime, timezone
        from typing import Any, Callable, Dict, List, Optional, Tuple

        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.fx as fx

        warnings.filterwarnings("ignore")

        try:
            import torchvision
            from torchvision import models
        except Exception as exc:
            raise RuntimeError("torchvision is required for the CNN benchmarks") from exc

        try:
            import triton
            import triton.language as tl
            TRITON_AVAILABLE = True
            TRITON_VERSION = getattr(triton, "__version__", "unknown")
        except Exception as exc:
            TRITON_AVAILABLE = False
            TRITON_VERSION = "unavailable"
            triton = None
            tl = None
            print("Triton is unavailable; only the optional Triton ablation will be skipped:", repr(exc))

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        assert DEVICE == "cuda", "Switch Colab to a GPU runtime before running this notebook."

        SEED = 1234

        def seed_everything(seed: int = SEED) -> None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        seed_everything()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        gpu = torch.cuda.get_device_properties(0)
        EXPERIMENT_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        RUNTIME_METADATA = {
            "experiment_id": EXPERIMENT_ID,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "torchvision": torchvision.__version__,
            "triton": TRITON_VERSION,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_memory_gb": round(gpu.total_memory / 1024**3, 3),
            "seed": SEED,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        }

        try:
            smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version,pstate,temperature.gpu,power.limit", "--format=csv,noheader"],
                text=True,
            ).strip()
            RUNTIME_METADATA["nvidia_smi"] = smi
        except Exception as exc:
            RUNTIME_METADATA["nvidia_smi"] = f"unavailable: {exc!r}"

        print(json.dumps(RUNTIME_METADATA, indent=2))
        """
    ),
    code(
        r"""
        # Main experiment switches. PAPER_MODE uses 500 measured forwards per full result (50 x 10).
        PAPER_MODE = True
        RUN_CNN_BENCHMARKS = True
        RUN_TRANSFORMER_MICROBENCHMARK = True
        RUN_TORCH_COMPILE = True
        RUN_INDUCTOR_BASELINE = True
        ENABLE_CUDA_GRAPHS = True
        RUN_TRITON_RELU_ABLATION = True
        RUN_OPERATOR_PROFILING = True

        # Optional shape-sensitivity study. Enable after the main run for the final paper appendix.
        RUN_BATCH_SCALING_STUDY = False
        BATCH_SCALING_MODEL = "ResNet-50"
        BATCH_SCALING_SIZES = [1, 8, 32, 128]

        COMPILE_MODE = "max-autotune"  # Also consider "reduce-overhead" in a separate ablation.
        AMP_DTYPE = torch.float16
        MIN_CUSTOM_GAIN = 0.02  # Require a 2% pilot gain before selecting a custom plan over native fallback.
        PARITY_RTOL = 5e-2
        PARITY_ATOL = 5e-2
        BOOTSTRAP_RESAMPLES = 5000

        WARMUP_STEPS = 20 if PAPER_MODE else 10
        BENCHMARK_ITERATIONS = 50 if PAPER_MODE else 20
        REPEATS = 10 if PAPER_MODE else 5
        PILOT_WARMUP_STEPS = 8
        PILOT_ITERATIONS = 20
        PILOT_REPEATS = 3

        BATCH_CANDIDATES = [128, 64, 32, 16, 8, 4, 1]
        CNN_MODEL_FACTORIES = {
            "ResNet-50": lambda: models.resnet50(weights=None),
            "MobileNet-V2": lambda: models.mobilenet_v2(weights=None),
            "VGG-16": lambda: models.vgg16(weights=None),
            "EfficientNet-B0": lambda: models.efficientnet_b0(weights=None),
            "DenseNet-121": lambda: models.densenet121(weights=None),
        }

        RESULT_DIR = "/content/custom_dl_optimizer_research_outputs"
        os.makedirs(RESULT_DIR, exist_ok=True)

        EXPERIMENT_CONFIG = {
            "paper_mode": PAPER_MODE,
            "run_cnn_benchmarks": RUN_CNN_BENCHMARKS,
            "run_transformer_microbenchmark": RUN_TRANSFORMER_MICROBENCHMARK,
            "run_torch_compile": RUN_TORCH_COMPILE,
            "run_inductor_baseline": RUN_INDUCTOR_BASELINE,
            "enable_cuda_graphs": ENABLE_CUDA_GRAPHS,
            "run_triton_relu_ablation": RUN_TRITON_RELU_ABLATION,
            "run_operator_profiling": RUN_OPERATOR_PROFILING,
            "run_batch_scaling_study": RUN_BATCH_SCALING_STUDY,
            "compile_mode": COMPILE_MODE,
            "amp_dtype": str(AMP_DTYPE),
            "min_custom_gain": MIN_CUSTOM_GAIN,
            "parity_rtol": PARITY_RTOL,
            "parity_atol": PARITY_ATOL,
            "warmup_steps": WARMUP_STEPS,
            "benchmark_iterations": BENCHMARK_ITERATIONS,
            "repeats": REPEATS,
            "pilot_warmup_steps": PILOT_WARMUP_STEPS,
            "pilot_iterations": PILOT_ITERATIONS,
            "pilot_repeats": PILOT_REPEATS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "batch_candidates": BATCH_CANDIDATES,
        }
        print("Outputs:", RESULT_DIR)
        print(json.dumps(EXPERIMENT_CONFIG, indent=2))
        """
    ),
    markdown(
        r"""
        ## Compiler and execution-plan implementation

        The compiler pass safely folds Conv2d-BatchNorm2d only when the convolution output has a single consumer. The execution portfolio then composes graph surgery, channels-last layout, AMP, TorchInductor, and optional CUDA Graph replay. Failed transformations fall back without invalidating the full run.
        """
    ),
    code(
        r"""
        def cleanup_cuda() -> None:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


        def instantiate_model(factory: Callable[[], nn.Module], seed: int = SEED) -> nn.Module:
            seed_everything(seed)
            return factory().eval().to(DEVICE)


        def prepare_input(x: torch.Tensor, channels_last: bool) -> torch.Tensor:
            if channels_last and x.dim() == 4:
                return x.contiguous(memory_format=torch.channels_last)
            return x.contiguous()


        def _set_submodule(root: nn.Module, target: str, module: nn.Module) -> None:
            atoms = target.split(".")
            parent = root
            for atom in atoms[:-1]:
                parent = getattr(parent, atom)
            setattr(parent, atoms[-1], module)


        def fuse_conv_bn_eval_fx(gm: fx.GraphModule) -> int:
            modules = dict(gm.named_modules())
            count = 0
            for bn_node in list(gm.graph.nodes):
                if bn_node.op != "call_module" or len(bn_node.args) != 1:
                    continue
                bn = modules.get(bn_node.target)
                conv_node = bn_node.args[0]
                if not isinstance(bn, nn.BatchNorm2d) or not isinstance(conv_node, fx.Node):
                    continue
                if conv_node.op != "call_module" or len(conv_node.users) != 1:
                    continue
                conv = modules.get(conv_node.target)
                if not isinstance(conv, nn.Conv2d):
                    continue
                fused = torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
                _set_submodule(gm, conv_node.target, fused)
                _set_submodule(gm, bn_node.target, nn.Identity())
                bn_node.replace_all_uses_with(conv_node)
                gm.graph.erase_node(bn_node)
                count += 1
            if count:
                gm.graph.eliminate_dead_code()
                gm.graph.lint()
                gm.recompile()
            return count


        class AMPWrapper(nn.Module):
            def __init__(self, model: nn.Module, enabled: bool, dtype: torch.dtype = AMP_DTYPE):
                super().__init__()
                self.model = model
                self.enabled = enabled
                self.dtype = dtype

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.enabled):
                    return self.model(x)


        class CUDAGraphWrapper(nn.Module):
            # Static-shape CUDA Graph replay with input-copy cost included in latency.

            def __init__(self, model: nn.Module, example_input: torch.Tensor, warmup: int = 3):
                super().__init__()
                self.model = model.eval()
                self.static_input = example_input.detach().clone()
                self.graph = torch.cuda.CUDAGraph()
                self.static_output = None

                side_stream = torch.cuda.Stream()
                side_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side_stream), torch.inference_mode():
                    for _ in range(warmup):
                        self.static_output = self.model(self.static_input)
                torch.cuda.current_stream().wait_stream(side_stream)
                torch.cuda.synchronize()
                with torch.cuda.graph(self.graph):
                    self.static_output = self.model(self.static_input)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.shape != self.static_input.shape or x.dtype != self.static_input.dtype:
                    return self.model(x)
                self.static_input.copy_(x)
                self.graph.replay()
                return self.static_output


        @dataclass(frozen=True)
        class PlanSpec:
            key: str
            label: str
            role: str
            fx_fold: bool
            channels_last: bool
            amp: bool
            compile: bool
            cuda_graph: bool


        @dataclass
        class PlanReport:
            plan_key: str = ""
            plan_label: str = ""
            conv_bn_fusions: int = 0
            fx_trace_ok: bool = True
            fx_error: str = ""
            channels_last: bool = False
            amp: bool = False
            compiled: bool = False
            compile_mode: str = "disabled"
            compile_error: str = ""
            cuda_graph: bool = False
            cuda_graph_error: str = ""
            setup_time_s: float = 0.0


        def make_plan_specs(channels_last: bool) -> Dict[str, PlanSpec]:
            native_label = "Native AMP/NHWC" if channels_last else "Native AMP"
            inductor_label = "AMP/NHWC + Inductor" if channels_last else "AMP + Inductor"
            fx_label = "FX Fold" if channels_last else "FX Trace"
            return {
                "eager_fp32": PlanSpec("eager_fp32", "PyTorch Eager FP32", "baseline", False, False, False, False, False),
                "native_amp": PlanSpec("native_amp", native_label, "fallback", False, channels_last, True, False, False),
                "native_inductor": PlanSpec("native_inductor", inductor_label, "fallback", False, channels_last, True, True, False),
                "fx_amp": PlanSpec("fx_amp", f"{fx_label} + {native_label}", "custom", True, channels_last, True, False, False),
                "fx_inductor": PlanSpec("fx_inductor", f"{fx_label} + {inductor_label}", "custom", True, channels_last, True, True, False),
                "fx_inductor_graph": PlanSpec("fx_inductor_graph", f"{fx_label} + {inductor_label} + CUDA Graph", "custom", True, channels_last, True, True, True),
            }


        def build_plan(
            factory: Callable[[], nn.Module],
            example_input: torch.Tensor,
            spec: PlanSpec,
        ) -> Tuple[nn.Module, torch.Tensor, PlanReport]:
            started = time.perf_counter()
            report = PlanReport(
                plan_key=spec.key,
                plan_label=spec.label,
                channels_last=spec.channels_last,
                amp=spec.amp,
            )
            model = instantiate_model(factory)

            if spec.fx_fold:
                try:
                    model = fx.symbolic_trace(model).eval()
                    report.conv_bn_fusions = fuse_conv_bn_eval_fx(model)
                except Exception as exc:
                    report.fx_trace_ok = False
                    report.fx_error = repr(exc)[:500]
                    model = instantiate_model(factory)

            if spec.channels_last:
                model = model.to(memory_format=torch.channels_last)
            plan_input = prepare_input(example_input, spec.channels_last)
            if spec.amp:
                model = AMPWrapper(model, enabled=True).eval().to(DEVICE)

            if spec.compile and RUN_TORCH_COMPILE and hasattr(torch, "compile"):
                try:
                    candidate = torch.compile(model, mode=COMPILE_MODE, fullgraph=False, dynamic=False)
                    with torch.inference_mode():
                        _ = candidate(plan_input)
                    torch.cuda.synchronize()
                    model = candidate.eval()
                    report.compiled = True
                    report.compile_mode = COMPILE_MODE
                except Exception as exc:
                    report.compile_mode = "fallback_eager"
                    report.compile_error = repr(exc)[:500]
                    torch.cuda.empty_cache()
            elif spec.compile:
                report.compile_mode = "unavailable"

            if spec.cuda_graph and ENABLE_CUDA_GRAPHS:
                try:
                    model = CUDAGraphWrapper(model.eval(), plan_input).eval()
                    with torch.inference_mode():
                        _ = model(plan_input)
                    torch.cuda.synchronize()
                    report.cuda_graph = True
                except Exception as exc:
                    report.cuda_graph_error = repr(exc)[:500]
                    torch.cuda.empty_cache()
            elif spec.cuda_graph:
                report.cuda_graph_error = "disabled"

            report.setup_time_s = time.perf_counter() - started
            return model.eval(), plan_input, report
        """
    ),
    markdown(
        r"""
        ## Measurement, uncertainty, and correctness

        CUDA events measure completed GPU work. Each reported sample is the mean latency over a block of iterations; the final statistic is the median across independent repeat blocks. The notebook retains all repeat samples and estimates 95% confidence intervals by bootstrap resampling. Setup/compilation time is reported separately from steady-state inference.
        """
    ),
    code(
        r"""
        def _stable_seed(label: str) -> int:
            return SEED + sum((i + 1) * ord(ch) for i, ch in enumerate(label)) % 100000


        def bootstrap_median_ci(samples: List[float], label: str, resamples: int = BOOTSTRAP_RESAMPLES) -> Tuple[float, float]:
            arr = np.asarray(samples, dtype=np.float64)
            if arr.size < 2:
                value = float(np.median(arr))
                return value, value
            rng = np.random.default_rng(_stable_seed(label))
            indices = rng.integers(0, arr.size, size=(resamples, arr.size))
            medians = np.median(arr[indices], axis=1)
            return tuple(float(v) for v in np.percentile(medians, [2.5, 97.5]))


        def bootstrap_speedup(
            baseline_samples: List[float],
            candidate_samples: List[float],
            label: str,
            resamples: int = BOOTSTRAP_RESAMPLES,
        ) -> Dict[str, float]:
            base = np.asarray(baseline_samples, dtype=np.float64)
            cand = np.asarray(candidate_samples, dtype=np.float64)
            rng = np.random.default_rng(_stable_seed(label))
            base_idx = rng.integers(0, base.size, size=(resamples, base.size))
            cand_idx = rng.integers(0, cand.size, size=(resamples, cand.size))
            ratios = np.median(base[base_idx], axis=1) / np.median(cand[cand_idx], axis=1)
            low, high = np.percentile(ratios, [2.5, 97.5])
            return {
                "point": float(np.median(base) / np.median(cand)),
                "ci_low": float(low),
                "ci_high": float(high),
                "probability_gt_1": float(np.mean(ratios > 1.0)),
            }


        def cuda_latency_ms(
            model: nn.Module,
            inputs: torch.Tensor,
            label: str,
            warmup: int = WARMUP_STEPS,
            iterations: int = BENCHMARK_ITERATIONS,
            repeats: int = REPEATS,
        ) -> Dict[str, Any]:
            model.eval()
            samples = []
            with torch.inference_mode():
                for _ in range(warmup):
                    _ = model(inputs)
                torch.cuda.synchronize()
                for _ in range(repeats):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(iterations):
                        _ = model(inputs)
                    end.record()
                    end.synchronize()
                    samples.append(float(start.elapsed_time(end) / iterations))

            arr = np.asarray(samples, dtype=np.float64)
            ci_low, ci_high = bootstrap_median_ci(samples, label)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            return {
                "median_ms": float(np.median(arr)),
                "ci_low_ms": ci_low,
                "ci_high_ms": ci_high,
                "mean_ms": mean,
                "std_ms": std,
                "cv_percent": 100.0 * std / mean if mean else 0.0,
                "p25_ms": float(np.percentile(arr, 25)),
                "p75_ms": float(np.percentile(arr, 75)),
                "best_ms": float(np.min(arr)),
                "samples_ms": samples,
                "iterations_per_repeat": iterations,
                "repeats": repeats,
            }


        def peak_inference_memory(model: nn.Module, inputs: torch.Tensor, iterations: int = 5) -> Dict[str, float]:
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for _ in range(iterations):
                    _ = model(inputs)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            return {
                "resident_before_mb": float(before / 1024**2),
                "peak_allocated_mb": float(peak / 1024**2),
                "peak_inference_delta_mb": float(max(0, peak - before) / 1024**2),
            }


        def compare_outputs(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, Any]:
            ref = reference.detach().float().cpu()
            cand = candidate.detach().float().cpu()
            if ref.shape != cand.shape:
                return {"allclose": False, "shape_match": False}
            diff = (ref - cand).abs()
            denom = torch.linalg.vector_norm(ref).clamp_min(1e-12)
            rel_l2 = torch.linalg.vector_norm(ref - cand) / denom
            cosine = F.cosine_similarity(ref.flatten(), cand.flatten(), dim=0)
            return {
                "shape_match": True,
                "allclose": bool(torch.allclose(ref, cand, rtol=PARITY_RTOL, atol=PARITY_ATOL)),
                "max_abs_error": float(diff.max().item()),
                "mean_abs_error": float(diff.mean().item()),
                "rmse": float(torch.sqrt(torch.mean((ref - cand) ** 2)).item()),
                "relative_l2_error": float(rel_l2.item()),
                "cosine_similarity": float(cosine.item()),
            }


        RAW_TIMING_ROWS: List[Dict[str, Any]] = []
        SELECTION_TRIALS: List[Dict[str, Any]] = []
        OPERATOR_PROFILE_ROWS: List[Dict[str, Any]] = []


        def collect_operator_profile(
            task: str,
            model_name: str,
            variant: str,
            model: nn.Module,
            inputs: torch.Tensor,
            active_steps: int = 3,
        ) -> None:
            if not RUN_OPERATOR_PROFILING:
                return
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            schedule = torch.profiler.schedule(wait=1, warmup=1, active=active_steps, repeat=1)
            with torch.profiler.profile(activities=activities, schedule=schedule) as prof:
                with torch.inference_mode():
                    for _ in range(active_steps + 2):
                        _ = model(inputs)
                        prof.step()
            torch.cuda.synchronize()
            events = sorted(
                prof.key_averages(),
                key=lambda event: float(getattr(event, "self_cuda_time_total", 0.0)),
                reverse=True,
            )[:15]
            for rank, event in enumerate(events, start=1):
                OPERATOR_PROFILE_ROWS.append({
                    "Task": task,
                    "Model": model_name,
                    "Variant": variant,
                    "Rank": rank,
                    "Operator": event.key,
                    "Calls": int(event.count),
                    "Self_CUDA_Time_us": float(getattr(event, "self_cuda_time_total", 0.0)),
                    "CUDA_Time_us": float(getattr(event, "cuda_time_total", 0.0)),
                    "Self_CPU_Time_us": float(getattr(event, "self_cpu_time_total", 0.0)),
                })


        def record_raw_samples(task: str, model: str, variant: str, phase: str, timing: Dict[str, Any]) -> None:
            for repeat_index, value in enumerate(timing["samples_ms"], start=1):
                RAW_TIMING_ROWS.append({
                    "Task": task,
                    "Model": model,
                    "Variant": variant,
                    "Phase": phase,
                    "Repeat": repeat_index,
                    "Iterations": timing["iterations_per_repeat"],
                    "Latency_ms": value,
                })


        def evaluate_plan(
            task: str,
            model_name: str,
            factory: Callable[[], nn.Module],
            x: torch.Tensor,
            reference: torch.Tensor,
            spec: PlanSpec,
            phase: str = "final",
            collect_profile: bool = False,
        ) -> Dict[str, Any]:
            cleanup_cuda()
            model, plan_input, report = build_plan(factory, x, spec)
            with torch.inference_mode():
                output = model(plan_input).detach().float().cpu()
            parity = compare_outputs(reference, output)
            timing = cuda_latency_ms(model, plan_input, f"{task}-{model_name}-{spec.key}-{phase}")
            memory = peak_inference_memory(model, plan_input)
            if collect_profile:
                collect_operator_profile(task, model_name, spec.label, model, plan_input)
            record_raw_samples(task, model_name, spec.label, phase, timing)
            result = {
                "spec": spec,
                "report": report,
                "parity": parity,
                "timing": timing,
                "memory": memory,
            }
            del model, plan_input, output
            cleanup_cuda()
            return result
        """
    ),
    markdown(
        r"""
        ## Regression-aware plan selection

        Candidate plans are first measured with short pilot samples. A custom plan is selected only when it is numerically valid and at least `MIN_CUSTOM_GAIN` faster than the best native fallback. The selected plan is then rebuilt and measured using the independent full protocol above. This makes fallback behavior part of the optimizer instead of hiding architecture-specific regressions.
        """
    ),
    code(
        r"""
        def pilot_plan(
            task: str,
            model_name: str,
            factory: Callable[[], nn.Module],
            x: torch.Tensor,
            reference: torch.Tensor,
            spec: PlanSpec,
        ) -> Dict[str, Any]:
            cleanup_cuda()
            model, plan_input, report = build_plan(factory, x, spec)
            with torch.inference_mode():
                output = model(plan_input).detach().float().cpu()
            parity = compare_outputs(reference, output)
            timing = cuda_latency_ms(
                model,
                plan_input,
                f"pilot-{task}-{model_name}-{spec.key}",
                warmup=PILOT_WARMUP_STEPS,
                iterations=PILOT_ITERATIONS,
                repeats=PILOT_REPEATS,
            )
            record_raw_samples(task, model_name, spec.label, "pilot_selection", timing)
            trial = {
                "Task": task,
                "Model": model_name,
                "Plan_Key": spec.key,
                "Plan": spec.label,
                "Role": spec.role,
                "Pilot_Median_ms": timing["median_ms"],
                "Pilot_CV_percent": timing["cv_percent"],
                "Parity": parity.get("allclose", False),
                "Relative_L2_Error": parity.get("relative_l2_error", float("nan")),
                "ConvBN_Fusions": report.conv_bn_fusions,
                "Compiled": report.compiled,
                "CUDA_Graph": report.cuda_graph,
                "Plan_Ready": (
                    report.fx_trace_ok
                    and (not spec.compile or report.compiled)
                    and (not spec.cuda_graph or report.cuda_graph)
                ),
                "Setup_Time_s": report.setup_time_s,
                "Compile_Error": report.compile_error,
                "CUDA_Graph_Error": report.cuda_graph_error,
            }
            SELECTION_TRIALS.append(trial)
            del model, plan_input, output
            cleanup_cuda()
            return trial


        def select_execution_plan(
            task: str,
            model_name: str,
            factory: Callable[[], nn.Module],
            x: torch.Tensor,
            reference: torch.Tensor,
            channels_last: bool,
        ) -> Tuple[PlanSpec, str]:
            specs = make_plan_specs(channels_last)
            candidate_keys = ["native_amp"]
            if RUN_INDUCTOR_BASELINE:
                candidate_keys.append("native_inductor")
            candidate_keys.append("fx_amp")
            if RUN_TORCH_COMPILE:
                candidate_keys.append("fx_inductor")
                if ENABLE_CUDA_GRAPHS:
                    candidate_keys.append("fx_inductor_graph")

            trials = [pilot_plan(task, model_name, factory, x, reference, specs[key]) for key in candidate_keys]
            valid = [
                trial for trial in trials
                if trial["Parity"] and trial["Plan_Ready"] and np.isfinite(trial["Pilot_Median_ms"])
            ]
            if not valid:
                selected = specs["native_amp"]
                reason = "No valid custom plan; selected native AMP fallback."
                for trial in trials:
                    trial["Selected"] = trial["Plan_Key"] == selected.key
                    trial["Selection_Reason"] = reason
                return selected, reason

            fallbacks = [trial for trial in valid if trial["Role"] == "fallback"]
            custom = [trial for trial in valid if trial["Role"] == "custom"]
            best_fallback = min(fallbacks, key=lambda row: row["Pilot_Median_ms"])
            best_custom = min(custom, key=lambda row: row["Pilot_Median_ms"]) if custom else None

            if best_custom is not None:
                required = best_fallback["Pilot_Median_ms"] * (1.0 - MIN_CUSTOM_GAIN)
                if best_custom["Pilot_Median_ms"] <= required:
                    gain = best_fallback["Pilot_Median_ms"] / best_custom["Pilot_Median_ms"]
                    reason = f"Custom pilot gain {gain:.3f}x exceeded the {MIN_CUSTOM_GAIN:.0%} guard band."
                    selected = specs[best_custom["Plan_Key"]]
                    for trial in trials:
                        trial["Selected"] = trial["Plan_Key"] == selected.key
                        trial["Selection_Reason"] = reason
                    return selected, reason

            reason = "Custom candidates did not clear the pilot guard band; selected fastest native fallback."
            selected = specs[best_fallback["Plan_Key"]]
            for trial in trials:
                trial["Selected"] = trial["Plan_Key"] == selected.key
                trial["Selection_Reason"] = reason
            return selected, reason
        """
    ),
    markdown(
        r"""
        ## Main workloads

        CNNs use equal random weights and inputs across every execution plan. The adaptive batch-size check is shared across plans. The transformer experiment is a LayerNorm-MLP microblock and is explicitly reported as a microbenchmark, not as full LLM evidence.
        """
    ),
    code(
        r"""
        def make_image_input(batch_size: int, seed: int = SEED) -> torch.Tensor:
            generator = torch.Generator(device=DEVICE)
            generator.manual_seed(seed)
            return torch.randn(batch_size, 3, 224, 224, device=DEVICE, generator=generator)


        def choose_batch_size(factory: Callable[[], nn.Module], candidates: List[int]) -> int:
            for batch in candidates:
                try:
                    cleanup_cuda()
                    model = instantiate_model(factory)
                    x = make_image_input(batch)
                    with torch.inference_mode():
                        _ = model(x)
                    torch.cuda.synchronize()
                    del model, x
                    cleanup_cuda()
                    return batch
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        print(f"Batch {batch} OOM; trying a smaller batch.")
                        cleanup_cuda()
                        continue
                    raise
            raise RuntimeError("No candidate batch size fits this GPU.")


        def _timing_columns(prefix: str, timing: Dict[str, Any]) -> Dict[str, float]:
            return {
                f"{prefix}_ms": timing["median_ms"],
                f"{prefix}_CI_Low_ms": timing["ci_low_ms"],
                f"{prefix}_CI_High_ms": timing["ci_high_ms"],
                f"{prefix}_CV_percent": timing["cv_percent"],
                f"{prefix}_Best_ms": timing["best_ms"],
            }


        def benchmark_workload(
            task: str,
            model_name: str,
            factory: Callable[[], nn.Module],
            x: torch.Tensor,
            units_per_forward: int,
            channels_last: bool,
        ) -> Dict[str, Any]:
            print(f"\n=== {task}: {model_name} | shape={tuple(x.shape)} ===")
            specs = make_plan_specs(channels_last)

            # Establish the FP32 reference and measure eager without rebuilding a second reference model.
            cleanup_cuda()
            eager_model, eager_input, eager_report = build_plan(factory, x, specs["eager_fp32"])
            params = sum(p.numel() for p in eager_model.parameters())
            with torch.inference_mode():
                reference = eager_model(eager_input).detach().float().cpu()
            eager_timing = cuda_latency_ms(eager_model, eager_input, f"{task}-{model_name}-eager-final")
            eager_memory = peak_inference_memory(eager_model, eager_input)
            collect_operator_profile(task, model_name, specs["eager_fp32"].label, eager_model, eager_input)
            record_raw_samples(task, model_name, specs["eager_fp32"].label, "final", eager_timing)
            del eager_model, eager_input
            cleanup_cuda()

            native = evaluate_plan(task, model_name, factory, x, reference, specs["native_amp"])
            if RUN_INDUCTOR_BASELINE:
                inductor = evaluate_plan(task, model_name, factory, x, reference, specs["native_inductor"])
            else:
                inductor = native

            selected_spec, selection_reason = select_execution_plan(
                task, model_name, factory, x, reference, channels_last
            )
            selected = evaluate_plan(
                task, model_name, factory, x, reference, selected_spec, collect_profile=True
            )

            speed_eager = bootstrap_speedup(
                eager_timing["samples_ms"], selected["timing"]["samples_ms"], f"{model_name}-vs-eager"
            )
            speed_native = bootstrap_speedup(
                native["timing"]["samples_ms"], selected["timing"]["samples_ms"], f"{model_name}-vs-native"
            )
            speed_inductor = bootstrap_speedup(
                inductor["timing"]["samples_ms"], selected["timing"]["samples_ms"], f"{model_name}-vs-inductor"
            )

            print(
                f"Eager {eager_timing['median_ms']:.3f} ms | Native {native['timing']['median_ms']:.3f} ms | "
                f"Inductor {inductor['timing']['median_ms']:.3f} ms | Selected {selected['timing']['median_ms']:.3f} ms"
            )
            print("Selected plan:", selected_spec.label)
            print("Policy:", selection_reason)

            row = {
                "Task": task,
                "Model": model_name,
                "Input_Shape": "x".join(str(v) for v in x.shape),
                "Batch": int(x.shape[0]),
                "Params_M": params / 1e6,
                "Selected_Plan_Key": selected_spec.key,
                "Selected_Plan": selected_spec.label,
                "Selection_Reason": selection_reason,
                "CustomDL_vs_Eager": speed_eager["point"],
                "CustomDL_vs_Eager_CI_Low": speed_eager["ci_low"],
                "CustomDL_vs_Eager_CI_High": speed_eager["ci_high"],
                "CustomDL_vs_Eager_P_gt_1": speed_eager["probability_gt_1"],
                "CustomDL_vs_Native": speed_native["point"],
                "CustomDL_vs_Native_CI_Low": speed_native["ci_low"],
                "CustomDL_vs_Native_CI_High": speed_native["ci_high"],
                "CustomDL_vs_Inductor": speed_inductor["point"],
                "CustomDL_vs_Inductor_CI_Low": speed_inductor["ci_low"],
                "CustomDL_vs_Inductor_CI_High": speed_inductor["ci_high"],
                "CustomDL_vs_Inductor_P_gt_1": speed_inductor["probability_gt_1"],
                "Throughput_Eager_units_s": units_per_forward * 1000.0 / eager_timing["median_ms"],
                "Throughput_CustomDL_units_s": units_per_forward * 1000.0 / selected["timing"]["median_ms"],
                "Eager_Peak_Delta_MB": eager_memory["peak_inference_delta_mb"],
                "Native_Peak_Delta_MB": native["memory"]["peak_inference_delta_mb"],
                "Inductor_Peak_Delta_MB": inductor["memory"]["peak_inference_delta_mb"],
                "CustomDL_Peak_Delta_MB": selected["memory"]["peak_inference_delta_mb"],
                "CustomDL_Setup_Time_s": selected["report"].setup_time_s,
                "ConvBN_Fusions": selected["report"].conv_bn_fusions,
                "Compiled": selected["report"].compiled,
                "CUDA_Graph": selected["report"].cuda_graph,
                "Native_AllClose": native["parity"].get("allclose", False),
                "Inductor_AllClose": inductor["parity"].get("allclose", False),
                "CustomDL_AllClose": selected["parity"].get("allclose", False),
                "CustomDL_MaxAbsErr": selected["parity"].get("max_abs_error", float("nan")),
                "CustomDL_MeanAbsErr": selected["parity"].get("mean_abs_error", float("nan")),
                "CustomDL_RMSE": selected["parity"].get("rmse", float("nan")),
                "CustomDL_RelativeL2": selected["parity"].get("relative_l2_error", float("nan")),
                "CustomDL_Cosine": selected["parity"].get("cosine_similarity", float("nan")),
                "Compile_Error": selected["report"].compile_error,
                "CUDA_Graph_Error": selected["report"].cuda_graph_error,
            }
            row.update(_timing_columns("Eager_FP32", eager_timing))
            row.update(_timing_columns("Native_AMP_NHWC", native["timing"]))
            row.update(_timing_columns("Inductor_AMP_NHWC", inductor["timing"]))
            row.update(_timing_columns("CustomDL_Research", selected["timing"]))
            return row


        def checkpoint_progress(rows: List[Dict[str, Any]]) -> None:
            pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, "checkpoint_results.csv"), index=False)
            pd.DataFrame(RAW_TIMING_ROWS).to_csv(
                os.path.join(RESULT_DIR, "checkpoint_raw_repeat_timings.csv"), index=False
            )
            pd.DataFrame(SELECTION_TRIALS).to_csv(
                os.path.join(RESULT_DIR, "checkpoint_plan_selection.csv"), index=False
            )
            pd.DataFrame(OPERATOR_PROFILE_ROWS).to_csv(
                os.path.join(RESULT_DIR, "checkpoint_operator_profiles.csv"), index=False
            )


        vision_results = []
        if RUN_CNN_BENCHMARKS:
            for model_name, factory in CNN_MODEL_FACTORIES.items():
                try:
                    batch = choose_batch_size(factory, BATCH_CANDIDATES)
                    x = make_image_input(batch)
                    vision_results.append(
                        benchmark_workload("Vision CNN", model_name, factory, x, batch, channels_last=True)
                    )
                    checkpoint_progress(vision_results)
                    del x
                    cleanup_cuda()
                except Exception as exc:
                    print(f"Skipping {model_name} after unrecoverable error: {exc!r}")
                    cleanup_cuda()

        vision_df = pd.DataFrame(vision_results)
        vision_df
        """
    ),
    code(
        r"""
        class TransformerMLPBlock(nn.Module):
            def __init__(self, dim: int = 768, hidden_dim: int = 3072):
                super().__init__()
                self.ln = nn.LayerNorm(dim)
                self.fc1 = nn.Linear(dim, hidden_dim)
                self.act = nn.GELU(approximate="tanh")
                self.fc2 = nn.Linear(hidden_dim, dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc2(self.act(self.fc1(self.ln(x))))


        def make_transformer_input(batch: int, seq_len: int, dim: int, seed: int = SEED) -> torch.Tensor:
            generator = torch.Generator(device=DEVICE)
            generator.manual_seed(seed)
            return torch.randn(batch, seq_len, dim, device=DEVICE, generator=generator)


        transformer_results = []
        if RUN_TRANSFORMER_MICROBENCHMARK:
            try:
                batch, seq_len, dim, hidden = 16, 128, 768, 3072
                x = make_transformer_input(batch, seq_len, dim)
                factory = lambda: TransformerMLPBlock(dim=dim, hidden_dim=hidden)
                transformer_results.append(
                    benchmark_workload(
                        "Transformer Microblock",
                        "LayerNorm-GELU-MLP",
                        factory,
                        x,
                        batch * seq_len,
                        channels_last=False,
                    )
                )
                checkpoint_progress(vision_results + transformer_results)
                del x
                cleanup_cuda()
            except Exception as exc:
                print("Skipping transformer microbenchmark after unrecoverable error:", repr(exc))
                cleanup_cuda()

        transformer_df = pd.DataFrame(transformer_results)
        transformer_df
        """
    ),
    markdown(
        r"""
        ## Optional Triton autotuning ablation

        This microbenchmark is intentionally separate from TorchInductor. It tests the package's hardware-adaptive kernel idea without introducing unsupported user-Triton/Inductor nesting. The result is an ablation, not part of the selected end-to-end path.
        """
    ),
    code(
        r"""
        triton_ablation_rows = []

        if RUN_TRITON_RELU_ABLATION and TRITON_AVAILABLE:
            @triton.autotune(
                configs=[
                    triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
                    triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
                    triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
                    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
                    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
                ],
                key=["n_elements"],
            )
            @triton.jit
            def _autotuned_relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
                offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                values = tl.load(x_ptr + offsets, mask=mask)
                tl.store(y_ptr + offsets, tl.maximum(values, 0.0), mask=mask)


            class AutotunedTritonReLU(nn.Module):
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    y = torch.empty_like(x)
                    n_elements = x.numel()
                    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
                    _autotuned_relu_kernel[grid](x, y, n_elements)
                    return y


            class TorchReLU(nn.Module):
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return F.relu(x)


            for n_elements in [2**20, 2**24]:
                x = torch.randn(n_elements, device=DEVICE, dtype=torch.float16)
                torch_model = TorchReLU().to(DEVICE)
                triton_model = AutotunedTritonReLU().to(DEVICE)
                torch_stats = cuda_latency_ms(
                    torch_model, x, f"torch-relu-{n_elements}", warmup=20, iterations=100, repeats=10
                )
                triton_stats = cuda_latency_ms(
                    triton_model, x, f"triton-relu-{n_elements}", warmup=20, iterations=100, repeats=10
                )
                parity = compare_outputs(torch_model(x), triton_model(x))
                speed = bootstrap_speedup(
                    torch_stats["samples_ms"], triton_stats["samples_ms"], f"relu-{n_elements}"
                )
                triton_ablation_rows.append({
                    "Elements": n_elements,
                    "DType": "float16",
                    "PyTorch_ms": torch_stats["median_ms"],
                    "Triton_ms": triton_stats["median_ms"],
                    "Triton_Speedup": speed["point"],
                    "Speedup_CI_Low": speed["ci_low"],
                    "Speedup_CI_High": speed["ci_high"],
                    "Parity": parity.get("allclose", False),
                })
                del x, torch_model, triton_model
                cleanup_cuda()
        elif RUN_TRITON_RELU_ABLATION:
            print("Triton ReLU ablation skipped because Triton is unavailable.")

        triton_ablation_df = pd.DataFrame(triton_ablation_rows)
        triton_ablation_df
        """
    ),
    markdown(
        r"""
        ## Optional batch-size sensitivity study

        Enable `RUN_BATCH_SCALING_STUDY` in the configuration cell for the final appendix. This repeats adaptive selection at each input shape; it is disabled by default because `max-autotune` compilation makes it substantially longer than the main experiment.
        """
    ),
    code(
        r"""
        batch_scaling_results = []
        if RUN_BATCH_SCALING_STUDY:
            factory = CNN_MODEL_FACTORIES[BATCH_SCALING_MODEL]
            for batch in BATCH_SCALING_SIZES:
                try:
                    x = make_image_input(batch, seed=SEED + batch)
                    row = benchmark_workload(
                        "Batch Scaling",
                        f"{BATCH_SCALING_MODEL}-B{batch}",
                        factory,
                        x,
                        batch,
                        channels_last=True,
                    )
                    batch_scaling_results.append(row)
                    del x
                    cleanup_cuda()
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        print(f"Skipping batch {batch}: OOM")
                        cleanup_cuda()
                        continue
                    raise

        batch_scaling_df = pd.DataFrame(batch_scaling_results)
        batch_scaling_df
        """
    ),
    markdown(
        r"""
        ## Export complete experimental evidence

        The aggregate CSV is convenient for tables. The raw timing CSV is the auditable source for uncertainty calculations. Selection trials document every considered execution plan, including fallback decisions and compile/CUDA Graph failures.
        """
    ),
    code(
        r"""
        frames = [df for df in [vision_df, transformer_df] if not df.empty]
        if not frames:
            raise RuntimeError("No benchmark results were produced.")

        all_results_df = pd.concat(frames, ignore_index=True, sort=False)
        selection_df = pd.DataFrame(SELECTION_TRIALS)
        raw_timing_df = pd.DataFrame(RAW_TIMING_ROWS)
        operator_profile_df = pd.DataFrame(OPERATOR_PROFILE_ROWS)

        output_paths = {
            "results_csv": os.path.join(RESULT_DIR, "custom_dl_optimizer_research_results.csv"),
            "results_json": os.path.join(RESULT_DIR, "custom_dl_optimizer_research_results.json"),
            "raw_timings_csv": os.path.join(RESULT_DIR, "raw_repeat_timings.csv"),
            "selection_csv": os.path.join(RESULT_DIR, "execution_plan_selection_trials.csv"),
            "operator_profile_csv": os.path.join(RESULT_DIR, "operator_profile_top_ops.csv"),
            "triton_csv": os.path.join(RESULT_DIR, "triton_relu_ablation.csv"),
            "batch_scaling_csv": os.path.join(RESULT_DIR, "batch_scaling_results.csv"),
            "metadata_json": os.path.join(RESULT_DIR, "experiment_metadata.json"),
            "pip_freeze_txt": os.path.join(RESULT_DIR, "pip_freeze.txt"),
        }

        all_results_df.to_csv(output_paths["results_csv"], index=False)
        all_results_df.to_json(output_paths["results_json"], orient="records", indent=2)
        raw_timing_df.to_csv(output_paths["raw_timings_csv"], index=False)
        selection_df.to_csv(output_paths["selection_csv"], index=False)
        operator_profile_df.to_csv(output_paths["operator_profile_csv"], index=False)
        triton_ablation_df.to_csv(output_paths["triton_csv"], index=False)
        batch_scaling_df.to_csv(output_paths["batch_scaling_csv"], index=False)

        metadata_bundle = {
            "runtime": RUNTIME_METADATA,
            "config": EXPERIMENT_CONFIG,
            "result_columns": list(all_results_df.columns),
            "notes": [
                "Final timings are distinct from pilot plan-selection timings.",
                "Compilation/setup time is excluded from steady-state forward latency and reported separately.",
                "Random inputs test numerical forward parity, not dataset-level accuracy.",
                "This benchmark alone does not establish state of the art.",
            ],
        }
        with open(output_paths["metadata_json"], "w", encoding="utf-8") as handle:
            json.dump(metadata_bundle, handle, indent=2)
        try:
            pip_freeze = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True
            )
        except Exception as exc:
            pip_freeze = f"pip freeze unavailable: {exc!r}\n"
        with open(output_paths["pip_freeze_txt"], "w", encoding="utf-8") as handle:
            handle.write(pip_freeze)

        display_cols = [
            "Model", "Batch", "Selected_Plan", "Eager_FP32_ms", "Native_AMP_NHWC_ms",
            "Inductor_AMP_NHWC_ms", "CustomDL_Research_ms", "CustomDL_vs_Eager",
            "CustomDL_vs_Native", "CustomDL_vs_Inductor", "CustomDL_AllClose",
        ]
        display(all_results_df[display_cols].round(4))
        print("Saved evidence bundle:")
        for path in output_paths.values():
            print("-", path)
        """
    ),
    markdown(
        r"""
        ## Publication figures

        The latency plot uses a logarithmic axis so the transformer microblock remains visible beside CNNs. Speedup uncertainty is shown with bootstrap 95% confidence intervals. The plan-selection heatmap exposes fallbacks rather than presenting every transformation as a win.
        """
    ),
    code(
        r"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
        plt.rcParams.update({
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        })

        models_order = all_results_df["Model"].tolist()
        x_pos = np.arange(len(models_order))
        variants = [
            ("Eager_FP32", "Eager FP32", "#4C78A8"),
            ("Native_AMP_NHWC", "AMP/NHWC", "#F28E2B"),
            ("Inductor_AMP_NHWC", "AMP/NHWC + Inductor", "#59A14F"),
            ("CustomDL_Research", "Custom-DL adaptive", "#B24745"),
        ]

        fig, ax = plt.subplots(figsize=(13, 5.8))
        width = 0.2
        for index, (prefix, label, color) in enumerate(variants):
            values = all_results_df[f"{prefix}_ms"].to_numpy(float)
            low = all_results_df[f"{prefix}_CI_Low_ms"].to_numpy(float)
            high = all_results_df[f"{prefix}_CI_High_ms"].to_numpy(float)
            errors = np.vstack([values - low, high - values])
            ax.bar(x_pos + (index - 1.5) * width, values, width, label=label, color=color, yerr=errors, capsize=2)
        ax.set_yscale("log")
        ax.set_title("Inference Latency with Bootstrap 95% Confidence Intervals")
        ax.set_ylabel("Median latency per forward pass (ms, log scale)")
        ax.set_xticks(x_pos, models_order, rotation=22, ha="right")
        ax.legend(ncol=2, frameon=True)
        fig.tight_layout()
        latency_png = os.path.join(RESULT_DIR, "figure_1_latency_with_ci.png")
        latency_pdf = os.path.join(RESULT_DIR, "figure_1_latency_with_ci.pdf")
        fig.savefig(latency_png, bbox_inches="tight")
        fig.savefig(latency_pdf, bbox_inches="tight")
        plt.show()

        fig, ax = plt.subplots(figsize=(12.5, 5.2))
        speed = all_results_df["CustomDL_vs_Inductor"].to_numpy(float)
        speed_low = all_results_df["CustomDL_vs_Inductor_CI_Low"].to_numpy(float)
        speed_high = all_results_df["CustomDL_vs_Inductor_CI_High"].to_numpy(float)
        errors = np.vstack([speed - speed_low, speed_high - speed])
        colors = ["#2F7D4A" if value >= 1.0 else "#B24745" for value in speed]
        bars = ax.bar(x_pos, speed, color=colors, yerr=errors, capsize=4)
        ax.axhline(1.0, color="black", linewidth=1.0)
        ax.set_title("Adaptive Custom-DL Speedup over AMP/NHWC + TorchInductor")
        ax.set_ylabel("Speedup (higher is better)")
        ax.set_xticks(x_pos, models_order, rotation=22, ha="right")
        for bar, value in zip(bars, speed):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}x", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        speed_png = os.path.join(RESULT_DIR, "figure_2_speedup_vs_inductor_ci.png")
        speed_pdf = os.path.join(RESULT_DIR, "figure_2_speedup_vs_inductor_ci.pdf")
        fig.savefig(speed_png, bbox_inches="tight")
        fig.savefig(speed_pdf, bbox_inches="tight")
        plt.show()

        if not selection_df.empty:
            selection_plot = selection_df.copy()
            fallback_best = selection_plot[selection_plot["Role"] == "fallback"].groupby("Model")["Pilot_Median_ms"].min()
            selection_plot["Speedup_vs_Best_Fallback"] = selection_plot.apply(
                lambda row: fallback_best.get(row["Model"], np.nan) / row["Pilot_Median_ms"], axis=1
            )
            heatmap = selection_plot.pivot(index="Plan", columns="Model", values="Speedup_vs_Best_Fallback")
            fig, ax = plt.subplots(figsize=(12.5, 4.8))
            sns.heatmap(heatmap, annot=True, fmt=".2f", center=1.0, cmap="vlag_r", linewidths=0.5, ax=ax)
            ax.set_title("Pilot Plan Ablation: Speedup vs Fastest Native Fallback")
            ax.set_xlabel("")
            ax.set_ylabel("")
            fig.tight_layout()
            selection_png = os.path.join(RESULT_DIR, "figure_3_plan_selection_heatmap.png")
            selection_pdf = os.path.join(RESULT_DIR, "figure_3_plan_selection_heatmap.pdf")
            fig.savefig(selection_png, bbox_inches="tight")
            fig.savefig(selection_pdf, bbox_inches="tight")
            plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        axes[0].bar(x_pos, all_results_df["CustomDL_Peak_Delta_MB"], color="#4C78A8")
        axes[0].set_title("Peak Additional Allocation During Inference")
        axes[0].set_ylabel("Peak allocation delta (MiB)")
        axes[0].set_xticks(x_pos, models_order, rotation=25, ha="right")
        parity_values = np.maximum(all_results_df["CustomDL_RelativeL2"].to_numpy(float), 1e-12)
        axes[1].bar(x_pos, parity_values, color="#F28E2B")
        axes[1].set_yscale("log")
        axes[1].set_title("Numerical Error vs Eager FP32")
        axes[1].set_ylabel("Relative L2 error (log scale)")
        axes[1].set_xticks(x_pos, models_order, rotation=25, ha="right")
        fig.tight_layout()
        diagnostics_png = os.path.join(RESULT_DIR, "figure_4_memory_and_parity.png")
        diagnostics_pdf = os.path.join(RESULT_DIR, "figure_4_memory_and_parity.pdf")
        fig.savefig(diagnostics_png, bbox_inches="tight")
        fig.savefig(diagnostics_pdf, bbox_inches="tight")
        plt.show()
        """
    ),
    markdown(
        r"""
        ## Paper-ready table and measured summary

        The generated prose distinguishes large gains over eager FP32 from incremental gains over strong native baselines. It reports geometric means for multiplicative speedups and never inserts a state-of-the-art claim automatically.
        """
    ),
    code(
        r"""
        def geometric_mean(values: pd.Series) -> float:
            values = pd.to_numeric(values, errors="coerce").dropna()
            values = values[values > 0]
            return float(np.exp(np.mean(np.log(values)))) if len(values) else float("nan")


        table_df = all_results_df.copy()
        table_df["Eager FP32, ms [95% CI]"] = table_df.apply(
            lambda r: f"{r.Eager_FP32_ms:.3f} [{r.Eager_FP32_CI_Low_ms:.3f}, {r.Eager_FP32_CI_High_ms:.3f}]", axis=1
        )
        table_df["Inductor, ms [95% CI]"] = table_df.apply(
            lambda r: f"{r.Inductor_AMP_NHWC_ms:.3f} [{r.Inductor_AMP_NHWC_CI_Low_ms:.3f}, {r.Inductor_AMP_NHWC_CI_High_ms:.3f}]", axis=1
        )
        table_df["Custom-DL, ms [95% CI]"] = table_df.apply(
            lambda r: f"{r.CustomDL_Research_ms:.3f} [{r.CustomDL_Research_CI_Low_ms:.3f}, {r.CustomDL_Research_CI_High_ms:.3f}]", axis=1
        )
        table_df["Speedup vs Inductor [95% CI]"] = table_df.apply(
            lambda r: f"{r.CustomDL_vs_Inductor:.3f} [{r.CustomDL_vs_Inductor_CI_Low:.3f}, {r.CustomDL_vs_Inductor_CI_High:.3f}]", axis=1
        )
        paper_cols = [
            "Model", "Batch", "Selected_Plan", "Eager FP32, ms [95% CI]",
            "Inductor, ms [95% CI]", "Custom-DL, ms [95% CI]",
            "CustomDL_vs_Eager", "Speedup vs Inductor [95% CI]", "CustomDL_AllClose",
        ]
        paper_table = table_df[paper_cols].rename(columns={
            "Selected_Plan": "Selected plan",
            "CustomDL_vs_Eager": "Speedup vs eager",
            "CustomDL_AllClose": "Parity",
        })
        latex_table = paper_table.to_latex(index=False, escape=True, float_format="%.3f")
        latex_path = os.path.join(RESULT_DIR, "paper_table_results_with_ci.tex")
        with open(latex_path, "w", encoding="utf-8") as handle:
            handle.write(latex_table)
        print(latex_table)

        vision_only = all_results_df[all_results_df["Task"] == "Vision CNN"].copy()
        summary_scope = vision_only if not vision_only.empty else all_results_df
        best_index = summary_scope["CustomDL_vs_Eager"].idxmax()
        best_row = summary_scope.loc[best_index]
        eager_geomean = geometric_mean(summary_scope["CustomDL_vs_Eager"])
        native_geomean = geometric_mean(summary_scope["CustomDL_vs_Native"])
        inductor_geomean = geometric_mean(summary_scope["CustomDL_vs_Inductor"])
        parity_count = int(summary_scope["CustomDL_AllClose"].sum())
        fallback_count = int(summary_scope["Selected_Plan_Key"].isin(["native_amp", "native_inductor"]).sum())

        measured_summary = (
            f"On {RUNTIME_METADATA['gpu_name']}, the adaptive Custom-DL pipeline achieved up to "
            f"{best_row['CustomDL_vs_Eager']:.2f}x speedup over PyTorch eager FP32 on {best_row['Model']} "
            f"and a {eager_geomean:.2f}x geometric-mean speedup across {len(summary_scope)} CNN workloads. "
            f"Against native AMP/NHWC and AMP/NHWC plus TorchInductor, geometric-mean speedups were "
            f"{native_geomean:.2f}x and {inductor_geomean:.2f}x, respectively. The regression-aware "
            f"policy selected a native fallback for {fallback_count} of {len(summary_scope)} CNNs when "
            f"custom plans did not clear the {MIN_CUSTOM_GAIN:.0%} pilot guard band. Numerical parity "
            f"passed for {parity_count} of {len(summary_scope)} CNNs at rtol={PARITY_RTOL:g} and atol={PARITY_ATOL:g}. "
            "These results are specific to the recorded environment and do not by themselves establish state of the art."
        )
        summary_path = os.path.join(RESULT_DIR, "generated_measured_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write(measured_summary)
        print("\nMeasured paper summary:\n", measured_summary)
        print("\nLaTeX table:", latex_path)
        print("Summary:", summary_path)
        """
    ),
    markdown(
        r"""
        ## Interpretation checklist

        Before copying numbers into the paper:

        1. Confirm every `CustomDL_AllClose` value is `True` and inspect relative L2 error.
        2. Use the confidence interval, not only the point estimate, when describing small gains.
        3. Treat a confidence interval crossing 1.0x as inconclusive rather than a win.
        4. Report selected native fallbacks as part of the profile-guided policy.
        5. Keep compile/setup time separate from steady-state latency and state the serving assumption.
        6. Run the optional batch-size study and dataset-level accuracy validation before a full submission.
        7. Add controlled TensorRT/Torch-TensorRT comparisons before making any SOTA claim.

        The output directory now contains the complete evidence bundle needed to update the paper: aggregate results, raw samples, selection trials, environment metadata, PNG/PDF figures, a LaTeX table, and a measured summary paragraph.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT} with {len(cells)} cells")
