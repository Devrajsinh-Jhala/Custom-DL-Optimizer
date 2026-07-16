import time

import torch
from torchvision.models import resnet50

from custom_dl_optimizer import (
    DeploymentConstraints,
    ExecutionTarget,
    InferenceOptimizer,
    OptimizationPolicy,
)


def benchmark(model, inputs, iterations=50, warmup=10):
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)

        if inputs.is_cuda:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                model(inputs)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / iterations

        start_time = time.perf_counter()
        for _ in range(iterations):
            model(inputs)
        return (time.perf_counter() - start_time) * 1000 / iterations


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32

    model = resnet50(weights=None).eval()
    inputs = torch.randn(batch_size, 3, 224, 224)
    baseline_model = model.to(device)
    baseline_inputs = inputs.to(device)

    baseline_ms = benchmark(baseline_model, baseline_inputs)
    policy = OptimizationPolicy(
        objective="lifecycle_latency",
        enable_compile=torch.cuda.is_available(),
        compile_mode="max-autotune",
        constraints=DeploymentConstraints(expected_calls=50_000),
    )
    optimizer = InferenceOptimizer(target=ExecutionTarget(device), policy=policy)
    decision = optimizer.select_signature(model, inputs)
    optimized_ms = benchmark(decision, inputs)

    print(f"Baseline latency: {baseline_ms:.3f} ms")
    print(f"Optimized latency: {optimized_ms:.3f} ms")
    print(f"Speedup: {baseline_ms / optimized_ms:.2f}x")
    print(f"Selected plan: {decision.selected_plan}")
    selected = decision.report.selected_candidate
    if selected is not None:
        print(
            "Selection median / P90: "
            f"{selected.latency_ms:.3f} / {selected.latency_p90_ms:.3f} ms"
        )
        print(
            "Setup / first call: "
            f"{selected.setup_time_s:.3f} / {selected.first_call_time_s:.3f} s"
        )
        print(f"Break-even calls: {selected.break_even_calls_vs_baseline}")
        print(
            "Selection cost CI: "
            f"{selected.selection_cost_ci_low_ms:.3f} / "
            f"{selected.selection_cost_ci_high_ms:.3f} ms"
        )


if __name__ == "__main__":
    main()
