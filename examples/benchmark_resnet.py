import time

import torch
from torchvision.models import resnet50

from custom_dl_optimizer import AutoOptimizer, OptimizationConfig


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
    config = OptimizationConfig(
        enable_compile=torch.cuda.is_available(),
        compile_mode="max-autotune",
    )
    optimizer = AutoOptimizer(model, device=device, config=config)
    optimized, report = optimizer.optimize_with_report(inputs)
    args, kwargs = optimizer.prepare_inputs(
        inputs,
        channels_last=report.channels_last,
    )
    optimized_ms = benchmark(optimized, args[0])

    print(f"Baseline latency: {baseline_ms:.3f} ms")
    print(f"Optimized latency: {optimized_ms:.3f} ms")
    print(f"Speedup: {baseline_ms / optimized_ms:.2f}x")
    print(f"Selected plan: {report.selected_plan}")


if __name__ == "__main__":
    main()
