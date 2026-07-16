import torch
import torch.nn as nn

from custom_dl_optimizer import (
    DeploymentConstraints,
    ExecutionTarget,
    InferenceOptimizer,
    MeasurementPolicy,
    OptimizationPolicy,
)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10),
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN().eval()
    inputs = torch.randn(8, 3, 224, 224)

    policy = OptimizationPolicy(
        objective="lifecycle_latency",
        enable_compile=torch.cuda.is_available(),
        measurement=MeasurementPolicy(iterations=5),
        constraints=DeploymentConstraints(expected_calls=10_000),
    )
    optimizer = InferenceOptimizer(target=ExecutionTarget(device), policy=policy)
    decision = optimizer.select_signature(model, inputs)

    with torch.inference_mode():
        baseline = model(inputs).float()
        output = decision(inputs).float().cpu()

    print("Output shape:", tuple(output.shape))
    print("Parity:", torch.allclose(baseline, output, rtol=5e-2, atol=5e-2))
    print("Selected plan:", decision.selected_plan)
    print("Reason:", decision.report.selection_reason)
    selected = decision.report.selected_candidate
    if selected is not None:
        print("Median / P90 (ms):", selected.latency_ms, selected.latency_p90_ms)
        print("Cold start (s):", selected.first_call_time_s)
        print("Projected total (ms):", selected.projected_total_ms)
        print("Break-even calls:", selected.break_even_calls_vs_baseline)
        print(
            "Selection cost CI (ms):",
            selected.selection_cost_ci_low_ms,
            selected.selection_cost_ci_high_ms,
        )
    decision.save_report("custom_dl_optimizer_report.json")


if __name__ == "__main__":
    main()
