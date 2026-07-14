import torch
import torch.nn as nn

from custom_dl_optimizer import OptimizationConfig, Optimizer


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

    config = OptimizationConfig(
        enable_compile=torch.cuda.is_available(),
        expected_calls=10_000,
        selection_iterations=5,
    )
    optimizer = Optimizer(device=device, config=config)
    result = optimizer.optimize(model, inputs)

    with torch.inference_mode():
        baseline = model(inputs).float()
        output = result(inputs).float().cpu()

    print("Output shape:", tuple(output.shape))
    print("Parity:", torch.allclose(baseline, output, rtol=5e-2, atol=5e-2))
    print("Selected plan:", result.selected_plan)
    print("Reason:", result.report.selection_reason)
    selected = result.report.selected_candidate
    if selected is not None:
        print("Median / P90 (ms):", selected.latency_ms, selected.latency_p90_ms)
        print("Cold start (s):", selected.first_call_time_s)
        print("Projected total (ms):", selected.projected_total_ms)
        print("Break-even calls:", selected.break_even_calls_vs_baseline)
    result.save_report("custom_dl_optimizer_report.json")


if __name__ == "__main__":
    main()
