import torch
import torch.nn as nn

from custom_dl_optimizer import AutoOptimizer, OptimizationConfig


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
        selection_iterations=5,
    )
    optimizer = AutoOptimizer(model, device=device, config=config)
    optimized, report = optimizer.optimize_with_report(inputs)
    args, kwargs = optimizer.prepare_inputs(
        inputs,
        channels_last=report.channels_last,
    )

    with torch.inference_mode():
        baseline = model(inputs).float()
        output = optimized(*args, **kwargs).float().cpu()

    print("Output shape:", tuple(output.shape))
    print("Parity:", torch.allclose(baseline, output, rtol=5e-2, atol=5e-2))
    print("Selected plan:", report.selected_plan)
    print("Reason:", report.selection_reason)


if __name__ == "__main__":
    main()
