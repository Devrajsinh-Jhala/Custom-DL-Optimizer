import torch
import torch.nn as nn

from custom_dl_optimizer import OptimizationAgentToolkit, OptimizationConfig, Optimizer


def main():
    model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16)).eval()
    sample = torch.randn(8, 16)
    optimizer = Optimizer(
        device="cuda" if torch.cuda.is_available() else "cpu",
        config=OptimizationConfig(
            enable_profiling=False,
            selection_iterations=5,
        ),
    )
    toolkit = OptimizationAgentToolkit(optimizer)
    toolkit.register_workload(
        "mlp-b8",
        model,
        sample,
        description="Small MLP inference workload",
    )

    print(toolkit.tool_schemas())
    print(toolkit.invoke("custom_dl_list_workloads"))
    print(toolkit.invoke("custom_dl_optimize", {"workload": "mlp-b8"}))


if __name__ == "__main__":
    main()
