from pathlib import Path

import torch
import torch.nn as nn

from custom_dl_optimizer import (
    OptimizationConfig,
    Optimizer,
    WorkloadCase,
    WorkloadProfile,
)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
profile = WorkloadProfile(
    name="classifier-traffic",
    expected_calls=50_000,
    cases=(
        WorkloadCase("batch-1", args=(torch.randn(1, 128),), weight=70),
        WorkloadCase("batch-8", args=(torch.randn(8, 128),), weight=25),
        WorkloadCase("batch-32", args=(torch.randn(32, 128),), weight=5),
    ),
)
optimizer = Optimizer(
    device=device,
    config=OptimizationConfig(
        enable_compile=torch.cuda.is_available(),
        dynamic_shapes=True,
        plan_cache_dir=str(Path(".custom-dl-cache")),
        max_first_call_time_s=60,
        min_speedup=1.02,
    ),
)
result = optimizer.optimize_workload(TinyClassifier().eval(), profile)
result.save_bundle("artifacts/workload-profile")

print(result.selected_plan)
print(result.report.selection_reason)
print("cache_hit", result.report.cache_hit)
for warning in result.report.warnings:
    print("warning", warning)
for candidate in result.report.candidates:
    print(
        candidate.name,
        candidate.latency_ms,
        candidate.latency_p99_ms,
        candidate.projected_total_ms,
        candidate.constraint_violations,
    )
