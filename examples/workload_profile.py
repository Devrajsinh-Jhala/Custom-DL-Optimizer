from pathlib import Path

import torch
import torch.nn as nn

from custom_dl_optimizer import (
    DeploymentConstraints,
    ExecutionTarget,
    InferenceOptimizer,
    OptimizationPolicy,
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
optimizer = InferenceOptimizer(
    target=ExecutionTarget(device),
    policy=OptimizationPolicy(
        objective="lifecycle_latency",
        enable_compile=torch.cuda.is_available(),
        dynamic_shapes=True,
        plan_cache_dir=str(Path(".custom-dl-cache")),
        constraints=DeploymentConstraints(
            expected_calls=50_000,
            max_first_call_time_s=60,
            min_speedup=1.02,
        ),
    ),
)
decision = optimizer.select(TinyClassifier().eval(), profile)
decision.save_bundle("artifacts/workload-profile")

print(decision.selected_plan)
print(decision.report.selection_reason)
print("cache_hit", decision.report.cache_hit)
for warning in decision.report.warnings:
    print("warning", warning)
for candidate in decision.report.candidates:
    print(
        candidate.name,
        candidate.latency_ms,
        candidate.latency_p99_ms,
        candidate.projected_total_ms,
        candidate.selection_cost_ci_low_ms,
        candidate.selection_cost_ci_high_ms,
        candidate.constraint_violations,
    )
