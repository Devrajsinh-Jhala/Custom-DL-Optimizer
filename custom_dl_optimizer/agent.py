from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from .optimizer import InferenceOptimizer, OptimizationDecision
from .workload import WorkloadCase, WorkloadProfile


@dataclass
class _RegisteredWorkload:
    model: nn.Module
    profile: WorkloadProfile
    description: str


class OptimizationAgentToolkit:
    """Dependency-neutral function tools for in-process agent orchestration."""

    def __init__(self, optimizer: InferenceOptimizer | None = None) -> None:
        self.optimizer = optimizer or InferenceOptimizer()
        self._workloads: dict[str, _RegisteredWorkload] = {}
        self._results: dict[str, OptimizationDecision] = {}

    def register_workload(
        self,
        name: str,
        model: nn.Module,
        *example_args: Any,
        description: str = "",
        **example_kwargs: Any,
    ) -> None:
        normalized = name.strip()
        if not normalized:
            raise ValueError("workload name must not be empty")
        if normalized in self._workloads:
            raise ValueError(f"A workload named {normalized!r} is already registered")
        if not example_args and not example_kwargs:
            raise ValueError("At least one example input is required")
        self._workloads[normalized] = _RegisteredWorkload(
            model=model,
            profile=WorkloadProfile(
                name=normalized,
                cases=(
                    WorkloadCase(
                        name="default",
                        args=example_args,
                        kwargs=example_kwargs,
                    ),
                ),
                expected_calls=self.optimizer.config.expected_calls,
            ),
            description=description,
        )

    def register_workload_profile(
        self,
        name: str,
        model: nn.Module,
        profile: WorkloadProfile,
        *,
        description: str = "",
    ) -> None:
        normalized = name.strip()
        if not normalized:
            raise ValueError("workload name must not be empty")
        if normalized in self._workloads:
            raise ValueError(f"A workload named {normalized!r} is already registered")
        self._workloads[normalized] = _RegisteredWorkload(
            model=model,
            profile=profile,
            description=description,
        )

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return portable JSON schemas accepted by common tool-calling agents."""

        no_arguments = {"type": "object", "properties": {}, "additionalProperties": False}
        workload_argument = {
            "type": "object",
            "properties": {
                "workload": {
                    "type": "string",
                    "description": "Name of a workload registered by the host application.",
                }
            },
            "required": ["workload"],
            "additionalProperties": False,
        }
        return [
            {
                "name": "custom_dl_inspect_runtime",
                "description": "Inspect available PyTorch, CUDA, Triton, and compiler capabilities.",
                "parameters": no_arguments,
            },
            {
                "name": "custom_dl_list_workloads",
                "description": "List optimization workloads explicitly registered by the host.",
                "parameters": no_arguments,
            },
            {
                "name": "custom_dl_optimize",
                "description": "Optimize one registered workload and return measured plan evidence.",
                "parameters": workload_argument,
            },
            {
                "name": "custom_dl_get_report",
                "description": "Return the latest optimization report for a registered workload.",
                "parameters": workload_argument,
            },
        ]

    def invoke(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke a declared tool without evaluating code supplied by the caller."""

        arguments = arguments or {}
        if tool_name == "custom_dl_inspect_runtime":
            return self.optimizer.inspect_runtime().as_dict()
        if tool_name == "custom_dl_list_workloads":
            return {
                "workloads": [
                    {"name": name, "description": workload.description}
                    for name, workload in sorted(self._workloads.items())
                ]
            }

        workload_name = arguments.get("workload")
        if not isinstance(workload_name, str) or workload_name not in self._workloads:
            raise KeyError(f"Unknown registered workload: {workload_name!r}")
        if tool_name == "custom_dl_optimize":
            workload = self._workloads[workload_name]
            result = self.optimizer.select(
                workload.model,
                workload.profile,
            )
            self._results[workload_name] = result
            return result.report.as_dict()
        if tool_name == "custom_dl_get_report":
            if workload_name not in self._results:
                raise RuntimeError(f"Workload {workload_name!r} has not been optimized yet")
            return self._results[workload_name].report.as_dict()
        raise KeyError(f"Unknown optimization tool: {tool_name!r}")
