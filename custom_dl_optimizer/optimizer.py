from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import OptimizationConfig
from .core.engine import AutoOptimizer
from .providers import CandidateProvider
from .report import OptimizationReport
from .runtime import RuntimeCapabilities, inspect_runtime


@dataclass
class OptimizationResult:
    """Selected module and the evidence used to select it."""

    module: nn.Module
    report: OptimizationReport

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    @property
    def selected_plan(self) -> str:
        return self.report.selected_plan

    def save_report(self, path: str | Path) -> Path:
        return self.report.save(path)


class Optimizer:
    """Optimize PyTorch modules and return a reusable, auditable result."""

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        config: OptimizationConfig | None = None,
        providers: tuple[CandidateProvider, ...] = (),
    ) -> None:
        self.device = device
        self.config = config or OptimizationConfig()
        self._providers: dict[str, CandidateProvider] = {}
        for provider in providers:
            self.register_provider(provider)

    @property
    def providers(self) -> tuple[CandidateProvider, ...]:
        return tuple(self._providers.values())

    def register_provider(self, provider: CandidateProvider) -> None:
        name = provider.name.strip()
        if not name:
            raise ValueError("provider.name must not be empty")
        if name in {"eager_fp32", "native", "fx", "fx_inductor"}:
            raise ValueError(f"{name!r} is reserved by a built-in candidate")
        if name in self._providers:
            raise ValueError(f"A provider named {name!r} is already registered")
        self._providers[name] = provider

    def inspect_runtime(self) -> RuntimeCapabilities:
        return inspect_runtime(self.device)

    def optimize(
        self,
        model: nn.Module,
        *example_args: Any,
        **example_kwargs: Any,
    ) -> OptimizationResult:
        engine = AutoOptimizer(
            model,
            device=self.device,
            config=self.config,
            providers=self.providers,
        )
        module, report = engine.optimize_with_report(*example_args, **example_kwargs)
        return OptimizationResult(module=module, report=report)
