from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .cache import PlanCache, create_plan_cache_key
from .config import OptimizationConfig
from .core.engine import AutoOptimizer
from .providers import CandidateProvider
from .report import OptimizationReport
from .runtime import RuntimeCapabilities, inspect_runtime
from .workload import WorkloadCase, WorkloadProfile


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

    def save_bundle(self, directory: str | Path) -> Path:
        """Save a portable decision manifest and report without claiming executable export."""

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        report_path = self.report.save(destination / "report.json")
        manifest = {
            "schema_version": 1,
            "selected_plan": self.selected_plan,
            "module_class": (
                f"{type(self.module).__module__}.{type(self.module).__qualname__}"
            ),
            "report": report_path.name,
            "cache_key": self.report.cache_key,
            "executable_serialized": False,
            "note": (
                "This bundle records the decision and evidence. Executable serialization "
                "is owned by the selected backend provider."
            ),
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination


class Optimizer:
    """Optimize PyTorch modules and return a reusable, auditable result."""

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        config: OptimizationConfig | None = None,
        providers: tuple[CandidateProvider, ...] = (),
        cache: PlanCache | None = None,
    ) -> None:
        self.device = device
        self.config = config or OptimizationConfig()
        self.cache = cache or (
            PlanCache(self.config.plan_cache_dir)
            if self.config.plan_cache_dir is not None
            else None
        )
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
        profile = WorkloadProfile(
            name="single-signature",
            cases=(
                WorkloadCase(
                    name="default",
                    args=example_args,
                    kwargs=example_kwargs,
                ),
            ),
            expected_calls=self.config.expected_calls,
        )
        return self.optimize_workload(model, profile)

    def optimize_workload(
        self,
        model: nn.Module,
        profile: WorkloadProfile,
    ) -> OptimizationResult:
        """Optimize one model against a weighted workload profile."""

        runtime = self.inspect_runtime()
        cache_key = ""
        record = None
        artifact_root = None
        cache_started = time.perf_counter()
        if self.cache is not None:
            cache_key = create_plan_cache_key(
                model,
                profile,
                self.config,
                runtime,
                self.providers,
            )
            artifact_root = self.cache.root / "artifacts" / cache_key
            if self.config.reuse_cached_plan:
                record = self.cache.load(cache_key)
        cache_lookup_time_s = time.perf_counter() - cache_started
        engine = AutoOptimizer(
            model,
            device=self.device,
            config=self.config,
            providers=self.providers,
            cache_key=cache_key,
            artifact_root=artifact_root,
        )
        module, report = engine.optimize_workload_with_report(
            profile,
            preferred_plan=record.selected_plan if record is not None else None,
            cached_latency_ms=record.latency_ms if record is not None else None,
        )
        report.cache_lookup_time_s = cache_lookup_time_s
        report.optimization_time_s += cache_lookup_time_s
        if self.cache is not None:
            selected = report.selected_candidate
            if selected is not None and selected.latency_ms is not None:
                report.cache_record_path = str(self.cache.record_path(cache_key))
                if not report.cache_hit:
                    self.cache.save(
                        key=cache_key,
                        selected_plan=report.selected_plan,
                        latency_ms=selected.latency_ms,
                        report=report.as_dict(),
                    )
        return OptimizationResult(module=module, report=report)
