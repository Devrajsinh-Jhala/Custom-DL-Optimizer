from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .cache import PlanCache, create_plan_cache_key
from .config import OptimizationConfig
from .core.engine import AutoOptimizer
from .policy import ExecutionTarget, OptimizationPolicy
from .providers import BuiltPlan, ExecutionProvider
from .report import OptimizationReport
from .runtime import RuntimeCapabilities, inspect_runtime
from .workload import WorkloadCase, WorkloadProfile


@dataclass
class OptimizationDecision:
    """Selected executable plan and the evidence supporting that decision."""

    plan: BuiltPlan
    report: OptimizationReport

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.plan.runner(*args, **kwargs)

    @property
    def runner(self) -> Any:
        return self.plan.runner

    @property
    def module(self) -> nn.Module:
        return self.plan.as_module()

    @property
    def selected_plan(self) -> str:
        return self.report.selected_plan

    def close(self) -> None:
        self.plan.close()

    def save_report(self, path: str | Path) -> Path:
        return self.report.save(path)

    def save_bundle(self, directory: str | Path) -> Path:
        """Save a versioned decision manifest and its evidence report."""

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        report_path = self.report.save(destination / "report.json")
        manifest = {
            "schema_version": 3,
            "selected_plan": self.selected_plan,
            "runner_class": (
                f"{type(self.plan.runner).__module__}."
                f"{type(self.plan.runner).__qualname__}"
            ),
            "report": report_path.name,
            "cache_key": self.report.cache_key,
            "artifacts": list(self.plan.artifacts),
            "metadata": self.plan.metadata,
            "executable_serialized": False,
            "note": (
                "The provider owns executable serialization. This bundle records the "
                "selection decision, evidence, and provider artifact references."
            ),
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination


class InferenceOptimizer:
    """Select an auditable inference plan for a target and workload."""

    def __init__(
        self,
        *,
        target: ExecutionTarget | None = None,
        policy: OptimizationPolicy | None = None,
        providers: tuple[ExecutionProvider, ...] = (),
        cache: PlanCache | None = None,
        device: str | torch.device | None = None,
        config: OptimizationConfig | None = None,
    ) -> None:
        if target is not None and device is not None:
            raise ValueError("Pass target or device, not both")
        if policy is not None and config is not None:
            raise ValueError("Pass policy or the v2 config compatibility argument, not both")
        self.target = target or ExecutionTarget(device=device)
        self._legacy_config_supplied = config is not None
        self.policy = policy or (
            OptimizationPolicy.from_engine_config(config)
            if config is not None
            else OptimizationPolicy()
        )
        self.config = self.policy.to_engine_config()
        self.device = self.target.resolve_device()
        self.cache = cache or (
            PlanCache(self.policy.plan_cache_dir)
            if self.policy.plan_cache_dir is not None
            else None
        )
        self._providers: dict[str, ExecutionProvider] = {}
        for provider in providers:
            self.register_provider(provider)

    @property
    def providers(self) -> tuple[ExecutionProvider, ...]:
        return tuple(self._providers.values())

    def register_provider(self, provider: ExecutionProvider) -> None:
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

    def select(
        self,
        model: nn.Module,
        workload: WorkloadProfile,
    ) -> OptimizationDecision:
        """Select one plan for a representative weighted workload."""

        expected_calls = None
        if self.policy.objective == "lifecycle_latency" or (
            self._legacy_config_supplied and workload.expected_calls is not None
        ):
            expected_calls = workload.expected_calls
            if expected_calls is None:
                expected_calls = self.policy.constraints.expected_calls
            if expected_calls is None:
                raise ValueError("lifecycle_latency selection requires expected_calls")
        if expected_calls != workload.expected_calls:
            workload = replace(workload, expected_calls=expected_calls)

        runtime = self.inspect_runtime()
        cache_key = ""
        record = None
        artifact_root = None
        cache_started = time.perf_counter()
        if self.cache is not None:
            cache_key = create_plan_cache_key(
                model,
                workload,
                self.config,
                runtime,
                self.providers,
            )
            artifact_root = self.cache.root / "artifacts" / cache_key
            if self.policy.reuse_cached_plan:
                record = self.cache.load(cache_key)
        cache_lookup_time_s = time.perf_counter() - cache_started

        engine = AutoOptimizer(
            model,
            device=self.device,
            config=self.config,
            policy=self.policy,
            target=self.target,
            providers=self.providers,
            cache_key=cache_key,
            artifact_root=artifact_root,
        )
        module, report = engine.optimize_workload_with_report(
            workload,
            preferred_plan=record.selected_plan if record is not None else None,
            cached_latency_ms=record.latency_ms if record is not None else None,
        )
        report.cache_lookup_time_s = cache_lookup_time_s
        report.optimization_time_s += cache_lookup_time_s
        selected = report.selected_candidate
        if self.cache is not None and selected is not None and selected.latency_ms is not None:
            report.cache_record_path = str(self.cache.record_path(cache_key))
            if not report.cache_hit:
                self.cache.save(
                    key=cache_key,
                    selected_plan=report.selected_plan,
                    latency_ms=selected.latency_ms,
                    report=report.as_dict(),
                )
        plan = BuiltPlan(
            runner=module,
            artifacts=tuple(selected.artifacts if selected is not None else ()),
            metadata={
                "selected_plan": report.selected_plan,
                "provider": selected.provider_metadata if selected is not None else {},
                "policy": asdict(self.policy),
            },
        )
        return OptimizationDecision(plan=plan, report=report)

    def select_signature(
        self,
        model: nn.Module,
        *example_args: Any,
        **example_kwargs: Any,
    ) -> OptimizationDecision:
        workload = WorkloadProfile(
            name="single-signature",
            cases=(WorkloadCase(name="default", args=example_args, kwargs=example_kwargs),),
            expected_calls=self.policy.constraints.expected_calls,
        )
        return self.select(model, workload)

    # v2 method names remain for one migration cycle but return the v3 decision object.
    def optimize(self, model: nn.Module, *args: Any, **kwargs: Any) -> OptimizationDecision:
        return self.select_signature(model, *args, **kwargs)

    def optimize_workload(
        self,
        model: nn.Module,
        profile: WorkloadProfile,
    ) -> OptimizationDecision:
        return self.select(model, profile)


OptimizationResult = OptimizationDecision
Optimizer = InferenceOptimizer
