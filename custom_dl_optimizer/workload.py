from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkloadCase:
    """One representative serving signature and its relative traffic weight."""

    name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def __post_init__(self) -> None:
        normalized = self.name.strip()
        if not normalized:
            raise ValueError("workload case name must not be empty")
        if not self.args and not self.kwargs:
            raise ValueError("workload case must contain at least one input")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("workload case weight must be finite and positive")
        object.__setattr__(self, "name", normalized)
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", dict(self.kwargs))


@dataclass(frozen=True)
class WorkloadProfile:
    """A weighted distribution of inputs used for correctness and plan selection."""

    cases: tuple[WorkloadCase, ...]
    name: str = "workload"
    expected_calls: int | None = None

    def __post_init__(self) -> None:
        cases = tuple(self.cases)
        if not cases:
            raise ValueError("workload profile must contain at least one case")
        names = [case.name for case in cases]
        if len(set(names)) != len(names):
            raise ValueError("workload case names must be unique")
        if self.expected_calls is not None and self.expected_calls < 1:
            raise ValueError("expected_calls must be at least 1 when provided")
        if self.expected_calls is not None and self.expected_calls < len(cases):
            raise ValueError("expected_calls must cover every workload case at least once")
        normalized = self.name.strip()
        if not normalized:
            raise ValueError("workload profile name must not be empty")
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "name", normalized)

    @property
    def normalized_weights(self) -> dict[str, float]:
        total = sum(case.weight for case in self.cases)
        return {case.name: case.weight / total for case in self.cases}

    @classmethod
    def single(
        cls,
        *args: Any,
        name: str = "default",
        expected_calls: int | None = None,
        **kwargs: Any,
    ) -> WorkloadProfile:
        return cls(
            cases=(WorkloadCase(name=name, args=args, kwargs=kwargs),),
            expected_calls=expected_calls,
        )
