from __future__ import annotations

import hashlib
import io
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import OptimizationConfig
from .runtime import RuntimeCapabilities
from .workload import WorkloadProfile

_CACHE_SCHEMA_VERSION = 3


def _package_version() -> str:
    try:
        return version("custom-dl-optimizer")
    except PackageNotFoundError:
        return "source"


def _update_text(digest: Any, value: Any) -> None:
    digest.update(str(value).encode("utf-8", errors="backslashreplace"))
    digest.update(b"\0")


def _update_value_signature(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        _update_text(
            digest,
            (
                "tensor",
                tuple(value.shape),
                tuple(value.stride()),
                str(value.dtype),
                bool(value.requires_grad),
            ),
        )
        return
    if isinstance(value, dict):
        _update_text(digest, "dict")
        for key in sorted(value, key=str):
            _update_text(digest, key)
            _update_value_signature(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        _update_text(digest, type(value).__name__)
        for item in value:
            _update_value_signature(digest, item)
        return
    if value is None or isinstance(value, (bool, int, float, str)):
        _update_text(digest, (type(value).__name__, value))
        return
    _update_text(digest, (type(value).__module__, type(value).__qualname__))


def _update_model(digest: Any, model: nn.Module) -> None:
    _update_text(digest, (type(model).__module__, type(model).__qualname__))
    _update_text(digest, repr(model))
    for name, value in sorted(model.state_dict().items()):
        _update_text(digest, name)
        if not isinstance(value, torch.Tensor):
            _update_text(digest, repr(value))
            continue
        tensor = value.detach().contiguous().cpu()
        _update_text(digest, (tuple(tensor.shape), str(tensor.dtype)))
        try:
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        except (RuntimeError, TypeError):
            buffer = io.BytesIO()
            torch.save(tensor, buffer)
            digest.update(buffer.getvalue())
        digest.update(b"\0")


def _provider_identity(provider: Any) -> dict[str, Any]:
    identity = getattr(provider, "cache_identity", None)
    if callable(identity):
        supplied = identity()
        if not isinstance(supplied, dict):
            raise TypeError("provider.cache_identity() must return a dictionary")
        return supplied
    return {
        "name": str(getattr(provider, "name", "")),
        "class": f"{type(provider).__module__}.{type(provider).__qualname__}",
    }


def create_plan_cache_key(
    model: nn.Module,
    profile: WorkloadProfile,
    config: OptimizationConfig,
    runtime: RuntimeCapabilities,
    providers: tuple[Any, ...] = (),
) -> str:
    """Return a content-addressed key for a model, workload, policy, and runtime."""

    digest = hashlib.sha256()
    _update_text(digest, _CACHE_SCHEMA_VERSION)
    _update_model(digest, model)
    _update_text(digest, profile.name)
    _update_text(digest, profile.expected_calls)
    for case in profile.cases:
        _update_text(digest, (case.name, case.weight))
        _update_value_signature(digest, (case.args, case.kwargs))

    config_payload = asdict(config)
    for ignored in (
        "copy_model",
        "verbose",
        "plan_cache_dir",
        "reuse_cached_plan",
        "cache_validation_iterations",
        "cache_max_latency_regression",
    ):
        config_payload.pop(ignored, None)
    _update_text(digest, json.dumps(config_payload, sort_keys=True, default=str))
    _update_text(digest, json.dumps(runtime.as_dict(), sort_keys=True, default=str))
    provider_payload = [_provider_identity(provider) for provider in providers]
    _update_text(digest, json.dumps(provider_payload, sort_keys=True, default=str))
    return digest.hexdigest()


@dataclass(frozen=True)
class PlanCacheRecord:
    key: str
    selected_plan: str
    latency_ms: float
    report: dict[str, Any]
    created_at: str
    package_version: str
    schema_version: int = _CACHE_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlanCacheRecord:
        return cls(
            key=str(payload["key"]),
            selected_plan=str(payload["selected_plan"]),
            latency_ms=float(payload["latency_ms"]),
            report=dict(payload.get("report", {})),
            created_at=str(payload["created_at"]),
            package_version=str(payload.get("package_version", "unknown")),
            schema_version=int(payload.get("schema_version", 0)),
        )


class PlanCache:
    """Persistent selected-plan records and provider-owned artifact directories."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def record_path(self, key: str) -> Path:
        return self.root / "records" / f"{key}.json"

    def artifact_dir(self, key: str, provider_name: str) -> Path:
        safe_name = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in provider_name
        )
        destination = self.root / "artifacts" / key / safe_name
        destination.mkdir(parents=True, exist_ok=True)
        self._write_marker()
        return destination

    def _write_marker(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / ".custom-dl-optimizer-cache").write_text(
            f"schema={_CACHE_SCHEMA_VERSION}\n",
            encoding="ascii",
        )

    def load(self, key: str) -> PlanCacheRecord | None:
        path = self.record_path(key)
        if not path.is_file():
            return None
        try:
            record = PlanCacheRecord.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None
        if record.schema_version != _CACHE_SCHEMA_VERSION or record.key != key:
            return None
        return record

    def save(
        self,
        *,
        key: str,
        selected_plan: str,
        latency_ms: float,
        report: dict[str, Any],
    ) -> Path:
        destination = self.record_path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._write_marker()
        record = PlanCacheRecord(
            key=key,
            selected_plan=selected_plan,
            latency_ms=latency_ms,
            report=report,
            created_at=datetime.now(timezone.utc).isoformat(),
            package_version=_package_version(),
        )
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(asdict(record), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    def records(self) -> list[PlanCacheRecord]:
        records_dir = self.root / "records"
        if not records_dir.is_dir():
            return []
        records: list[PlanCacheRecord] = []
        for path in sorted(records_dir.glob("*.json")):
            record = self.load(path.stem)
            if record is not None:
                records.append(record)
        return records

    def clear(self) -> int:
        if not self.root.exists():
            return 0
        count = len(self.records())
        for directory_name in ("records", "artifacts"):
            directory = self.root / directory_name
            if directory.is_dir():
                shutil.rmtree(directory)
        marker = self.root / ".custom-dl-optimizer-cache"
        if marker.is_file():
            marker.unlink()
        try:
            self.root.rmdir()
        except OSError:
            pass
        return count
