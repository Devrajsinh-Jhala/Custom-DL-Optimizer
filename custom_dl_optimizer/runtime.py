from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from importlib.util import find_spec
from typing import Any

import torch


@dataclass(frozen=True)
class RuntimeCapabilities:
    """Serializable description of the runtime used for plan selection."""

    python_version: str
    torch_version: str
    device: str
    device_type: str
    device_name: str
    cuda_version: str | None
    cudnn_version: str | None
    compute_capability: str | None
    torch_compile_available: bool
    triton_available: bool
    amp_available: bool
    channels_last_available: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_runtime(
    device: str | torch.device | None = None,
) -> RuntimeCapabilities:
    """Inspect optimization capabilities without mutating global PyTorch state."""

    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if resolved.type == "cuda" and not torch.cuda.is_available():
        resolved = torch.device("cpu")

    device_name = platform.processor() or "CPU"
    compute_capability: str | None = None
    if resolved.type == "cuda":
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(index)
        major, minor = torch.cuda.get_device_capability(index)
        compute_capability = f"{major}.{minor}"

    cudnn_version = torch.backends.cudnn.version()
    return RuntimeCapabilities(
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        device=str(resolved),
        device_type=resolved.type,
        device_name=device_name,
        cuda_version=torch.version.cuda,
        cudnn_version=str(cudnn_version) if cudnn_version is not None else None,
        compute_capability=compute_capability,
        torch_compile_available=hasattr(torch, "compile"),
        triton_available=find_spec("triton") is not None and resolved.type == "cuda",
        amp_available=resolved.type == "cuda",
        channels_last_available=resolved.type == "cuda",
    )
