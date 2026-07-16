from __future__ import annotations

import io
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any

import torch
import torch.nn as nn

from .providers import BuiltPlan, ProviderAvailability, ProviderContext


@dataclass(frozen=True)
class TorchTensorRTProvider:
    """Optional Torch-TensorRT candidate with backend-owned engine caching."""

    name: str = "torch_tensorrt"
    compile_options: dict[str, Any] = field(default_factory=dict)
    enable_engine_cache: bool = True

    def probe(self, context: ProviderContext) -> ProviderAvailability:
        if context.device.type != "cuda":
            return ProviderAvailability.unsupported("Torch-TensorRT requires CUDA")
        if find_spec("torch_tensorrt") is None:
            return ProviderAvailability.unsupported("torch_tensorrt is not installed")
        return ProviderAvailability.supported()

    def cache_identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "compile_options": self.compile_options,
            "enable_engine_cache": self.enable_engine_cache,
        }

    def build(self, model: nn.Module, context: ProviderContext) -> BuiltPlan:
        import torch_tensorrt  # noqa: F401

        options = dict(self.compile_options)
        if self.enable_engine_cache:
            options.setdefault("cache_built_engines", True)
            options.setdefault("reuse_cached_engines", True)
            if context.artifact_dir is not None:
                context.artifact_dir.mkdir(parents=True, exist_ok=True)
                options.setdefault(
                    "timing_cache_path",
                    str(context.artifact_dir / "timing-cache.bin"),
                )
        runner = torch.compile(
            model,
            backend="torch_tensorrt",
            dynamic=context.policy.dynamic_shapes,
            options=options,
        )
        artifacts = ()
        if context.artifact_dir is not None:
            artifacts = (str(context.artifact_dir),)
        return BuiltPlan(
            runner=runner,
            artifacts=artifacts,
            metadata={"backend": "torch_tensorrt", "options": options},
        )


def _flatten_output(value: Any, tensors: list[torch.Tensor]) -> Any:
    if isinstance(value, torch.Tensor):
        index = len(tensors)
        tensors.append(value)
        return ("tensor", index)
    if isinstance(value, tuple):
        return ("tuple", [_flatten_output(item, tensors) for item in value])
    if isinstance(value, list):
        return ("list", [_flatten_output(item, tensors) for item in value])
    if isinstance(value, dict):
        return (
            "dict",
            [(key, _flatten_output(item, tensors)) for key, item in value.items()],
        )
    raise TypeError("ONNX Runtime provider supports tensor containers as outputs")


def _rebuild_output(specification: Any, tensors: list[torch.Tensor]) -> Any:
    kind, payload = specification
    if kind == "tensor":
        return tensors[payload]
    if kind == "tuple":
        return tuple(_rebuild_output(item, tensors) for item in payload)
    if kind == "list":
        return [_rebuild_output(item, tensors) for item in payload]
    if kind == "dict":
        return {key: _rebuild_output(item, tensors) for key, item in payload}
    raise RuntimeError(f"Unknown ONNX output specification: {kind!r}")


def _numpy_dtype(dtype: torch.dtype) -> Any:
    import numpy as np

    mapping = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }
    if dtype not in mapping:
        raise TypeError(f"ONNX Runtime I/O binding does not support {dtype}")
    return mapping[dtype]


class _ONNXRuntimeModule(nn.Module):
    def __init__(
        self,
        session: Any,
        *,
        device: torch.device,
        input_names: tuple[str, ...],
        output_names: tuple[str, ...],
        output_specification: Any,
    ) -> None:
        super().__init__()
        self.session = session
        self.device = device
        self.input_names = input_names
        self.output_names = output_names
        self.output_specification = output_specification

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise TypeError("ONNX Runtime provider currently accepts positional inputs only")
        if len(args) != len(self.input_names) or not all(
            isinstance(value, torch.Tensor) for value in args
        ):
            raise TypeError("ONNX Runtime provider requires flat positional tensor inputs")
        tensors = [value.detach().contiguous() for value in args]
        if self.device.type == "cuda":
            binding = self.session.io_binding()
            device_id = self.device.index or 0
            for name, tensor in zip(self.input_names, tensors, strict=True):
                binding.bind_input(
                    name=name,
                    device_type="cuda",
                    device_id=device_id,
                    element_type=_numpy_dtype(tensor.dtype),
                    shape=tuple(tensor.shape),
                    buffer_ptr=tensor.data_ptr(),
                )
            for name in self.output_names:
                binding.bind_output(name, "cuda", device_id)
            self.session.run_with_iobinding(binding)
            ort_outputs = binding.get_outputs()
            outputs: list[torch.Tensor] = []
            for value in ort_outputs:
                try:
                    outputs.append(torch.utils.dlpack.from_dlpack(value))
                except (TypeError, RuntimeError):
                    outputs.append(torch.from_numpy(value.numpy()).to(self.device))
        else:
            feeds = {
                name: tensor.cpu().numpy()
                for name, tensor in zip(self.input_names, tensors, strict=True)
            }
            outputs = [
                torch.from_numpy(value)
                for value in self.session.run(list(self.output_names), feeds)
            ]
        return _rebuild_output(self.output_specification, outputs)


@dataclass(frozen=True)
class ONNXRuntimeProvider:
    """Optional ONNX Runtime candidate with CUDA I/O binding when available."""

    name: str = "onnxruntime"
    execution_providers: tuple[str, ...] | None = None
    provider_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    opset_version: int = 17

    def _providers(self, device: torch.device) -> tuple[str, ...]:
        if self.execution_providers is not None:
            return self.execution_providers
        if device.type == "cuda":
            return ("CUDAExecutionProvider", "CPUExecutionProvider")
        return ("CPUExecutionProvider",)

    def probe(self, context: ProviderContext) -> ProviderAvailability:
        if find_spec("onnx") is None or find_spec("onnxruntime") is None:
            return ProviderAvailability.unsupported("onnx and onnxruntime are required")
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        requested = self._providers(context.device)
        if context.device.type == "cuda":
            available_on_target = any(
                provider in available
                for provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider")
                if provider in requested
            )
            return (
                ProviderAvailability.supported()
                if available_on_target
                else ProviderAvailability.unsupported(
                    "Requested ONNX Runtime GPU execution provider is unavailable"
                )
            )
        if "CPUExecutionProvider" not in available:
            return ProviderAvailability.unsupported(
                "ONNX Runtime CPUExecutionProvider is unavailable"
            )
        return ProviderAvailability.supported()

    def cache_identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "execution_providers": self.execution_providers,
            "provider_options": self.provider_options,
            "opset_version": self.opset_version,
        }

    def build(self, model: nn.Module, context: ProviderContext) -> BuiltPlan:
        if context.example_kwargs:
            raise TypeError("ONNX Runtime provider currently accepts positional inputs only")
        if not all(isinstance(value, torch.Tensor) for value in context.example_args):
            raise TypeError("ONNX Runtime provider requires flat positional tensor inputs")

        import onnxruntime as ort

        input_names = tuple(
            f"input_{index}" for index in range(len(context.example_args))
        )
        with torch.inference_mode():
            reference = model(*context.example_args)
        flat_outputs: list[torch.Tensor] = []
        output_specification = _flatten_output(reference, flat_outputs)
        output_names = tuple(f"output_{index}" for index in range(len(flat_outputs)))
        dynamic_axes = None
        if context.policy.dynamic_shapes:
            dynamic_axes = {
                name: {axis: f"{name}_dim_{axis}" for axis in range(value.dim())}
                for name, value in zip(
                    input_names,
                    context.example_args,
                    strict=True,
                )
            }

        destination: str | io.BytesIO
        if context.artifact_dir is not None:
            context.artifact_dir.mkdir(parents=True, exist_ok=True)
            destination = str(context.artifact_dir / "model.onnx")
        else:
            destination = io.BytesIO()
        torch.onnx.export(
            model,
            context.example_args,
            destination,
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes=dynamic_axes,
            opset_version=self.opset_version,
            do_constant_folding=True,
        )
        model_source = destination.getvalue() if isinstance(destination, io.BytesIO) else destination
        requested = self._providers(context.device)
        available = set(ort.get_available_providers())
        providers = [provider for provider in requested if provider in available]
        options = [self.provider_options.get(provider, {}) for provider in providers]
        session = ort.InferenceSession(
            model_source,
            providers=providers,
            provider_options=options,
        )
        runner = _ONNXRuntimeModule(
            session,
            device=context.device,
            input_names=input_names,
            output_names=output_names,
            output_specification=output_specification,
        )
        artifacts = (destination,) if isinstance(destination, str) else ()
        return BuiltPlan(
            runner=runner,
            artifacts=artifacts,
            metadata={"backend": "onnxruntime", "execution_providers": providers},
        )


@dataclass(frozen=True)
class TorchAOQuantizationProvider:
    """Optional maintained TorchAO quantization candidate."""

    scheme: str = "int8_weight_only"
    name: str = ""
    group_size: int = 128
    compile_model: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", f"torchao_{self.scheme}")

    def probe(self, context: ProviderContext) -> ProviderAvailability:
        if find_spec("torchao") is None:
            return ProviderAvailability.unsupported("torchao is not installed")
        return ProviderAvailability.supported()

    def cache_identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scheme": self.scheme,
            "group_size": self.group_size,
            "compile_model": self.compile_model,
        }

    def build(self, model: nn.Module, context: ProviderContext) -> BuiltPlan:
        import torchao.quantization as quantization

        factories = {
            "int8_weight_only": lambda: quantization.Int8WeightOnlyConfig(),
            "int8_dynamic": lambda: quantization.Int8DynamicActivationInt8WeightConfig(),
            "int4_weight_only": lambda: quantization.Int4WeightOnlyConfig(
                group_size=self.group_size
            ),
            "float8_weight_only": lambda: quantization.Float8WeightOnlyConfig(),
            "float8_dynamic": lambda: quantization.Float8DynamicActivationFloat8WeightConfig(),
        }
        if self.scheme not in factories:
            supported = ", ".join(sorted(factories))
            raise ValueError(f"Unknown TorchAO scheme {self.scheme!r}; choose from {supported}")
        quantization.quantize_(model, factories[self.scheme]())
        if self.compile_model:
            model = torch.compile(
                model,
                dynamic=context.policy.dynamic_shapes,
                mode=context.policy.compile_mode,
            )
        return BuiltPlan(
            runner=model,
            metadata={"backend": "torchao", "scheme": self.scheme},
        )
