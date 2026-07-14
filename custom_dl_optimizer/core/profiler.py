from collections.abc import Iterable
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from custom_dl_optimizer.report import OperatorProfile

_IGNORED_EVENT_SUBSTRINGS = (
    "model_inference",
    "cudaDeviceSynchronize",
    "Activity Buffer",
    "ProfilerStep",
)


def _event_time_us(event: Any) -> float:
    for attribute in (
        "self_device_time_total",
        "self_cuda_time_total",
        "self_cpu_time_total",
    ):
        value = float(getattr(event, attribute, 0.0) or 0.0)
        if value > 0:
            return value
    return 0.0


def analyze_bottlenecks(
    model: torch.nn.Module,
    example_args: tuple[Any, ...],
    example_kwargs: dict[str, Any] | None = None,
    warmup_steps: int = 3,
    top_k: int = 10,
) -> list[OperatorProfile]:
    """Return the slowest useful operators from one representative inference."""

    kwargs = example_kwargs or {}
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup_steps):
            model(*example_args, **kwargs)

    has_cuda_input = any(
        isinstance(value, torch.Tensor) and value.is_cuda
        for value in _walk_values((example_args, kwargs))
    )
    if has_cuda_input:
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if has_cuda_input:
        activities.append(ProfilerActivity.CUDA)

    with torch.inference_mode():
        with profile(activities=activities, record_shapes=True) as profiler:
            with record_function("model_inference"):
                model(*example_args, **kwargs)

    useful_events = [
        event
        for event in profiler.key_averages()
        if not any(token in event.key for token in _IGNORED_EVENT_SUBSTRINGS)
    ]
    useful_events.sort(key=_event_time_us, reverse=True)
    return [
        OperatorProfile(
            name=event.key,
            self_time_us=_event_time_us(event),
            calls=int(event.count),
        )
        for event in useful_events[:top_k]
    ]


def _walk_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value
