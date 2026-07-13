import torch
import torch.nn as nn
import torch.nn.functional as F

import custom_dl_optimizer
from custom_dl_optimizer import AutoOptimizer, OptimizationConfig
from custom_dl_optimizer.core.graph_surgeon import optimize_graph
from custom_dl_optimizer.core.triton_kernels import TritonReLU

FAST_CONFIG = OptimizationConfig(
    enable_profiling=False,
    enable_compile=False,
    selection_warmup=0,
    selection_iterations=1,
    selection_repeats=1,
    min_speedup=1.0,
    copy_model=False,
)


def test_version_exported():
    assert custom_dl_optimizer.__version__ == "1.1.0"


def test_cpu_optimizer_runs_and_exposes_report():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    inputs = torch.randn(2, 4)
    optimizer = AutoOptimizer(model, device="cpu", config=FAST_CONFIG)

    optimized = optimizer.optimize(inputs)

    with torch.inference_mode():
        baseline = model(inputs)
        actual = optimized(inputs)
    assert torch.allclose(baseline, actual)
    assert optimizer.last_report is not None
    assert optimizer.last_report.selected_plan in {"native", "fx"}
    assert any(candidate.selected for candidate in optimizer.last_report.candidates)


def test_multi_input_model_is_supported():
    class AddModel(nn.Module):
        def forward(self, left, right, scale=1.0):
            return (left + right) * scale

    model = AddModel().eval()
    left = torch.randn(2, 4)
    right = torch.randn(2, 4)
    optimized = AutoOptimizer(model, device="cpu", config=FAST_CONFIG).optimize(
        left,
        right,
        scale=0.5,
    )
    assert torch.allclose(optimized(left, right, scale=0.5), model(left, right, scale=0.5))


def test_nested_relu_is_replaced_in_fx_graph():
    class Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

        def forward(self, x):
            return self.block(x)

    optimized, report = optimize_graph(Nested().eval(), enable_conv_bn_folding=False)
    assert report.module_relu_replacements == 1
    assert any(isinstance(module, TritonReLU) for module in optimized.modules())


def test_functional_relu_is_replaced():
    class FunctionalRelu(nn.Module):
        def forward(self, x):
            return F.relu(x)

    optimized, report = optimize_graph(FunctionalRelu().eval(), enable_conv_bn_folding=False)
    assert report.functional_relu_replacements == 1
    assert any(isinstance(module, TritonReLU) for module in optimized.modules())


def test_inplace_relu_is_not_replaced():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(inplace=True)).eval()
    optimized, report = optimize_graph(model, enable_conv_bn_folding=False)
    assert report.skipped_inplace_relu == 1
    assert not any(isinstance(module, TritonReLU) for module in optimized.modules())


def test_conv_batchnorm_is_folded_with_parity():
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(),
    ).eval()
    inputs = torch.randn(2, 3, 8, 8)
    with torch.inference_mode():
        expected = model(inputs)
    optimized, report = optimize_graph(model, enable_triton=False)
    with torch.inference_mode():
        actual = optimized(inputs)
    assert report.conv_bn_fusions == 1
    assert torch.allclose(expected, actual, rtol=1e-5, atol=1e-5)


def test_fx_trace_failure_falls_back_to_native():
    class DataDependentModel(nn.Module):
        def forward(self, x):
            if x.sum() > 0:
                return x * 2
            return x - 2

    inputs = torch.ones(2, 4)
    optimizer = AutoOptimizer(
        DataDependentModel().eval(),
        device="cpu",
        config=FAST_CONFIG,
    )
    optimized = optimizer.optimize(inputs)
    assert optimizer.last_report is not None
    assert not optimizer.last_report.graph.traced
    assert optimizer.last_report.selected_plan == "native"
    assert torch.equal(optimized(inputs), inputs * 2)
