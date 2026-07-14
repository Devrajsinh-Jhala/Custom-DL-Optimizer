
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

from custom_dl_optimizer.report import GraphSurgeryReport

from .triton_kernels import TritonReLU


def _set_submodule(root: nn.Module, target: str, module: nn.Module) -> None:
    atoms = target.split(".")
    parent = root
    for atom in atoms[:-1]:
        parent = getattr(parent, atom)
    setattr(parent, atoms[-1], module)


def _fold_conv_bn(gm: fx.GraphModule, report: GraphSurgeryReport) -> None:
    modules = dict(gm.named_modules())
    for bn_node in list(gm.graph.nodes):
        if bn_node.op != "call_module" or len(bn_node.args) != 1:
            continue

        bn = modules.get(bn_node.target)
        conv_node = bn_node.args[0]
        if not isinstance(bn, nn.BatchNorm2d) or not isinstance(conv_node, fx.Node):
            continue
        if conv_node.op != "call_module" or len(conv_node.users) != 1:
            continue

        conv = modules.get(conv_node.target)
        if not isinstance(conv, nn.Conv2d):
            continue

        fused = torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        _set_submodule(gm, conv_node.target, fused)
        _set_submodule(gm, bn_node.target, nn.Identity())
        bn_node.replace_all_uses_with(conv_node)
        gm.graph.erase_node(bn_node)
        report.conv_bn_fusions += 1


def _replace_relu_modules(gm: fx.GraphModule, report: GraphSurgeryReport) -> None:
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if node.op != "call_module":
            continue
        module = modules.get(node.target)
        if not isinstance(module, nn.ReLU):
            continue
        if module.inplace:
            report.skipped_inplace_relu += 1
            continue
        _set_submodule(gm, node.target, TritonReLU())
        report.module_relu_replacements += 1


def _replace_functional_relu(gm: fx.GraphModule, report: GraphSurgeryReport) -> None:
    supported_targets = {torch.relu, F.relu}
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in supported_targets:
            continue
        if bool(node.kwargs.get("inplace", False)):
            report.skipped_inplace_relu += 1
            continue

        module_name = f"_custom_relu_{report.functional_relu_replacements}"
        gm.add_submodule(module_name, TritonReLU())
        node.op = "call_module"
        node.target = module_name
        node.args = (node.args[0],)
        node.kwargs = {}
        report.functional_relu_replacements += 1


def optimize_graph(
    model: nn.Module,
    *,
    enable_conv_bn_folding: bool = True,
    enable_triton: bool = True,
) -> tuple[nn.Module, GraphSurgeryReport]:
    """Trace and rewrite supported inference patterns."""

    report = GraphSurgeryReport()
    try:
        gm = fx.symbolic_trace(model).eval()
        report.traced = True
        if enable_conv_bn_folding:
            _fold_conv_bn(gm, report)
        if enable_triton:
            _replace_relu_modules(gm, report)
            _replace_functional_relu(gm, report)
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        return gm, report
    except Exception as exc:
        report.error = repr(exc)[:1000]
        return model, report


def inject_custom_kernels(model: nn.Module) -> nn.Module:
    """Backward-compatible helper that returns the rewritten graph module."""

    optimized, _ = optimize_graph(
        model,
        enable_conv_bn_folding=False,
        enable_triton=True,
    )
    return optimized
