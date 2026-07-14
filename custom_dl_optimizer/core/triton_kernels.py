import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - depends on platform-specific installs
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.maximum(x, 0.0)
        tl.store(y_ptr + offsets, y, mask=mask)


def _has_dense_storage(x):
    return x.is_contiguous() or (
        x.dim() == 4 and x.is_contiguous(memory_format=torch.channels_last)
    )


class TritonReLU(nn.Module):
    """ReLU module backed by an autotuned Triton kernel when safe."""

    def forward(self, x):
        if not TRITON_AVAILABLE or not x.is_cuda or not _has_dense_storage(x):
            return F.relu(x)

        y = torch.empty_like(x, memory_format=torch.preserve_format)
        n_elements = x.numel()

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        relu_kernel[grid](x, y, n_elements)
        return y
