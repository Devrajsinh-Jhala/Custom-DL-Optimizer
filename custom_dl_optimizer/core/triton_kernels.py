import torch
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0) 
    tl.store(y_ptr + offsets, y, mask=mask)

class TritonReLU(nn.Module):
    def forward(self, x):
        if not x.is_cuda: return torch.relu(x) # Safety check
        x = x.contiguous() 
        y = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        relu_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)
        return y
