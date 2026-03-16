import torch.fx as fx
import torch.nn as nn
from .triton_kernels import TritonReLU

def inject_custom_kernels(model: nn.Module) -> nn.Module:
    print("[Graph Surgeon] Tracing model into FX AST Graph...")
    gm = fx.symbolic_trace(model)
    
    replacements = 0
    for node in gm.graph.nodes:
        if node.op == 'call_module':
            target_mod = dict(gm.named_modules())[node.target]
            if isinstance(target_mod, nn.ReLU):
                setattr(gm, node.target, TritonReLU())
                replacements += 1
                
    gm.graph.lint() 
    gm.recompile()  
    print(f"✅ [Graph Surgeon] Injected custom Triton kernels in {replacements} locations.")
    return gm
