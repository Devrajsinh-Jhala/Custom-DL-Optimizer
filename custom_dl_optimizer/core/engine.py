import torch
import torch.nn as nn
from .profiler import analyze_bottlenecks
from .graph_surgeon import inject_custom_kernels

class AutoOptimizer:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.device = device

    def optimize(self, dummy_input):
        print("\n" + "="*50)
        print("🚀 STARTING DL-OPTIMIZER PIPELINE")
        print("="*50)
        
        analyze_bottlenecks(self.model, dummy_input)
        optimized_model = inject_custom_kernels(self.model)
        
        print("[Memory Ops] Converting memory layout to Channels-Last (NHWC)...")
        optimized_model = optimized_model.to(memory_format=torch.channels_last)
        
        class AMPWrapper(nn.Module):
            def __init__(self, core_model):
                super().__init__()
                self.core_model = core_model
            def forward(self, x):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    return self.core_model(x)
                    
        print("[Precision] Wrapping model in FP16 Auto-Mixed Precision...")
        optimized_model = AMPWrapper(optimized_model)
        
        print("✅ PIPELINE COMPLETE. Model is ready for inference.\n")
        return optimized_model
