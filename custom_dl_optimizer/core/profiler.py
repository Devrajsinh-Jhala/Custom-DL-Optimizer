import torch
from torch.profiler import profile, record_function, ProfilerActivity

def analyze_bottlenecks(model, dummy_input):
    print("[Profiler] Running warm-up...")
    for _ in range(3): model(dummy_input)
    
    print("[Profiler] Tracing execution...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(dummy_input)
            
    events = prof.key_averages()
    
    # FIX: Safely extract CUDA time regardless of PyTorch version
    def get_time(e):
        return getattr(e, "self_cuda_time_total", getattr(e, "self_cpu_time_total", 0))
        
    sorted_events = sorted(events, key=get_time, reverse=True)
    
    slowest_op = sorted_events[0]
    if "model_inference" in slowest_op.key and len(sorted_events) > 1:
        slowest_op = sorted_events[1]
        
    print(f"⚠️ [Profiler] Bottleneck Found: '{slowest_op.key}' is heavily taxing the hardware.")
    return slowest_op.key
