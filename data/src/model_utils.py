import torch
from contextlib import contextmanager
import gc

@contextmanager
def memory_management_fn():
    """Context manager for memory management"""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def get_detailed_memory_usage():
    """Get detailed GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i} ({torch.cuda.get_device_name(i)}):")
            print(f"Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
            print(f"Max allocated: {torch.cuda.max_memory_allocated(i) / 1024**2:.2f} MB")
            print(f"Max cached: {torch.cuda.max_memory_reserved(i) / 1024**2:.2f} MB")


