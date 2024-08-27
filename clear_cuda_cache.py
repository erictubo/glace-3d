# clear_cuda_cache.py
import torch

torch.cuda.empty_cache()
print("CUDA cache cleared.")