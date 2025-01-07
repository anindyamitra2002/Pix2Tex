import torch
import random
import numpy as np
import os
from typing import Optional

class Utils:
    @staticmethod
    def set_seed(seed: int) -> None:
        """Set random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def setup_logging_dir(output_dir: str) -> str:
        """Setup logging directory"""
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def get_gpu_memory_usage() -> Optional[str]:
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_cached = torch.cuda.memory_reserved() / 1024**2
            return f"GPU Memory: Allocated={memory_allocated:.2f}MB, Cached={memory_cached:.2f}MB"
        return None

    @staticmethod
    def optimize_memory_usage():
        """Optimize memory usage for training"""
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()