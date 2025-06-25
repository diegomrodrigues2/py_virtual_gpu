"""Service layer for managing GPU instances."""

from .gpu_manager import GPUManager, get_gpu_manager

__all__ = ["GPUManager", "get_gpu_manager"]
