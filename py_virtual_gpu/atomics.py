from __future__ import annotations

from .virtualgpu import VirtualGPU
from .memory import DevicePointer


def atomicAdd(ptr: DevicePointer, value: int, num_bytes: int = 4) -> int:
    """Atomically add ``value`` to integer at ``ptr`` in global memory."""

    if not isinstance(ptr, DevicePointer):
        raise TypeError("ptr must be a DevicePointer")
    gpu = VirtualGPU.get_current()
    return gpu.global_mem.atomic_add(ptr.offset, value, num_bytes)


def atomicAdd_float32(ptr: DevicePointer, value: float) -> float:
    """Atomically add ``value`` to a 32-bit float at ``ptr``."""

    if not isinstance(ptr, DevicePointer):
        raise TypeError("ptr must be a DevicePointer")
    gpu = VirtualGPU.get_current()
    return gpu.global_mem.atomic_add_float32(ptr.offset, value)


def atomicAdd_float64(ptr: DevicePointer, value: float) -> float:
    """Atomically add ``value`` to a 64-bit float at ``ptr``."""

    if not isinstance(ptr, DevicePointer):
        raise TypeError("ptr must be a DevicePointer")
    gpu = VirtualGPU.get_current()
    return gpu.global_mem.atomic_add_float64(ptr.offset, value)


def atomicSub(ptr: DevicePointer, value: int, num_bytes: int = 4) -> int:
    """Atomically subtract ``value`` from integer at ``ptr``."""

    if not isinstance(ptr, DevicePointer):
        raise TypeError("ptr must be a DevicePointer")
    gpu = VirtualGPU.get_current()
    return gpu.global_mem.atomic_sub(ptr.offset, value, num_bytes)


def atomicCAS(ptr: DevicePointer, expected: int, new: int, num_bytes: int = 4) -> bool:
    """Compare-and-swap integer at ``ptr``."""

    if not isinstance(ptr, DevicePointer):
        raise TypeError("ptr must be a DevicePointer")
    gpu = VirtualGPU.get_current()
    return gpu.global_mem.atomic_cas(ptr.offset, expected, new, num_bytes)


def atomicMax(ptr: DevicePointer, value: int, num_bytes: int = 4) -> int:
    """Atomically store ``max(current, value)`` at ``ptr``."""

    if not isinstance(ptr, DevicePointer):
        raise TypeError("ptr must be a DevicePointer")
    gpu = VirtualGPU.get_current()
    return gpu.global_mem.atomic_max(ptr.offset, value, num_bytes)


def atomicMin(ptr: DevicePointer, value: int, num_bytes: int = 4) -> int:
    """Atomically store ``min(current, value)`` at ``ptr``."""

    if not isinstance(ptr, DevicePointer):
        raise TypeError("ptr must be a DevicePointer")
    gpu = VirtualGPU.get_current()
    return gpu.global_mem.atomic_min(ptr.offset, value, num_bytes)


def atomicExchange(ptr: DevicePointer, value: int, num_bytes: int = 4) -> int:
    """Atomically exchange value at ``ptr`` and return old value."""

    if not isinstance(ptr, DevicePointer):
        raise TypeError("ptr must be a DevicePointer")
    gpu = VirtualGPU.get_current()
    return gpu.global_mem.atomic_exchange(ptr.offset, value, num_bytes)


__all__ = [
    "atomicAdd",
    "atomicAdd_float32",
    "atomicAdd_float64",
    "atomicSub",
    "atomicCAS",
    "atomicMax",
    "atomicMin",
    "atomicExchange",
]
