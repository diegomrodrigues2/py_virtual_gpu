"""Utility classes for device memory management."""

from __future__ import annotations


from typing import Any

from .global_memory import GlobalMemory
from .shared_memory import SharedMemory


class DevicePointer:
    """Opaque reference to a location inside a device memory space."""

    def __init__(self, offset: int, memory: GlobalMemory | SharedMemory, *, element_size: int = 4) -> None:
        self.offset = offset
        self.memory = memory
        self.element_size = element_size

    # ------------------------------------------------------------------
    # Pointer arithmetic
    # ------------------------------------------------------------------
    def __add__(self, value: int) -> "DevicePointer":
        return DevicePointer(self.offset + value * self.element_size, self.memory, element_size=self.element_size)

    def __sub__(self, value: int) -> "DevicePointer":
        return DevicePointer(self.offset - value * self.element_size, self.memory, element_size=self.element_size)

    def __iadd__(self, value: int) -> "DevicePointer":
        self.offset += value * self.element_size
        return self

    def __isub__(self, value: int) -> "DevicePointer":
        self.offset -= value * self.element_size
        return self

    # ------------------------------------------------------------------
    # Memory access helpers
    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> bytes:
        """Return the element at ``index`` from the current device's global memory."""

        from .virtualgpu import VirtualGPU

        off = self.offset + index * self.element_size
        gpu = VirtualGPU.get_current()
        return gpu.global_memory.read(off, self.element_size)

    def __setitem__(self, index: int, data: bytes) -> None:
        """Store ``data`` into ``index`` on the current device's global memory."""

        from .virtualgpu import VirtualGPU

        if len(data) != self.element_size:
            raise ValueError("data length must match element_size")
        off = self.offset + index * self.element_size
        gpu = VirtualGPU.get_current()
        gpu.global_memory.write(off, data)

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        mem_name = type(self.memory).__name__
        return (
            f"<DevicePointer mem={mem_name} offset={self.offset} "
            f"elem_size={self.element_size}>"
        )

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, DevicePointer)
            and self.offset == other.offset
            and self.memory is other.memory
        )

    def __int__(self) -> int:  # pragma: no cover - convenience
        return self.offset

    def to_int(self) -> int:
        return self.offset


__all__ = ["DevicePointer"]
