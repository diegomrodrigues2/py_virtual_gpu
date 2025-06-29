"""Utility classes for device memory management."""

from __future__ import annotations


from typing import Any, Type

import numpy as np

from .types import Numeric

from .global_memory import GlobalMemory
from .shared_memory import SharedMemory


class DevicePointer:
    """Opaque reference to a location inside a device memory space."""

    def __init__(
        self,
        offset: int,
        memory: GlobalMemory | SharedMemory,
        *,
        element_size: int = 4,
        dtype: Type[Numeric] | None = None,
    ) -> None:
        self.offset = offset
        self.memory = memory
        self.dtype = dtype
        if dtype is not None:
            element_size = int(np.dtype(dtype.dtype).itemsize)
        self.element_size = element_size

    # ------------------------------------------------------------------
    # Pointer arithmetic
    # ------------------------------------------------------------------
    def __add__(self, value: int) -> "DevicePointer":
        return DevicePointer(
            self.offset + value * self.element_size,
            self.memory,
            element_size=self.element_size,
            dtype=self.dtype,
        )

    def __sub__(self, value: int) -> "DevicePointer":
        return DevicePointer(
            self.offset - value * self.element_size,
            self.memory,
            element_size=self.element_size,
            dtype=self.dtype,
        )

    def __iadd__(self, value: int) -> "DevicePointer":
        self.offset += value * self.element_size
        return self

    def __isub__(self, value: int) -> "DevicePointer":
        self.offset -= value * self.element_size
        return self

    # ------------------------------------------------------------------
    # Memory access helpers
    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> bytes | Numeric:
        """Return the element at ``index`` from ``self.memory``."""

        off = self.offset + index * self.element_size
        raw = self.memory.read(off, self.element_size)
        if self.dtype is None:
            return raw
        value = np.frombuffer(raw, dtype=self.dtype.dtype)[0]
        return self.dtype(value)

    def __setitem__(self, index: int, data: bytes | Numeric) -> None:
        """Store ``data`` at ``index`` inside ``self.memory``."""

        if isinstance(data, (bytes, bytearray)):
            if len(data) != self.element_size:
                raise ValueError("data length must match element_size")
            raw = bytes(data)
        else:
            if self.dtype is None or not isinstance(data, self.dtype):
                raise TypeError("data must be bytes or an instance of the pointer dtype")
            raw = np.array(data.value, dtype=self.dtype.dtype).tobytes()
        off = self.offset + index * self.element_size
        self.memory.write(off, raw)

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        mem_name = type(self.memory).__name__
        dtype_name = None
        if self.dtype is not None:
            dtype_name = getattr(self.dtype, "__name__", str(self.dtype))
        return (
            f"<DevicePointer mem={mem_name} offset={self.offset} "
            f"elem_size={self.element_size} dtype={dtype_name}>"
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
