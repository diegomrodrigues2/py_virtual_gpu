"""Global memory simulation using ``multiprocessing.Array``.

This module defines :class:`GlobalMemory`, a simple manager for a chunk of
bytes that is shared among processes. It is deliberately minimal and will
be expanded in later issues.
"""

from __future__ import annotations

from multiprocessing import Array, Lock
from ctypes import c_byte
from typing import Dict, List, Tuple


class GlobalMemory:
    """Simulated global memory accessible to all thread blocks."""

    def __init__(self, size: int) -> None:
        """Create a block of ``size`` bytes backed by shared memory."""
        self.size: int = size
        self.buffer = Array(c_byte, size, lock=False)
        self.lock = Lock()
        self.allocations: Dict[int, int] = {}
        self._free_list: List[Tuple[int, int]] = [(0, size)]

    # ------------------------------------------------------------------
    # Allocation helpers
    # ------------------------------------------------------------------
    def malloc(self, size: int) -> int:
        """Allocate ``size`` bytes and return the offset inside ``buffer``."""

        for idx, (offset, block_size) in enumerate(self._free_list):
            if block_size >= size:
                self.allocations[offset] = size
                del self._free_list[idx]
                if block_size > size:
                    self._free_list.insert(idx, (offset + size, block_size - size))
                return offset
        raise MemoryError("Out of global memory")

    def free(self, ptr: int) -> None:
        """Release the block starting at ``ptr``.

        Raises
        ------
        ValueError
            If ``ptr`` was not previously allocated or has been freed.
        """

        size = self.allocations.pop(ptr, None)
        if size is None:
            raise ValueError("Invalid or double free")

        self._free_list.append((ptr, size))
        self._free_list = self._coalesce(self._free_list)

    @staticmethod
    def _coalesce(regions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge adjacent free regions."""

        regs = sorted(regions)
        merged: List[Tuple[int, int]] = []
        for off, sz in regs:
            if merged and merged[-1][0] + merged[-1][1] == off:
                prev_off, prev_sz = merged.pop()
                merged.append((prev_off, prev_sz + sz))
            else:
                merged.append((off, sz))
        return merged

    # ------------------------------------------------------------------
    # Raw memory access
    # ------------------------------------------------------------------
    def read(self, ptr: int, size: int) -> bytes:
        """Return ``size`` bytes from ``ptr``."""
        with self.lock:
            return bytes(self.buffer[ptr : ptr + size])

    def write(self, ptr: int, data: bytes) -> None:
        """Write ``data`` starting at ``ptr``."""
        with self.lock:
            self.buffer[ptr : ptr + len(data)] = data

    def memcpy(self, dest_ptr: int, src: int | bytes, size: int, direction: str) -> bytes | None:
        """Copy memory between host and device.

        Parameters
        ----------
        dest_ptr:
            Destination offset inside the device memory when copying
            Host->Device or Device->Device.
        src:
            Source bytes when copying Host->Device, or source offset when
            Device->Device. Ignored for Device->Host.
        size:
            Number of bytes to transfer.
        direction:
            One of ``'HostToDevice'``, ``'DeviceToHost'`` or ``'DeviceToDevice'``.
        """
        if direction == "HostToDevice":
            if not isinstance(src, (bytes, bytearray)):
                raise TypeError("src must be bytes for HostToDevice copies")
            self.write(dest_ptr, bytes(src[:size]))
            return None
        if direction == "DeviceToHost":
            return self.read(dest_ptr, size)
        if direction == "DeviceToDevice":
            if not isinstance(src, int):
                raise TypeError("src must be an int offset for DeviceToDevice")
            temp = self.read(src, size)
            self.write(dest_ptr, temp)
            return None
        raise ValueError(f"Unknown direction: {direction}")

__all__ = ["GlobalMemory"]

