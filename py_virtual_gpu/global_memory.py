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

    def __init__(
        self,
        size: int,
        latency_cycles: int = 200,
        bandwidth_bytes_per_cycle: int = 32,
    ) -> None:
        """Create a block of ``size`` bytes backed by shared memory."""
        self.size: int = size
        self.buffer = Array(c_byte, size, lock=False)
        self.lock = Lock()
        self.allocations: Dict[int, int] = {}
        self._free_list: List[Tuple[int, int]] = [(0, size)]
        self.latency_cycles = latency_cycles
        self.bandwidth_bpc = bandwidth_bytes_per_cycle

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
    def read(self, ptr: int | "DevicePointer", size: int) -> bytes:
        """Return ``size`` bytes starting at ``ptr``.

        Parameters
        ----------
        ptr:
            Offset inside the buffer or a :class:`DevicePointer` referring
            to this memory.
        size:
            Number of bytes to read.
        """

        off = ptr.offset if hasattr(ptr, "offset") else ptr
        with self.lock:
            return bytes(self.buffer[off : off + size])

    def write(self, ptr: int | "DevicePointer", data: bytes) -> None:
        """Write ``data`` starting at ``ptr``.

        ``ptr`` can be an integer offset or a :class:`DevicePointer` owned by
        this memory.
        """

        off = ptr.offset if hasattr(ptr, "offset") else ptr
        with self.lock:
            self.buffer[off : off + len(data)] = data

    # ------------------------------------------------------------------
    # Atomic operations
    # ------------------------------------------------------------------
    def atomic_add(self, ptr: int, value: int, num_bytes: int = 4) -> int:
        """Atomically add ``value`` to integer at ``ptr``."""

        if ptr < 0 or ptr + num_bytes > self.size:
            raise IndexError("GlobalMemory atomic out of bounds")
        with self.lock:
            raw = memoryview(self.buffer)[ptr : ptr + num_bytes].tobytes()
            old = int.from_bytes(raw, byteorder="little", signed=True)
            new = old + value
            self.buffer[ptr : ptr + num_bytes] = new.to_bytes(
                num_bytes, byteorder="little", signed=True
            )
            return old

    def atomic_sub(self, ptr: int, value: int, num_bytes: int = 4) -> int:
        """Atomically subtract ``value`` from integer at ``ptr``."""

        if ptr < 0 or ptr + num_bytes > self.size:
            raise IndexError("GlobalMemory atomic out of bounds")
        with self.lock:
            raw = memoryview(self.buffer)[ptr : ptr + num_bytes].tobytes()
            old = int.from_bytes(raw, byteorder="little", signed=True)
            new = old - value
            self.buffer[ptr : ptr + num_bytes] = new.to_bytes(
                num_bytes, byteorder="little", signed=True
            )
            return old

    def atomic_cas(self, ptr: int, expected: int, new: int, num_bytes: int = 4) -> bool:
        """Compare-and-swap value at ``ptr``."""

        if ptr < 0 or ptr + num_bytes > self.size:
            raise IndexError("GlobalMemory atomic out of bounds")
        with self.lock:
            raw = memoryview(self.buffer)[ptr : ptr + num_bytes].tobytes()
            current = int.from_bytes(raw, byteorder="little", signed=True)
            if current == expected:
                self.buffer[ptr : ptr + num_bytes] = new.to_bytes(
                    num_bytes, byteorder="little", signed=True
                )
                return True
            return False

    def atomic_max(self, ptr: int, value: int, num_bytes: int = 4) -> int:
        """Atomically store ``max(current, value)`` at ``ptr``."""

        if ptr < 0 or ptr + num_bytes > self.size:
            raise IndexError("GlobalMemory atomic out of bounds")
        with self.lock:
            raw = memoryview(self.buffer)[ptr : ptr + num_bytes].tobytes()
            old = int.from_bytes(raw, byteorder="little", signed=True)
            m = max(old, value)
            self.buffer[ptr : ptr + num_bytes] = m.to_bytes(
                num_bytes, byteorder="little", signed=True
            )
            return old

    def atomic_min(self, ptr: int, value: int, num_bytes: int = 4) -> int:
        """Atomically store ``min(current, value)`` at ``ptr``."""

        if ptr < 0 or ptr + num_bytes > self.size:
            raise IndexError("GlobalMemory atomic out of bounds")
        with self.lock:
            raw = memoryview(self.buffer)[ptr : ptr + num_bytes].tobytes()
            old = int.from_bytes(raw, byteorder="little", signed=True)
            m = min(old, value)
            self.buffer[ptr : ptr + num_bytes] = m.to_bytes(
                num_bytes, byteorder="little", signed=True
            )
            return old

    def atomic_exchange(self, ptr: int, value: int, num_bytes: int = 4) -> int:
        """Atomically replace value at ``ptr`` and return old value."""

        if ptr < 0 or ptr + num_bytes > self.size:
            raise IndexError("GlobalMemory atomic out of bounds")
        with self.lock:
            raw = memoryview(self.buffer)[ptr : ptr + num_bytes].tobytes()
            old = int.from_bytes(raw, byteorder="little", signed=True)
            self.buffer[ptr : ptr + num_bytes] = value.to_bytes(
                num_bytes, byteorder="little", signed=True
            )
            return old

    def memcpy(
        self, dest_ptr: int, src: int | bytes, size: int, direction: str
    ) -> bytes | None:
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
