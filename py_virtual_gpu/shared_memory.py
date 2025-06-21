"""Shared memory simulation used by :class:`ThreadBlock`."""

from __future__ import annotations

from multiprocessing import Array, Lock
from ctypes import c_byte


class SharedMemory:
    """Simple byte-addressable shared memory buffer."""

    def __init__(self, size: int) -> None:
        """Create a shared buffer of ``size`` bytes."""

        self.size: int = size
        self.buffer = Array(c_byte, size, lock=False)
        self.lock = Lock()

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------
    def read(self, offset: int, size: int) -> bytes:
        """Return ``size`` bytes starting from ``offset``."""

        if offset < 0 or offset + size > self.size:
            raise IndexError("SharedMemory read out of bounds")
        view = memoryview(self.buffer)
        return view[offset : offset + size].tobytes()

    def write(self, offset: int, data: bytes) -> None:
        """Write ``data`` starting at ``offset``."""

        end = offset + len(data)
        if offset < 0 or end > self.size:
            raise IndexError("SharedMemory write out of bounds")
        self.buffer[offset:end] = data

    # ------------------------------------------------------------------
    # Atomic operations
    # ------------------------------------------------------------------
    def atomic_add(self, offset: int, value: int, num_bytes: int = 4) -> int:
        """Atomically add ``value`` to an integer stored at ``offset``."""

        if offset < 0 or offset + num_bytes > self.size:
            raise IndexError("SharedMemory atomic out of bounds")
        with self.lock:
            raw = memoryview(self.buffer)[offset : offset + num_bytes].tobytes()
            old = int.from_bytes(raw, byteorder="little", signed=True)
            new = old + value
            self.buffer[offset : offset + num_bytes] = new.to_bytes(
                num_bytes, byteorder="little", signed=True
            )
            return old


__all__ = ["SharedMemory"]
