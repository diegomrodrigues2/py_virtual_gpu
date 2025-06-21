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

    def atomic_sub(self, offset: int, value: int, num_bytes: int = 4) -> int:
        """Atomically subtract ``value`` from an integer at ``offset``."""

        if offset < 0 or offset + num_bytes > self.size:
            raise IndexError("SharedMemory atomic out of bounds")
        with self.lock:
            raw = memoryview(self.buffer)[offset : offset + num_bytes].tobytes()
            old = int.from_bytes(raw, byteorder="little", signed=True)
            new = old - value
            self.buffer[offset : offset + num_bytes] = new.to_bytes(
                num_bytes, byteorder="little", signed=True
            )
            return old

    def atomic_cas(
        self, offset: int, expected: int, new: int, num_bytes: int = 4
    ) -> bool:
        """Compare-and-swap integer at ``offset`` with locking."""

        if offset < 0 or offset + num_bytes > self.size:
            raise IndexError("SharedMemory atomic out of bounds")
        with self.lock:
            raw = memoryview(self.buffer)[offset : offset + num_bytes].tobytes()
            current = int.from_bytes(raw, byteorder="little", signed=True)
            if current == expected:
                self.buffer[offset : offset + num_bytes] = new.to_bytes(
                    num_bytes, byteorder="little", signed=True
                )
                return True
            return False

    def atomic_max(self, offset: int, value: int, num_bytes: int = 4) -> int:
        """Atomically store ``max(current, value)`` and return the old value."""

        if offset < 0 or offset + num_bytes > self.size:
            raise IndexError("SharedMemory atomic out of bounds")
        with self.lock:
            raw = memoryview(self.buffer)[offset : offset + num_bytes].tobytes()
            old = int.from_bytes(raw, byteorder="little", signed=True)
            m = max(old, value)
            self.buffer[offset : offset + num_bytes] = m.to_bytes(
                num_bytes, byteorder="little", signed=True
            )
            return old

    def atomic_min(self, offset: int, value: int, num_bytes: int = 4) -> int:
        """Atomically store ``min(current, value)`` and return the old value."""

        if offset < 0 or offset + num_bytes > self.size:
            raise IndexError("SharedMemory atomic out of bounds")
        with self.lock:
            raw = memoryview(self.buffer)[offset : offset + num_bytes].tobytes()
            old = int.from_bytes(raw, byteorder="little", signed=True)
            m = min(old, value)
            self.buffer[offset : offset + num_bytes] = m.to_bytes(
                num_bytes, byteorder="little", signed=True
            )
            return old

    def atomic_exchange(self, offset: int, value: int, num_bytes: int = 4) -> int:
        """Atomically replace value at ``offset`` and return the old value."""

        if offset < 0 or offset + num_bytes > self.size:
            raise IndexError("SharedMemory atomic out of bounds")
        with self.lock:
            raw = memoryview(self.buffer)[offset : offset + num_bytes].tobytes()
            old = int.from_bytes(raw, byteorder="little", signed=True)
            self.buffer[offset : offset + num_bytes] = value.to_bytes(
                num_bytes, byteorder="little", signed=True
            )
            return old


__all__ = ["SharedMemory"]
