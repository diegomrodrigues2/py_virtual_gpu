"""Shared memory simulation used by :class:`ThreadBlock`."""

from __future__ import annotations

from multiprocessing import Array, Lock
from ctypes import c_byte
from collections import Counter
import struct


class SharedMemory:
    """Simple byte-addressable shared memory buffer with bank tracking."""

    def __init__(self, size: int, num_banks: int = 32, bank_stride: int = 4) -> None:
        """Create a shared buffer of ``size`` bytes.

        Parameters
        ----------
        size:
            Total buffer size in bytes.
        num_banks:
            Number of conceptual memory banks for conflict detection.
        bank_stride:
            Stride in bytes that maps consecutive addresses to different banks.
        """

        self.size: int = size
        self.buffer = Array(c_byte, size, lock=False)
        self.lock = Lock()
        self.num_banks = num_banks
        self.bank_stride = bank_stride

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

    def atomic_add_float32(self, offset: int, value: float) -> float:
        """Atomically add ``value`` to a 32-bit float at ``offset``."""

        if offset < 0 or offset + 4 > self.size:
            raise IndexError("SharedMemory atomic out of bounds")
        with self.lock:
            raw = memoryview(self.buffer)[offset : offset + 4].tobytes()
            old = struct.unpack("<f", raw)[0]
            new = old + value
            self.buffer[offset : offset + 4] = struct.pack("<f", new)
            return old

    def atomic_add_float64(self, offset: int, value: float) -> float:
        """Atomically add ``value`` to a 64-bit float at ``offset``."""

        if offset < 0 or offset + 8 > self.size:
            raise IndexError("SharedMemory atomic out of bounds")
        with self.lock:
            raw = memoryview(self.buffer)[offset : offset + 8].tobytes()
            old = struct.unpack("<d", raw)[0]
            new = old + value
            self.buffer[offset : offset + 8] = struct.pack("<d", new)
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

    # ------------------------------------------------------------------
    # Bank conflict detection
    # ------------------------------------------------------------------
    def detect_bank_conflicts(self, addrs: list[int]) -> int:
        """Return number of extra threads that collide on the same bank.

        For each address ``addr`` the bank index is computed as
        ``(addr // bank_stride) % num_banks``. If ``n`` threads access the
        same bank simultaneously, ``n-1`` conflicts are counted.
        """

        banks = [ (addr // self.bank_stride) % self.num_banks for addr in addrs ]
        counts = Counter(banks)
        return sum(cnt - 1 for cnt in counts.values() if cnt > 1)


__all__ = ["SharedMemory"]
