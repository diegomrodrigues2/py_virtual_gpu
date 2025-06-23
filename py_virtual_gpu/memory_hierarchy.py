from __future__ import annotations

"""Hierarchical memory space abstractions with latency and bandwidth stats."""

from abc import ABC, abstractmethod
from math import ceil
from multiprocessing import Array, Lock
from ctypes import c_byte

__all__ = [
    "MemorySpace",
    "RegisterFile",
    "SharedMemory",
    "L1Cache",
    "L2Cache",
    "GlobalMemorySpace",
    "ConstantMemory",
    "LocalMemory",
    "HostMemory",
]


class MemorySpace(ABC):
    """Abstract memory space that accounts for latency and bandwidth."""

    def __init__(self, size: int, latency_cycles: int, bandwidth_bytes_per_cycle: int) -> None:
        self.size = size
        self.latency_cycles = latency_cycles
        self.bandwidth_bpc = bandwidth_bytes_per_cycle
        self.stats = {"reads": 0, "writes": 0, "cycles": 0}

    @abstractmethod
    def read(self, offset: int, size: int) -> bytes:
        """Return ``size`` bytes starting from ``offset``."""

    @abstractmethod
    def write(self, offset: int, data: bytes) -> None:
        """Write ``data`` starting at ``offset``."""

    # ------------------------------------------------------------------
    # Accounting helpers
    # ------------------------------------------------------------------
    def _account(self, size: int, is_read: bool) -> None:
        if is_read:
            self.stats["reads"] += 1
        else:
            self.stats["writes"] += 1
        cycles = self.latency_cycles + ceil(size / self.bandwidth_bpc)
        self.stats["cycles"] += cycles

    def reset_stats(self) -> None:
        self.stats = {"reads": 0, "writes": 0, "cycles": 0}


class RegisterFile(MemorySpace):
    """Private per-thread register file."""

    def __init__(self, per_thread_regs: int = 8 * 1024) -> None:
        super().__init__(per_thread_regs, latency_cycles=1, bandwidth_bytes_per_cycle=per_thread_regs)
        self.buffer = bytearray(self.size)

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or offset + size > self.size:
            raise IndexError("RegisterFile read out of bounds")
        self._account(size, True)
        return bytes(self.buffer[offset : offset + size])

    def write(self, offset: int, data: bytes) -> None:
        end = offset + len(data)
        if offset < 0 or end > self.size:
            raise IndexError("RegisterFile write out of bounds")
        self._account(len(data), False)
        self.buffer[offset:end] = data


class SharedMemory(MemorySpace):
    """On-chip memory shared by threads of a block."""

    def __init__(self, size: int = 96 * 1024, latency_cycles: int = 10, bandwidth_bytes_per_cycle: int = 96 * 1024) -> None:
        super().__init__(size, latency_cycles, bandwidth_bytes_per_cycle)
        self.buffer = Array(c_byte, size, lock=False)
        self.lock = Lock()

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or offset + size > self.size:
            raise IndexError("SharedMemory read out of bounds")
        self._account(size, True)
        view = memoryview(self.buffer)
        return view[offset : offset + size].tobytes()

    def write(self, offset: int, data: bytes) -> None:
        end = offset + len(data)
        if offset < 0 or end > self.size:
            raise IndexError("SharedMemory write out of bounds")
        self._account(len(data), False)
        view = memoryview(self.buffer)
        view[offset:end] = data


class L1Cache(MemorySpace):
    """Simple conceptual L1 cache."""

    def __init__(self, size: int = 16 * 1024, latency_cycles: int = 5, bandwidth_bytes_per_cycle: int = 128) -> None:
        super().__init__(size, latency_cycles, bandwidth_bytes_per_cycle)
        self.buffer = bytearray(self.size)

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or offset + size > self.size:
            raise IndexError("L1Cache read out of bounds")
        self._account(size, True)
        return bytes(self.buffer[offset : offset + size])

    def write(self, offset: int, data: bytes) -> None:
        end = offset + len(data)
        if offset < 0 or end > self.size:
            raise IndexError("L1Cache write out of bounds")
        self._account(len(data), False)
        self.buffer[offset:end] = data


class L2Cache(MemorySpace):
    """Conceptual L2 cache shared across the GPU."""

    def __init__(self, size: int = 40 * 1024 * 1024, latency_cycles: int = 50, bandwidth_bytes_per_cycle: int = 256) -> None:
        super().__init__(size, latency_cycles, bandwidth_bytes_per_cycle)
        self.buffer = bytearray(self.size)

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or offset + size > self.size:
            raise IndexError("L2Cache read out of bounds")
        self._account(size, True)
        return bytes(self.buffer[offset : offset + size])

    def write(self, offset: int, data: bytes) -> None:
        end = offset + len(data)
        if offset < 0 or end > self.size:
            raise IndexError("L2Cache write out of bounds")
        self._account(len(data), False)
        self.buffer[offset:end] = data


class GlobalMemorySpace(MemorySpace):
    """Off-chip global memory."""

    def __init__(self, size: int = 32 * 1024 * 1024 * 1024, latency_cycles: int = 200, bandwidth_bytes_per_cycle: int = 32) -> None:
        super().__init__(size, latency_cycles, bandwidth_bytes_per_cycle)
        self.buffer = bytearray(self.size)

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or offset + size > self.size:
            raise IndexError("GlobalMemory read out of bounds")
        self._account(size, True)
        return bytes(self.buffer[offset : offset + size])

    def write(self, offset: int, data: bytes) -> None:
        end = offset + len(data)
        if offset < 0 or end > self.size:
            raise IndexError("GlobalMemory write out of bounds")
        self._account(len(data), False)
        self.buffer[offset:end] = data


class ConstantMemory(MemorySpace):
    """Read-only constant memory with its own cache."""

    def __init__(self, size: int = 64 * 1024, latency_cycles: int = 5, bandwidth_bytes_per_cycle: int = 128) -> None:
        super().__init__(size, latency_cycles, bandwidth_bytes_per_cycle)
        self.buffer = bytearray(self.size)

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or offset + size > self.size:
            raise IndexError("ConstantMemory read out of bounds")
        self._account(size, True)
        return bytes(self.buffer[offset : offset + size])

    def write(self, offset: int, data: bytes) -> None:
        raise RuntimeError("ConstantMemory is read-only")


class LocalMemory(MemorySpace):
    """Per-thread spill-over memory with global memory latency."""

    def __init__(self, size: int, latency_cycles: int = 200, bandwidth_bytes_per_cycle: int = 32) -> None:
        super().__init__(size, latency_cycles, bandwidth_bytes_per_cycle)
        self.buffer = bytearray(self.size)

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or offset + size > self.size:
            raise IndexError("LocalMemory read out of bounds")
        self._account(size, True)
        return bytes(self.buffer[offset : offset + size])

    def write(self, offset: int, data: bytes) -> None:
        end = offset + len(data)
        if offset < 0 or end > self.size:
            raise IndexError("LocalMemory write out of bounds")
        self._account(len(data), False)
        self.buffer[offset:end] = data


class HostMemory(MemorySpace):
    """Memory residing on the host side."""

    def __init__(self, size: int = 0, latency_cycles: int = 1000, bandwidth_bytes_per_cycle: int = 16) -> None:
        super().__init__(size, latency_cycles, bandwidth_bytes_per_cycle)
        self.buffer = bytearray(self.size)

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or offset + size > self.size:
            raise IndexError("HostMemory read out of bounds")
        self._account(size, True)
        return bytes(self.buffer[offset : offset + size])

    def write(self, offset: int, data: bytes) -> None:
        end = offset + len(data)
        if offset < 0 or end > self.size:
            raise IndexError("HostMemory write out of bounds")
        self._account(len(data), False)
        self.buffer[offset:end] = data

