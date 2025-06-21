"""Minimal shared memory placeholder used by ``StreamingMultiprocessor``."""

class SharedMemory:
    """Simple byte-addressable shared memory buffer."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.buffer = bytearray(size)

    def read(self, offset: int, size: int) -> bytes:
        return bytes(self.buffer[offset : offset + size])

    def write(self, offset: int, data: bytes) -> None:
        self.buffer[offset : offset + len(data)] = data


__all__ = ["SharedMemory"]
