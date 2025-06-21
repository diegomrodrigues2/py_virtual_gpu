"""Utility classes for device memory management."""

from __future__ import annotations


class DevicePointer:
    """Opaque reference to an offset inside the global memory."""

    def __init__(self, offset: int) -> None:
        self.offset = offset

    def __repr__(self) -> str:
        return f"<DevicePointer offset={self.offset}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DevicePointer) and self.offset == other.offset


__all__ = ["DevicePointer"]
