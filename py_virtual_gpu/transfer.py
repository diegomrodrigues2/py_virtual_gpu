from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TransferEvent"]

@dataclass
class TransferEvent:
    """Record a host<->device memory copy."""

    direction: str  # "H2D" or "D2H"
    size: int
    start_cycle: int
    end_cycle: int
