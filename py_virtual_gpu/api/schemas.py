from __future__ import annotations

from pydantic import BaseModel


class GPUSummary(BaseModel):
    """Basic information about a simulated GPU."""

    id: int
    num_sms: int
    global_mem_size: int
    shared_mem_size: int


class TransferRecord(BaseModel):
    """Serialized memory transfer event."""

    direction: str
    size: int
    start_cycle: int
    end_cycle: int


class GlobalMemState(BaseModel):
    """State of the GPU global memory."""

    size: int
    used: int


class SMState(BaseModel):
    """State information for a single SM."""

    id: int
    status: str
    counters: dict[str, int]


class GPUState(BaseModel):
    """Detailed snapshot of a GPU."""

    id: int
    global_memory: GlobalMemState
    transfer_log: list[TransferRecord]
    sms: list[SMState]

