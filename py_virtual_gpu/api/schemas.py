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


class KernelLaunchRecord(BaseModel):
    """Serialized kernel launch event."""

    name: str
    grid_dim: tuple[int, int, int]
    block_dim: tuple[int, int, int]
    start_cycle: int


class GlobalMemState(BaseModel):
    """State of the GPU global memory."""

    size: int
    used: int


class SMState(BaseModel):
    """State information for a single SM."""

    id: int
    status: str
    counters: dict[str, int]


class GPUConfig(BaseModel):
    """Configuration details for a GPU."""

    num_sms: int
    global_mem_size: int
    shared_mem_size: int
    registers_per_sm_total: int | None = None


class GPUState(BaseModel):
    """Detailed snapshot of a GPU."""

    id: int
    name: str
    config: GPUConfig
    global_memory: GlobalMemState
    transfer_log: list[TransferRecord]
    sms: list[SMState]
    overall_load: int
    temperature: int | None = None
    power_draw_watts: int | None = None


class GPUMetrics(BaseModel):
    """Aggregated metrics about GPU execution and transfers."""

    id: int
    instructions: int
    global_accesses: int
    shared_accesses: int
    divergences: int
    memory_stats: dict[str, int]
    transfer_stats: dict[str, int]


class BlockSummary(BaseModel):
    """Minimal information about a block scheduled on an SM."""

    block_idx: tuple[int, int, int]
    status: str


class WarpSummary(BaseModel):
    """Basic state for a warp queued on an SM."""

    id: int
    active_threads: int


class DivergenceRecord(BaseModel):
    """Serialized record of a warp divergence event."""

    warp_id: int
    pc: int
    mask_before: list[bool]
    mask_after: list[bool]


class BlockEventRecord(BaseModel):
    """Serialized record of a thread block event."""

    block_idx: tuple[int, int, int]
    sm_id: int
    phase: str
    start_cycle: int


class SMDetailed(BaseModel):
    """Detailed information for a single StreamingMultiprocessor."""

    id: int
    blocks: list[BlockSummary]
    warps: list[WarpSummary]
    divergence_log: list[DivergenceRecord]
    counters: dict[str, int]
    block_event_log: list[BlockEventRecord]


class MemorySlice(BaseModel):
    """Serialized view of a contiguous memory region."""

    offset: int
    size: int
    data: str


