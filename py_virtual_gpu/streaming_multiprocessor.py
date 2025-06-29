"""Basic skeleton of the StreamingMultiprocessor class."""

from __future__ import annotations

from queue import Queue
from typing import List, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime

from .shared_memory import SharedMemory  # type: ignore
from .thread_block import ThreadBlock  # type: ignore
from .thread import Thread  # type: ignore
from .warp import Warp
from .dispatch import Instruction


@dataclass
class BlockEvent:
    """Record the execution of a thread block on this SM."""

    block_idx: tuple[int, int, int]
    sm_id: int
    phase: str  # "start" or "end"
    start_cycle: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

if TYPE_CHECKING:  # pragma: no cover - type hinting
    from .virtualgpu import VirtualGPU


@dataclass
class DivergenceEvent:
    """Capture information about a warp divergence event."""

    warp_id: int
    pc: int
    mask_before: List[bool]
    mask_after: List[bool]
    start_cycle: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class StreamingMultiprocessor:
    """Simulated Streaming Multiprocessor (SM)."""

    def __init__(
        self,
        id: int,
        shared_mem_size: int,
        max_registers_per_thread: int,
        warp_size: int = 32,
        *,
        parent_gpu: Optional["VirtualGPU"] = None,
        fp16_cycles: int = 2,
        fp32_cycles: int = 4,
        fp64_cycles: int = 8,
    ) -> None:
        """Initialize the SM with configuration parameters."""
        self.id: int = id
        self.block_queue: Queue = Queue()
        self.warp_queue: Queue = Queue()
        self.shared_mem: SharedMemory = SharedMemory(shared_mem_size)
        self.max_registers_per_thread: int = max_registers_per_thread
        self.warp_size: int = warp_size
        self.fp16_cycles = fp16_cycles
        self.fp32_cycles = fp32_cycles
        self.fp64_cycles = fp64_cycles
        self.schedule_policy: str = "round_robin"
        self.counters: Dict[str, int] = {
            "warps_executed": 0,
            "warp_divergences": 0,
            "non_coalesced_accesses": 0,
            "bank_conflicts": 0,
            "barrier_wait_ms": 0,
            "cycles": 0,
        }
        self.stats: Dict[str, int] = {"extra_cycles": 0}
        self.divergence_log: List[DivergenceEvent] = []
        self.block_log: List[BlockEvent] = []
        self.gpu = parent_gpu

    # ------------------------------------------------------------------
    # Cycle accounting
    # ------------------------------------------------------------------
    def _dtype_cycles(self, dtype: object) -> int:
        """Return latency in cycles for the given ``dtype``."""

        import numpy as np

        if dtype is None:
            return 0
        try:
            dt = np.dtype(dtype)
        except Exception:
            return 0
        if dt == np.float16:
            return self.fp16_cycles
        if dt == np.float32:
            return self.fp32_cycles
        if dt == np.float64:
            return self.fp64_cycles
        return 0

    def account_instruction(self, inst: Instruction) -> None:
        """Update cycle counters for ``inst`` based on operand dtype."""

        dtype = None
        for op in inst.operands:
            if hasattr(op, "dtype"):
                dtype = getattr(op, "dtype")
                break
            if isinstance(op, type) and hasattr(op, "dtype"):
                dtype = getattr(op, "dtype")
                break
        cycles = self._dtype_cycles(dtype)
        if cycles:
            self.counters["cycles"] += cycles
            if self.gpu is not None:
                self.gpu._cycle_counter += cycles

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def fetch_and_execute(self) -> None:
        """Fetch a block from ``block_queue`` and execute it."""
        while not self.block_queue.empty():
            block = self.block_queue.get()
            self.execute_block(block)

    def execute_block(self, block: ThreadBlock) -> None:
        """Execute ``block`` either directly or via warp scheduling."""

        start = self.gpu.current_cycle() if self.gpu is not None else len(self.block_log)
        idx = getattr(block, "block_idx", (0, 0, 0))
        self.block_log.append(BlockEvent(idx, self.id, "start", start))

        # ``ThreadBlock`` instances created by :class:`VirtualGPU` store the
        # kernel function and its arguments. When present we run the block
        # directly using Python threads which prevents the unfinished ``Warp``
        # logic from looping indefinitely.
        if hasattr(block, "kernel_func"):
            func = getattr(block, "kernel_func")
            args = getattr(block, "kernel_args", tuple())
            block.execute(func, *args)
            num_warps = (len(block.threads) + self.warp_size - 1) // self.warp_size
            self.counters["warps_executed"] += num_warps
            self.counters["barrier_wait_ms"] += int(getattr(block, "barrier_wait_time", 0.0) * 1000)
            if self.gpu is not None:
                self.gpu._cycle_counter += 1
            idx = getattr(block, "block_idx", (0, 0, 0))
            self.block_log.append(BlockEvent(idx, self.id, "end", self.gpu.current_cycle() if self.gpu is not None else len(self.block_log)))
            return

        # Fallback for tests or custom blocks that do not define a kernel.
        warps: List[Warp] = []
        threads = block.threads
        for idx in range(0, len(threads), self.warp_size):
            warp_threads = threads[idx : idx + self.warp_size]
            warp = Warp(id=len(warps), threads=warp_threads, sm=self)
            warps.append(warp)
            self.warp_queue.put(warp)

        if self.schedule_policy == "sequential":
            self._run_sequential(warps)
            while not self.warp_queue.empty():
                self.warp_queue.get()
        elif self.schedule_policy == "round_robin":
            self._run_round_robin()
        else:
            self.dispatch()

        self.counters["barrier_wait_ms"] += int(getattr(block, "barrier_wait_time", 0.0) * 1000)
        if self.gpu is not None:
            self.gpu._cycle_counter += 1
        idx = getattr(block, "block_idx", (0, 0, 0))
        self.block_log.append(BlockEvent(idx, self.id, "end", self.gpu.current_cycle() if self.gpu is not None else len(self.block_log)))

    def execute_warp(self, warp_threads: List[Thread]) -> None:
        """Conceptual lock-step execution of a warp."""
        # Placeholder: the real implementation will iterate over instructions
        # and manage SIMT execution and divergence.
        divergent = False  # In a real system, detect divergence here.
        if divergent:
            self.record_divergence(warp_threads)

    def _run_sequential(self, warps: List[Warp]) -> None:
        """Run each warp to completion sequentially."""
        for warp in warps:
            warp.execute()
            self.counters["warps_executed"] += 1

    def _run_round_robin(self) -> None:
        """Run warps in a round-robin manner until all complete."""
        while True:
            try:
                warp: Warp = self.warp_queue.get_nowait()
            except Exception:
                break
            warp.execute()
            self.counters["warps_executed"] += 1
            if any(warp.active_mask):
                self.warp_queue.put(warp)

    def dispatch(self) -> None:
        """Issue one instruction to each scheduled warp in round-robin."""

        while True:
            try:
                warp: Warp = self.warp_queue.get_nowait()
            except Exception:
                break
            warp.execute()
            self.counters["warps_executed"] += 1
            if any(warp.active_mask):
                self.warp_queue.put(warp)

    # ------------------------------------------------------------------
    # Divergence and counters
    # ------------------------------------------------------------------
    def record_divergence(
        self,
        warp: Warp,
        pc: int,
        mask_before: List[bool],
        mask_after: List[bool],
    ) -> None:
        """Record a warp divergence event and store it in ``divergence_log``."""

        self.counters["warp_divergences"] += 1
        start = self.gpu.current_cycle() if self.gpu is not None else len(self.divergence_log)
        event = DivergenceEvent(
            warp_id=warp.id,
            pc=pc,
            mask_before=mask_before.copy(),
            mask_after=mask_after.copy(),
            start_cycle=start,
        )
        self.divergence_log.append(event)
        if self.gpu is not None:
            self.gpu._cycle_counter += 1

    def get_divergence_log(self) -> List[DivergenceEvent]:
        """Return a list with all recorded divergence events."""

        return list(self.divergence_log)

    def clear_divergence_log(self) -> None:
        """Clear ``divergence_log`` without resetting counters."""

        self.divergence_log.clear()

    def get_block_event_log(self) -> List[BlockEvent]:
        """Return a list of recorded block execution events."""

        return list(self.block_log)

    def clear_block_event_log(self) -> None:
        """Clear ``block_log`` without resetting counters."""

        self.block_log.clear()

    # ------------------------------------------------------------------
    # Statistics reporting
    # ------------------------------------------------------------------
    def report_coalescing_stats(self) -> Dict[str, int]:
        """Return current statistics for memory coalescing."""

        return {
            "non_coalesced_accesses": self.counters["non_coalesced_accesses"],
            "extra_cycles": self.stats["extra_cycles"],
        }

    def report_bank_conflict_stats(self) -> Dict[str, int]:
        """Return current statistics for shared memory bank conflicts."""

        return {
            "bank_conflicts": self.counters["bank_conflicts"],
            "extra_cycles": self.stats["extra_cycles"],
        }

    # ------------------------------------------------------------------
    # Maintenance helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the queue and reset counters."""
        self.block_queue = Queue()
        self.warp_queue = Queue()
        for key in self.counters:
            self.counters[key] = 0
        for key in self.stats:
            self.stats[key] = 0
        self.divergence_log.clear()

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<SM id={self.id} queue_size={self.block_queue.qsize()} "
            f"warps={self.counters['warps_executed']} "
            f"divergences={self.counters['warp_divergences']}>"
        )


__all__ = ["StreamingMultiprocessor", "DivergenceEvent", "BlockEvent"]
