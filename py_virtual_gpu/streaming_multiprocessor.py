"""Basic skeleton of the StreamingMultiprocessor class."""

from __future__ import annotations

from multiprocessing import Queue
from typing import List, Dict

from .shared_memory import SharedMemory  # type: ignore
from .thread_block import ThreadBlock  # type: ignore
from .thread import Thread  # type: ignore


class StreamingMultiprocessor:
    """Simulated Streaming Multiprocessor (SM)."""

    def __init__(
        self,
        id: int,
        shared_mem_size: int,
        max_registers_per_thread: int,
        warp_size: int = 32,
    ) -> None:
        """Initialize the SM with configuration parameters."""
        self.id: int = id
        self.block_queue: Queue = Queue()
        self.shared_mem: SharedMemory = SharedMemory(shared_mem_size)
        self.max_registers_per_thread: int = max_registers_per_thread
        self.warp_size: int = warp_size
        self.counters: Dict[str, int] = {
            "warps_executed": 0,
            "warp_divergences": 0,
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def fetch_and_execute(self) -> None:
        """Fetch a block from ``block_queue`` and execute it."""
        while not self.block_queue.empty():
            block = self.block_queue.get()
            self.execute_block(block)

    def execute_block(self, block: ThreadBlock) -> None:
        """Split the block into warps and execute them sequentially."""
        threads = block.threads
        for idx in range(0, len(threads), self.warp_size):
            warp = threads[idx : idx + self.warp_size]
            self.execute_warp(warp)
            self.counters["warps_executed"] += 1

    def execute_warp(self, warp_threads: List[Thread]) -> None:
        """Conceptual lock-step execution of a warp."""
        # Placeholder: the real implementation will iterate over instructions
        # and manage SIMT execution and divergence.
        divergent = False  # In a real system, detect divergence here.
        if divergent:
            self.record_divergence(warp_threads)

    # ------------------------------------------------------------------
    # Divergence and counters
    # ------------------------------------------------------------------
    def record_divergence(self, warp_threads: List[Thread]) -> None:
        """Record a warp divergence event."""
        self.counters["warp_divergences"] += 1

    # ------------------------------------------------------------------
    # Maintenance helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the queue and reset counters."""
        self.block_queue = Queue()
        for key in self.counters:
            self.counters[key] = 0

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<SM id={self.id} queue_size={self.block_queue.qsize()} "
            f"warps={self.counters['warps_executed']} "
            f"divergences={self.counters['warp_divergences']}>"
        )


__all__ = ["StreamingMultiprocessor"]
