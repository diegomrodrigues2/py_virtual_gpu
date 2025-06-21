"""Basic skeleton of the StreamingMultiprocessor class."""

from __future__ import annotations

from multiprocessing import Queue
from queue import Queue as LocalQueue
from typing import List, Dict

from .shared_memory import SharedMemory  # type: ignore
from .thread_block import ThreadBlock  # type: ignore
from .thread import Thread  # type: ignore
from .warp import Warp


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
        self.warp_queue: LocalQueue = LocalQueue()
        self.shared_mem: SharedMemory = SharedMemory(shared_mem_size)
        self.max_registers_per_thread: int = max_registers_per_thread
        self.warp_size: int = warp_size
        self.schedule_policy: str = "round_robin"
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
        """Split the block into warps and execute them."""
        warps: List[Warp] = []
        threads = block.threads
        for idx in range(0, len(threads), self.warp_size):
            warp_threads = threads[idx : idx + self.warp_size]
            warp = Warp(id=len(warps), threads=warp_threads)
            warps.append(warp)
            self.warp_queue.put(warp)

        if self.schedule_policy == "sequential":
            self._run_sequential(warps)
            while not self.warp_queue.empty():
                self.warp_queue.get()
        else:
            self._run_round_robin()

        self.counters["warps_executed"] += len(warps)

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

    def _run_round_robin(self) -> None:
        """Run warps in a round-robin manner until all complete."""
        while True:
            try:
                warp: Warp = self.warp_queue.get_nowait()
            except Exception:
                break
            warp.execute()
            if any(warp.active_mask):
                self.warp_queue.put(warp)

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
        self.warp_queue = LocalQueue()
        for key in self.counters:
            self.counters[key] = 0

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<SM id={self.id} queue_size={self.block_queue.qsize()} "
            f"warps={self.counters['warps_executed']} "
            f"divergences={self.counters['warp_divergences']}>"
        )


__all__ = ["StreamingMultiprocessor"]
