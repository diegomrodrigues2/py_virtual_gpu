from __future__ import annotations

from multiprocessing import Barrier
from typing import Callable, List, Tuple, Any

from .shared_memory import SharedMemory
from .thread import Thread


class ThreadBlock:
    """Represent a block of threads with shared memory and barrier sync."""

    def __init__(
        self,
        block_idx: Tuple[int, int, int],
        block_dim: Tuple[int, int, int],
        grid_dim: Tuple[int, int, int],
        shared_mem_size: int,
    ) -> None:
        """Create a ``ThreadBlock`` instance.

        Parameters
        ----------
        block_idx, block_dim, grid_dim:
            Identify the block within the grid and its dimensions.
        shared_mem_size:
            Size in bytes of the block's :class:`SharedMemory`.

        Notes
        -----
        A :class:`multiprocessing.Barrier` is created with ``block_dim`` product
        to emulate the behaviour of ``__syncthreads()``. All threads in the
        block share this barrier via :meth:`initialize_threads`.
        """

        self.block_idx: Tuple[int, int, int] = block_idx
        self.block_dim: Tuple[int, int, int] = block_dim
        self.grid_dim: Tuple[int, int, int] = grid_dim
        self.shared_mem: SharedMemory = SharedMemory(shared_mem_size)
        total_threads = block_dim[0] * block_dim[1] * block_dim[2]
        self.barrier: Barrier = Barrier(parties=total_threads)
        self.threads: List[Thread] = []
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------
    def initialize_threads(self, kernel_func: Callable[..., Any], *args: Any) -> None:
        """Instantiate :class:`Thread` objects for this block.

        Each created :class:`Thread` receives references to this block's
        ``SharedMemory`` and ``Barrier`` so that a kernel can synchronize via
        :meth:`barrier_sync`.
        """
        if self._initialized:
            return
        for z in range(self.block_dim[2]):
            for y in range(self.block_dim[1]):
                for x in range(self.block_dim[0]):
                    thread_idx = (x, y, z)
                    t = Thread()
                    # Store context attributes for future use
                    setattr(t, "thread_idx", thread_idx)
                    setattr(t, "block_idx", self.block_idx)
                    setattr(t, "block_dim", self.block_dim)
                    setattr(t, "grid_dim", self.grid_dim)
                    setattr(t, "shared_mem", self.shared_mem)
                    setattr(t, "barrier", self.barrier)
                    self.threads.append(t)
        self._initialized = True

    def execute(self, kernel_func: Callable[..., Any], *args: Any) -> None:
        """Run all threads in this block invoking their ``run`` method."""
        self.initialize_threads(kernel_func, *args)
        for t in self.threads:
            run = getattr(t, "run", None)
            if callable(run):
                params = (
                    t.thread_idx,
                    t.block_idx,
                    t.block_dim,
                    t.grid_dim,
                    *args,
                )
                run(kernel_func, *params)

    # ------------------------------------------------------------------
    # Synchronization
    # ------------------------------------------------------------------
    def barrier_sync(self) -> None:
        """Synchronize all threads in this block.

        This simply calls ``Barrier.wait`` so that every thread created by this
        block pauses until the last one reaches the same point, mimicking the
        semantics of ``__syncthreads()`` on a real GPU.
        """
        self.barrier.wait()

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<ThreadBlock idx={self.block_idx} "
            f"threads={len(self.threads)} block_dim={self.block_dim}>"
        )


__all__ = ["ThreadBlock"]
