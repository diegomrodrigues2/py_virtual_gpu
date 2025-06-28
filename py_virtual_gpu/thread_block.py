from __future__ import annotations

from multiprocessing import Barrier, Process
from uuid import uuid4
from multiprocessing import Array
from threading import BrokenBarrierError
from multiprocessing import Lock
from typing import Callable, List, Tuple, Any
from threading import Thread as _PyThread
from time import perf_counter

from .shared_memory import SharedMemory
from .errors import SynchronizationError
from .thread import Thread


class ThreadBlock:
    """Represent a block of threads with shared memory and barrier sync."""

    def __init__(
        self,
        block_idx: Tuple[int, int, int],
        block_dim: Tuple[int, int, int],
        grid_dim: Tuple[int, int, int],
        shared_mem_size: int,
        *,
        barrier_timeout: float | None = None,
    ) -> None:
        """Create a ``ThreadBlock`` instance.

        Parameters
        ----------
        block_idx, block_dim, grid_dim:
            Identify the block within the grid and its dimensions.
        shared_mem_size:
            Size in bytes of the block's :class:`SharedMemory`.
        barrier_timeout:
            Maximum time in seconds threads will wait on the barrier before
            raising :class:`SynchronizationError`.

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
        self.warp_buffer = Array('i', total_threads)
        self.uid: str = uuid4().hex
        self.threads: List[Thread] = []
        self._initialized: bool = False
        self.barrier_timeout = barrier_timeout
        self.barrier_wait_time: float = 0.0
        self._entry_times: list[float] = []
        self._barrier_lock = Lock()

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------
    def initialize_threads(
        self,
        kernel_func: Callable[..., Any],
        *args: Any,
        register_mem_size: int = 0,
        local_mem_size: int | None = None,
    ) -> None:
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
                    t = Thread(
                        register_mem_size=register_mem_size,
                        local_mem_size=local_mem_size,
                    )
                    # Store context attributes for future use
                    setattr(t, "thread_idx", thread_idx)
                    setattr(t, "block_idx", self.block_idx)
                    setattr(t, "block_dim", self.block_dim)
                    setattr(t, "grid_dim", self.grid_dim)
                    setattr(t, "shared_mem", self.shared_mem)
                    setattr(t, "barrier", self.barrier)
                    setattr(t, "barrier_uid", self.uid)
                    setattr(t, "warp_buffer", self.warp_buffer)
                    setattr(t, "barrier_timeout", self.barrier_timeout)
                    setattr(t, "block", self)
                    self.threads.append(t)
        self._initialized = True

    def execute(
        self,
        kernel_func: Callable[..., Any],
        *args: Any,
        use_threads: bool = False,
    ) -> None:
        """Run all threads in this block invoking their ``run`` method.

        Parameters
        ----------
        kernel_func:
            Kernel function each thread/process should execute.
        *args:
            Extra arguments forwarded to ``kernel_func``.
        use_threads:
            When ``True`` use ``threading.Thread`` instead of
            ``multiprocessing.Process``. This provides a fallback for
            environments where process forking is undesirable.
        """
        self.initialize_threads(kernel_func, *args)
        if not use_threads:
            # Spawn start method (used on Windows) requires all arguments to be
            # picklable. Kernel functions defined interactively often are not,
            # leading to ``PicklingError``. In that case we fall back to using
            # ``threading.Thread`` which works regardless of picklability and
            # preserves behaviour on platforms without ``fork``.
            import multiprocessing as _mp

            if _mp.get_start_method(allow_none=True) == "spawn":
                use_threads = True

        Worker = _PyThread if use_threads else Process
        workers: List[Worker] = []
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
                worker = Worker(target=run, args=(kernel_func, *params))
                workers.append(worker)
                worker.start()

        for worker in workers:
            worker.join()

    # ------------------------------------------------------------------
    # Synchronization
    # ------------------------------------------------------------------
    def barrier_sync(self) -> None:
        """Synchronize all threads in this block.

        This simply calls ``Barrier.wait`` so that every thread created by this
        block pauses until the last one reaches the same point, mimicking the
        semantics of ``__syncthreads()`` on a real GPU.
        """
        timestamp = perf_counter()
        with self._barrier_lock:
            self._entry_times.append(timestamp)
            parties = getattr(self.barrier, "parties", len(self.threads))
            if len(self._entry_times) == parties:
                diff = max(self._entry_times) - min(self._entry_times)
                self.barrier_wait_time += diff
                self._entry_times.clear()
        try:
            self.barrier.wait(timeout=self.barrier_timeout)
        except BrokenBarrierError as exc:
            raise SynchronizationError("Barrier wait timed out") from exc

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<ThreadBlock idx={self.block_idx} "
            f"threads={len(self.threads)} block_dim={self.block_dim}>"
        )


__all__ = ["ThreadBlock"]
