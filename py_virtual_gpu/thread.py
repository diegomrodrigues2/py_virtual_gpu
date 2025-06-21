"""Thread execution unit and register memory abstractions."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple
from multiprocessing import Barrier

from .shared_memory import SharedMemory
from .global_memory import GlobalMemory


class RegisterMemory:
    """Simple key-value store representing registers of a thread."""

    def __init__(self, size_bytes: int) -> None:
        self.size: int = size_bytes
        self._storage: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Basic register operations
    # ------------------------------------------------------------------
    def read(self, name: str) -> Any:
        """Return the value stored in register ``name`` or ``None``."""

        return self._storage.get(name)

    def write(self, name: str, value: Any) -> None:
        """Write ``value`` into register ``name``."""

        self._storage[name] = value

    def clear(self) -> None:
        """Clear all registers."""

        self._storage.clear()


class Thread:
    """Represent a single kernel thread with its own register set."""

    def __init__(
        self,
        thread_idx: Tuple[int, int, int] | None = None,
        block_idx: Tuple[int, int, int] | None = None,
        block_dim: Tuple[int, int, int] | None = None,
        grid_dim: Tuple[int, int, int] | None = None,
        register_mem_size: int = 0,
        shared_mem: SharedMemory | None = None,
        global_mem: GlobalMemory | None = None,
        barrier: Barrier | None = None,
    ) -> None:
        """Initialize a thread context.

        Parameters
        ----------
        thread_idx, block_idx, block_dim, grid_dim:
            Indices and dimensions identifying this thread within the grid.
        register_mem_size:
            Size in bytes of the private register file.
        shared_mem, global_mem:
            References to the memory spaces accessible by the thread.
        barrier:
            Optional :class:`multiprocessing.Barrier` used for ``syncthreads``-like
            synchronization within a block.
        """

        # Indices and dimensions
        self.thread_idx: Tuple[int, int, int] = thread_idx or (0, 0, 0)
        self.block_idx: Tuple[int, int, int] = block_idx or (0, 0, 0)
        self.block_dim: Tuple[int, int, int] = block_dim or (1, 1, 1)
        self.grid_dim: Tuple[int, int, int] = grid_dim or (1, 1, 1)

        # Private register memory
        self.registers = RegisterMemory(register_mem_size)

        # Memory references
        self.shared_mem = shared_mem
        self.global_mem = global_mem
        self.barrier = barrier

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(self, kernel_func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute ``kernel_func`` injecting thread indices and dimensions.

        The ``self.barrier`` attribute can be used within ``kernel_func``
        to synchronize with other threads of the same block, emulating the
        behaviour of ``__syncthreads()`` in CUDA.

        Parameters
        ----------
        kernel_func:
            Function representing the kernel to be executed.
        *args:
            Expected as ``(threadIdx, blockIdx, blockDim, gridDim, *user_args)``.

        Returns
        -------
        Any
            Whatever ``kernel_func`` returns.
        """

        if len(args) < 4:
            raise TypeError(
                "run expects threadIdx, blockIdx, blockDim and gridDim as the first four arguments"
            )

        threadIdx, blockIdx, blockDim, gridDim, *user_args = args
        # expose as attributes used by the kernel
        self.threadIdx = threadIdx
        self.blockIdx = blockIdx
        self.blockDim = blockDim
        self.gridDim = gridDim

        return kernel_func(threadIdx, blockIdx, blockDim, gridDim, *user_args, **kwargs)

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        tx, ty, tz = self.thread_idx
        bx, by, bz = self.block_idx
        return (
            f"<Thread idx=({tx},{ty},{tz}) blk=({bx},{by},{bz}) "
            f"regs={len(self.registers._storage)}>"
        )


__all__ = ["RegisterMemory", "Thread"]
