"""Skeleton implementation of the VirtualGPU class."""

from __future__ import annotations

from multiprocessing import Queue, Pool
from typing import List, Any, Tuple, Optional

# Placeholder imports for yet-to-be-implemented classes.
from .global_memory import GlobalMemory  # type: ignore  # noqa: F401
from .streaming_multiprocessor import StreamingMultiprocessor  # type: ignore  # noqa: F401
from .thread_block import ThreadBlock  # type: ignore  # noqa: F401


class VirtualGPU:
    """Simulated GPU device aggregating multiple SMs and global memory.

    This class orchestrates memory management and kernel execution. Refer to
    the architectural overview in ``RESEARCH.md`` for the design rationale.
    """

    def __init__(self, num_sms: int, global_mem_size: int) -> None:
        """Initialize the virtual device with ``num_sms`` SMs and global memory.

        Parameters
        ----------
        num_sms:
            Number of streaming multiprocessors simulated.
        global_mem_size:
            Size of the global memory space in bytes/words.
        """
        self.sms: List[StreamingMultiprocessor] = []
        self.global_memory: Optional[GlobalMemory] = None
        self.block_queue: Optional[Queue] = None
        self.pool: Optional[Pool] = None
        # Initialization logic for SMs and memory will be implemented in upcoming issues.

    def malloc(self, size: int) -> Any:
        """Allocate ``size`` units in global memory and return a device pointer."""
        raise NotImplementedError

    def free(self, ptr: Any) -> None:
        """Free a previously allocated device pointer from global memory."""
        raise NotImplementedError

    def memcpy(self, dest: Any, src: Any, size: int, direction: str) -> None:
        """Copy data between host and device according to ``direction``.

        Parameters
        ----------
        direction:
            Must indicate HostToDevice, DeviceToHost, or DeviceToDevice.
        """
        raise NotImplementedError

    def launch_kernel(
        self,
        kernel_func: Any,
        grid_dim: Tuple[int, ...],
        block_dim: Tuple[int, ...],
        *args: Any,
    ) -> None:
        """Divide the grid into blocks and schedule them for execution.

        The exact splitting strategy and block scheduling are described in
        ``RESEARCH.md`` and will be implemented in future issues.
        """
        # 1. Compute the block indices for a 1D/2D/3D grid.
        # 2. Create ``ThreadBlock`` instances with their coordinates.
        # 3. Enqueue blocks into ``block_queue`` or submit them to ``pool``.
        # 4. Pass simulated pointers and any shared memory context as needed.
        raise NotImplementedError

    def synchronize(self) -> None:
        """Wait for all enqueued kernels to finish execution."""
        raise NotImplementedError
