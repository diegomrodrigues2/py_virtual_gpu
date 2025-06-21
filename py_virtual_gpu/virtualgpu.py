"""Skeleton implementation of the VirtualGPU class."""

from __future__ import annotations

from multiprocessing import Queue, Pool
from typing import List, Any, Tuple, Optional

# Placeholder imports for yet-to-be-implemented classes.
from .global_memory import GlobalMemory
from .memory import DevicePointer
from .streaming_multiprocessor import StreamingMultiprocessor  # type: ignore  # noqa: F401
from .thread_block import ThreadBlock  # type: ignore  # noqa: F401


class VirtualGPU:
    """Simulated GPU device aggregating multiple SMs and global memory.

    This class orchestrates memory management and kernel execution. Refer to
    the architectural overview in ``RESEARCH.md`` for the design rationale.
    """

    _current: "VirtualGPU" | None = None

    @classmethod
    def set_current(cls, gpu: "VirtualGPU") -> None:
        """Set ``gpu`` as the active device for kernel launches."""

        cls._current = gpu

    @classmethod
    def get_current(cls) -> "VirtualGPU":
        """Return the active device or raise ``RuntimeError`` if unset."""

        if cls._current is None:
            raise RuntimeError("No current VirtualGPU set")
        return cls._current

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
        self.global_memory: GlobalMemory = GlobalMemory(global_mem_size)
        self.block_queue: Optional[Queue] = None
        self.pool: Optional[Pool] = None
        self._active_ptrs: set[int] = set()
        # Initialization logic for SMs will be implemented in upcoming issues.

    def malloc(self, size: int) -> Any:
        """Allocate ``size`` bytes in global memory and return a :class:`DevicePointer`."""

        offset = self.global_memory.malloc(size)
        self._active_ptrs.add(offset)
        return DevicePointer(offset)

    def free(self, ptr: Any) -> None:
        """Free a previously allocated :class:`DevicePointer`."""

        if not isinstance(ptr, DevicePointer):
            raise TypeError("ptr must be a DevicePointer")
        if ptr.offset not in self._active_ptrs:
            raise ValueError("Invalid or double free")
        self.global_memory.free(ptr.offset)
        self._active_ptrs.remove(ptr.offset)

    # ------------------------------------------------------------------
    # Data transfer helpers
    # ------------------------------------------------------------------
    def memcpy_host_to_device(
        self,
        dest_ptr: DevicePointer,
        src: bytes | bytearray | memoryview,
        size: int,
    ) -> None:
        """Copy ``size`` bytes from host ``src`` into device memory ``dest_ptr``.

        Parameters
        ----------
        dest_ptr:
            Destination pointer inside the device memory.
        src:
            Buffer residing on the host. Must contain at least ``size`` bytes.
        size:
            Number of bytes to transfer.

        Raises
        ------
        TypeError
            If ``dest_ptr`` is not a :class:`DevicePointer`.
        ValueError
            If ``dest_ptr`` is invalid or ``src`` is smaller than ``size``.
        """

        if not isinstance(dest_ptr, DevicePointer):
            raise TypeError("dest_ptr must be a DevicePointer")
        if dest_ptr.offset not in self._active_ptrs:
            raise ValueError("Invalid device pointer")
        if size < 0 or len(src) < size:
            raise ValueError("Host buffer too small for memcpy_host_to_device")

        data = bytes(src[:size])
        self.global_memory.write(dest_ptr.offset, data)

    def memcpy_device_to_host(
        self,
        dest: bytearray | memoryview,
        src_ptr: DevicePointer,
        size: int,
    ) -> None:
        """Copy ``size`` bytes from ``src_ptr`` in device memory into host ``dest``.

        Parameters
        ----------
        dest:
            Host buffer that will receive the data.
        src_ptr:
            Pointer to the source region in device memory.
        size:
            Number of bytes to transfer.

        Raises
        ------
        TypeError
            If ``src_ptr`` is not a :class:`DevicePointer`.
        ValueError
            If ``src_ptr`` is invalid or ``dest`` is smaller than ``size``.
        """

        if not isinstance(src_ptr, DevicePointer):
            raise TypeError("src_ptr must be a DevicePointer")
        if src_ptr.offset not in self._active_ptrs:
            raise ValueError("Invalid device pointer")
        if size < 0 or len(dest) < size:
            raise ValueError("Host buffer too small for memcpy_device_to_host")

        data = self.global_memory.read(src_ptr.offset, size)
        dest[:size] = data

    def memcpy(self, dest: Any, src: Any, size: int, direction: str) -> None:
        """Copy data between host and device according to ``direction``."""

        dest_ptr = dest.offset if isinstance(dest, DevicePointer) else dest
        src_ptr = src.offset if isinstance(src, DevicePointer) else src
        return self.global_memory.memcpy(dest_ptr, src_ptr, size, direction)

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
