"""Skeleton implementation of the VirtualGPU class."""

from __future__ import annotations

from multiprocessing import Queue, Pool
from typing import List, Any, Tuple, Optional, Callable

# Placeholder imports for yet-to-be-implemented classes.
from .global_memory import GlobalMemory
from .memory import DevicePointer
from .streaming_multiprocessor import StreamingMultiprocessor  # type: ignore  # noqa: F401
from .thread_block import ThreadBlock  # type: ignore  # noqa: F401


def _execute_block_worker(tb: ThreadBlock, func: Callable[..., Any], args: Tuple[Any, ...]) -> None:
    """Helper for ``multiprocessing.Pool`` to execute a block."""

    tb.execute(func, *args)


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

    def __init__(
        self,
        num_sms: int,
        global_mem_size: int,
        shared_mem_size: int = 0,
        *,
        use_pool: bool = False,
        sync_on_launch: bool = False,
    ) -> None:
        """Initialize the virtual device with ``num_sms`` SMs and global memory.

        Parameters
        ----------
        num_sms:
            Number of streaming multiprocessors simulated.
        global_mem_size:
            Size of the global memory space in bytes/words.
        shared_mem_size:
            Size of the per-block shared memory in bytes.
        use_pool:
            If ``True`` a :class:`multiprocessing.Pool` with ``num_sms`` workers
            will be created and each :class:`ThreadBlock` scheduled to it for
            execution.
        sync_on_launch:
            When ``True`` calls :meth:`synchronize` automatically at the end of
            :meth:`launch_kernel`.
        """
        self.sms: List[StreamingMultiprocessor] = [
            StreamingMultiprocessor(i, shared_mem_size, 64)
            for i in range(num_sms)
        ]
        self.global_memory: GlobalMemory = GlobalMemory(global_mem_size)
        self.shared_mem_size: int = shared_mem_size
        self.use_pool: bool = use_pool
        self.sync_on_launch: bool = sync_on_launch
        self.next_sm: int = 0
        self.pool: Optional[Pool] = Pool(processes=num_sms) if self.use_pool else None
        self._active_ptrs: set[int] = set()

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
        kernel_func: Callable[..., Any],
        grid_dim: Tuple[int, ...],
        block_dim: Tuple[int, ...],
        *args: Any,
    ) -> None:
        """Divide ``grid_dim`` into blocks and queue them for execution.

        Parameters
        ----------
        kernel_func:
            Kernel function to execute for each thread.
        grid_dim:
            Size of the grid expressed as ``(x, y, z)``.
        block_dim:
            Dimension of each block expressed as ``(x, y, z)``.
        args:
            Extra arguments forwarded to ``kernel_func``.

        Notes
        -----
        If ``use_pool`` was enabled on this :class:`VirtualGPU`, each
        :class:`ThreadBlock` is scheduled through ``Pool.apply_async``; otherwise
        blocks are dispatched to available SMs or executed synchronously when no
        SMs are present.
        """

        gx, gy, gz = (list(grid_dim) + [1, 1, 1])[:3]
        bx, by, bz = (list(block_dim) + [1, 1, 1])[:3]

        for z in range(gz):
            for y in range(gy):
                for x in range(gx):
                    block_idx = (x, y, z)
                    tb = ThreadBlock(
                        block_idx=block_idx,
                        block_dim=(bx, by, bz),
                        grid_dim=(gx, gy, gz),
                        shared_mem_size=self.shared_mem_size,
                    )
                    tb.kernel_func = kernel_func
                    tb.kernel_args = args
                    tb.initialize_threads(kernel_func, *args)
                    for t in tb.threads:
                        setattr(t, "global_mem", self.global_memory)

                    if self.pool is not None:
                        self.pool.apply_async(
                            _execute_block_worker,
                            args=(tb, kernel_func, args),
                        )
                    elif self.sms:
                        sm = self.sms[self.next_sm]
                        sm.block_queue.put(tb)
                        self.next_sm = (self.next_sm + 1) % len(self.sms)
                    else:
                        tb.execute(kernel_func, *args)

        if self.sync_on_launch:
            self.synchronize()

    def synchronize(self) -> None:
        """Wait for all queued blocks to complete execution.

        If a ``multiprocessing.Pool`` is active, it is closed and joined before
        draining the SM queues.
        """

        if self.pool is not None:
            self.pool.close()
            self.pool.join()

        for sm in self.sms:
            sm.fetch_and_execute()
