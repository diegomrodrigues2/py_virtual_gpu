"""Skeleton implementation of the VirtualGPU class."""

from __future__ import annotations

from multiprocessing import Queue, Pool, Barrier
from typing import List, Any, Tuple, Optional, Callable, Type
from math import ceil

# Placeholder imports for yet-to-be-implemented classes.
from .global_memory import GlobalMemory
from .memory import DevicePointer
from .streaming_multiprocessor import StreamingMultiprocessor  # type: ignore  # noqa: F401
from .thread_block import ThreadBlock  # type: ignore  # noqa: F401
from .memory_hierarchy import HostMemory, ConstantMemory
from .transfer import TransferEvent
from .types import Numeric
from dataclasses import dataclass


def _execute_block_worker(
    tb: ThreadBlock, func: Callable[..., Any], args: Tuple[Any, ...]
) -> None:
    """Helper for ``multiprocessing.Pool`` to execute a block."""

    tb.execute(func, *args)


@dataclass
class KernelLaunchEvent:
    """Record a kernel launch invocation."""

    name: str
    grid_dim: Tuple[int, int, int]
    block_dim: Tuple[int, int, int]
    start_cycle: int
    cycles: int = 0


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
        host_latency_cycles: int = 1000,
        host_bandwidth_bpc: int = 16,
        device_latency_cycles: int = 200,
        device_bandwidth_bpc: int = 32,
        constant_mem_size: int = 64 * 1024,
        barrier_timeout: float | None = None,
        preset: str | None = None,
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
        barrier_timeout:
            Maximum time in seconds threads wait on the block barrier before
            raising :class:`SynchronizationError`.
        preset:
            Optional GPU model preset (e.g. ``"RTX3080"`` or ``"A100"``) used to
            configure per-SM operation latencies.
        """
        sm_kwargs = {}
        if preset == "RTX3080":
            sm_kwargs = {"fp16_cycles": 2, "fp32_cycles": 4, "fp64_cycles": 8}
        elif preset == "A100":
            sm_kwargs = {"fp16_cycles": 1, "fp32_cycles": 2, "fp64_cycles": 4}

        self.sms: List[StreamingMultiprocessor] = [
            StreamingMultiprocessor(
                i,
                shared_mem_size,
                64,
                parent_gpu=self,
                **sm_kwargs,
            )
            for i in range(num_sms)
        ]
        self.global_memory: GlobalMemory = GlobalMemory(
            global_mem_size,
            latency_cycles=device_latency_cycles,
            bandwidth_bytes_per_cycle=device_bandwidth_bpc,
        )
        self.global_mem = self.global_memory  # alias for documentation purposes
        self.host_mem = HostMemory(
            size=global_mem_size,
            latency_cycles=host_latency_cycles,
            bandwidth_bytes_per_cycle=host_bandwidth_bpc,
            latency_cycles_host_to_device=host_latency_cycles,
            bandwidth_bpc_host_to_device=host_bandwidth_bpc,
        )
        self.constant_memory: ConstantMemory = ConstantMemory(constant_mem_size)
        self.const_memory = self.constant_memory  # backwards compatibility
        self.shared_mem_size: int = shared_mem_size
        self.barrier_timeout = barrier_timeout
        self.use_pool: bool = use_pool
        self.sync_on_launch: bool = sync_on_launch
        self.next_sm: int = 0
        self.pool: Optional[Pool] = Pool(processes=num_sms) if self.use_pool else None
        self._active_ptrs: set[int] = set()
        self.alloc_metadata: dict[int, tuple[int, int, Type[Numeric] | None, str | None]] = {}
        self._launched_blocks: List[ThreadBlock] = []
        self.transfer_log: List[TransferEvent] = []
        self.kernel_log: List[KernelLaunchEvent] = []
        self.counters: dict[str, int] = {"transfers": 0}
        self.stats: dict[str, int] = {"transfer_bytes": 0, "transfer_cycles": 0}
        self._cycle_counter: int = 0

    def malloc(
        self,
        size: int,
        *,
        dtype: Type[Numeric] | None = None,
        label: str | None = None,
    ) -> DevicePointer:
        """Allocate ``size`` elements and return a :class:`DevicePointer`.

        If ``dtype`` is provided ``size`` is interpreted as the number of
        elements of that type.
        """

        offset = self.global_memory.malloc(size, dtype=dtype)
        self._active_ptrs.add(offset)
        self.alloc_metadata[offset] = (offset, size, dtype, label)
        return DevicePointer(offset, memory=self.global_memory, dtype=dtype)

    def malloc_type(self, count: int, dtype: Type[Numeric]) -> DevicePointer:
        """Convenience wrapper to allocate ``count`` elements of ``dtype``."""

        return self.malloc(count, dtype=dtype)

    def free(self, ptr: DevicePointer) -> None:
        """Free a previously allocated :class:`DevicePointer`."""

        if not isinstance(ptr, DevicePointer):
            raise TypeError("ptr must be a DevicePointer")
        if ptr.memory is not self.global_memory or ptr.offset not in self._active_ptrs:
            raise ValueError("Invalid or double free")
        self.global_memory.free(ptr.offset)
        self._active_ptrs.remove(ptr.offset)
        self.alloc_metadata.pop(ptr.offset, None)

    # ------------------------------------------------------------------
    # Data transfer helpers
    # ------------------------------------------------------------------

    def current_cycle(self) -> int:
        """Return the current simulated cycle."""

        return self._cycle_counter

    def memcpy_host_to_device(
        self, host_buffer: bytes, device_ptr: DevicePointer
    ) -> None:
        """Copy ``host_buffer`` into ``device_ptr`` recording transfer metrics."""

        if not isinstance(device_ptr, DevicePointer):
            raise TypeError("device_ptr must be a DevicePointer")
        if (
            device_ptr.memory is not self.global_memory
            or device_ptr.offset not in self._active_ptrs
        ):
            raise ValueError("Invalid device pointer")

        size = len(host_buffer)
        start = self.current_cycle()
        # GlobalMemory.write now accepts a DevicePointer directly
        self.global_memory.write(device_ptr, host_buffer)
        cycles = self.host_mem.latency_cycles_host_to_device + ceil(
            size / self.host_mem.bandwidth_bpc_host_to_device
        )
        end = start + cycles
        self._cycle_counter = end
        self.stats["transfer_cycles"] += cycles
        self.stats["transfer_bytes"] += size
        self.counters["transfers"] += 1
        self.transfer_log.append(TransferEvent("H2D", size, start, end))

    def memcpy_device_to_host(self, device_ptr: DevicePointer, size: int) -> bytes:
        """Copy ``size`` bytes from ``device_ptr`` back to the host and return them.

        Parameters
        ----------
        device_ptr:
            Pointer to the source region in device memory.
        size:
            Number of bytes to transfer.

        Raises
        ------
        TypeError
            If ``device_ptr`` is not a :class:`DevicePointer`.
        ValueError
            If ``device_ptr`` is invalid or ``size`` is negative.
        """

        if not isinstance(device_ptr, DevicePointer):
            raise TypeError("device_ptr must be a DevicePointer")
        if (
            device_ptr.memory is not self.global_memory
            or device_ptr.offset not in self._active_ptrs
        ):
            raise ValueError("Invalid device pointer")
        if size < 0:
            raise ValueError("Size must be positive")

        start = self.current_cycle()
        # GlobalMemory.read now accepts a DevicePointer directly
        data = self.global_memory.read(device_ptr, size)
        cycles = self.global_memory.latency_cycles + ceil(
            size / self.global_memory.bandwidth_bpc
        )
        end = start + cycles
        self._cycle_counter = end
        self.stats["transfer_cycles"] += cycles
        self.stats["transfer_bytes"] += size
        self.counters["transfers"] += 1
        self.transfer_log.append(TransferEvent("D2H", size, start, end))
        return data

    def memcpy(
        self,
        dest: DevicePointer | int | bytes | None,
        src: DevicePointer | int | bytes | None,
        size: int,
        direction: str,
    ) -> None:
        """Copy data between host and device according to ``direction``."""

        # GlobalMemory.memcpy expects the device offset as the first argument
        # when performing a DeviceToHost transfer. In that case, the host
        # destination is ignored, so we pass ``None`` for the second argument.
        if direction == "DeviceToHost":
            src_ptr = src.offset if isinstance(src, DevicePointer) else src
            return self.global_memory.memcpy(src_ptr, None, size, direction)

        dest_ptr = dest.offset if isinstance(dest, DevicePointer) else dest
        src_ptr = src.offset if isinstance(src, DevicePointer) else src
        return self.global_memory.memcpy(dest_ptr, src_ptr, size, direction)

    def set_constant(self, data: bytes, offset: int = 0) -> None:
        """Copy ``data`` into constant memory starting at ``offset``."""

        end = offset + len(data)
        if offset < 0 or end > self.constant_memory.size:
            raise ValueError("Constant memory write out of bounds")

        start = self.current_cycle()
        view = memoryview(self.constant_memory.buffer)
        view[offset:end] = data
        cycles = self.host_mem.latency_cycles_host_to_device + ceil(
            len(data) / self.host_mem.bandwidth_bpc_host_to_device
        )
        self._cycle_counter = start + cycles
        self.stats["transfer_cycles"] += cycles
        self.stats["transfer_bytes"] += len(data)
        self.counters["transfers"] += 1
        self.transfer_log.append(TransferEvent("H2D", len(data), start, start + cycles))

    def read_constant(self, addr: int, size: int) -> bytes:
        """Return ``size`` bytes starting at ``addr`` from constant memory.

        This is a convenience wrapper around :meth:`ConstantMemory.read`
        so that kernels can call ``VirtualGPU.get_current().read_constant``
        instead of accessing ``thread.const_mem`` directly.
        """

        return self.constant_memory.read(addr, size)

    def launch_kernel(
        self,
        kernel_func: Callable[..., Any],
        grid_dim: Tuple[int, ...],
        block_dim: Tuple[int, ...],
        *args: Any,
        cooperative: bool = False,
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

        grid_barrier: Barrier | None = None
        if cooperative:
            total_threads = gx * gy * gz * bx * by * bz
            grid_barrier = Barrier(parties=total_threads)

        start = self.current_cycle()
        self.kernel_log.append(
            KernelLaunchEvent(
                name=getattr(kernel_func, "__name__", str(kernel_func)),
                grid_dim=(gx, gy, gz),
                block_dim=(bx, by, bz),
                start_cycle=start,
            )
        )
        self._cycle_counter += 1

        for z in range(gz):
            for y in range(gy):
                for x in range(gx):
                    block_idx = (x, y, z)
                    tb = ThreadBlock(
                        block_idx=block_idx,
                        block_dim=(bx, by, bz),
                        grid_dim=(gx, gy, gz),
                        shared_mem_size=self.shared_mem_size,
                        barrier_timeout=self.barrier_timeout,
                    )
                    tb.kernel_func = kernel_func
                    tb.kernel_args = args
                    tb.initialize_threads(kernel_func, *args)
                    for t in tb.threads:
                        setattr(t, "global_mem", self.global_memory)
                        setattr(t, "const_mem", self.constant_memory)
                        setattr(t, "constant_mem", self.constant_memory)
                        if grid_barrier is not None:
                            setattr(t, "grid_barrier", grid_barrier)
                            setattr(t, "grid_barrier_timeout", self.barrier_timeout)
                    self._launched_blocks.append(tb)

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

        end = self.current_cycle()
        for ev in self.kernel_log:
            if getattr(ev, "cycles", 0) == 0:
                ev.cycles = end - ev.start_cycle

    def get_memory_stats(self) -> dict[str, int]:
        """Aggregate spill statistics from all launched threads."""

        totals = {"spill_events": 0, "spill_bytes": 0, "spill_cycles": 0}
        for tb in self._launched_blocks:
            for t in tb.threads:
                stats = getattr(t, "get_spill_stats", lambda: {})()
                totals["spill_events"] += stats.get("spill_events", 0)
                totals["spill_bytes"] += stats.get("spill_bytes", 0)
                totals["spill_cycles"] += stats.get("spill_cycles", 0)
        return totals

    def get_transfer_stats(self) -> dict[str, int]:
        """Return aggregate statistics for host/device transfers."""

        return {
            "transfers": self.counters["transfers"],
            "transfer_bytes": self.stats["transfer_bytes"],
            "transfer_cycles": self.stats["transfer_cycles"],
        }

    def get_transfer_log(self) -> List[TransferEvent]:
        """Return a copy of the log with all transfer events."""

        return list(self.transfer_log)

    def get_kernel_log(self) -> List[KernelLaunchEvent]:
        """Return a copy of the log with all kernel launch events."""

        return list(self.kernel_log)
