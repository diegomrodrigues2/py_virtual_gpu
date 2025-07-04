from __future__ import annotations

from typing import List
from dataclasses import asdict
from uuid import uuid4

from ..virtualgpu import VirtualGPU


class GPUManager:
    """Singleton manager for :class:`VirtualGPU` instances."""

    _instance: "GPUManager" | None = None

    def __new__(cls) -> "GPUManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._gpus: List[VirtualGPU] = []
        return cls._instance

    # simple init that does nothing to avoid reinitializing on repeated calls
    def __init__(self) -> None:  # pragma: no cover - __new__ handles singleton init
        pass

    def add_gpu(self, gpu: VirtualGPU) -> int:
        """Register ``gpu`` and return its id."""

        self._gpus.append(gpu)
        return len(self._gpus) - 1

    def list_gpus(self) -> List[VirtualGPU]:
        """Return a list of registered GPUs."""

        return list(self._gpus)

    def get_gpu(self, id: int) -> VirtualGPU:
        """Return the GPU with ``id`` or raise ``IndexError`` if not found."""

        if id < 0 or id >= len(self._gpus):
            raise IndexError("Invalid GPU id")
        return self._gpus[id]

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def get_gpu_state(self, id: int):
        """Return a detailed snapshot of the GPU with ``id``."""

        from ..api.schemas import (
            GPUState,
            SMState,
            GlobalMemState,
            TransferRecord,
            GPUConfig,
            AllocationRecord,
        )

        gpu = self.get_gpu(id)

        gm = gpu.global_memory
        gm_state = GlobalMemState(size=gm.size, used=sum(gm.allocations.values()))

        transfer_log = [TransferRecord(**asdict(ev)) for ev in gpu.get_transfer_log()]

        sms = []
        for sm in gpu.sms:
            status = "idle" if sm.block_queue.empty() and sm.warp_queue.empty() else "busy"
            sms.append(SMState(id=sm.id, status=status, counters=sm.counters.copy()))

        active_sms = sum(1 for sm in sms if sm.status != "idle")
        overall_load = int((active_sms / len(sms)) * 100) if sms else 0

        config = GPUConfig(
            num_sms=len(gpu.sms),
            global_mem_size=gpu.global_memory.size,
            shared_mem_size=gpu.shared_mem_size,
            registers_per_sm_total=getattr(gpu.sms[0], "max_registers_per_thread", 0)
            * getattr(gpu.sms[0], "warp_size", 32)
            if gpu.sms
            else 0,
        )

        allocations = [
            AllocationRecord(
                offset=off,
                size=meta[1],
                dtype=meta[2].__name__ if meta[2] is not None else None,
                label=meta[3],
            )
            for off, meta in gpu.alloc_metadata.items()
        ]

        return GPUState(
            id=id,
            name=f"GPU {id}",
            config=config,
            global_memory=gm_state,
            transfer_log=transfer_log,
            sms=sms,
            overall_load=overall_load,
            allocations=allocations,
        )

    def get_gpu_metrics(self, id: int):
        """Return aggregated metrics for the GPU with ``id``."""

        from ..api.schemas import GPUMetrics

        gpu = self.get_gpu(id)

        mem_stats = gpu.get_memory_stats()
        transfer_stats = gpu.get_transfer_stats()

        instructions = 0
        global_accesses = 0
        shared_accesses = 0
        divergences = 0
        for sm in gpu.sms:
            instructions += sm.counters.get("warps_executed", 0)
            global_accesses += sm.counters.get("non_coalesced_accesses", 0)
            shared_accesses += sm.counters.get("bank_conflicts", 0)
            divergences += sm.counters.get("warp_divergences", 0)

        return GPUMetrics(
            id=id,
            instructions=instructions,
            global_accesses=global_accesses,
            shared_accesses=shared_accesses,
            divergences=divergences,
            memory_stats=mem_stats,
            transfer_stats=transfer_stats,
        )

    def get_sm_detail(self, gpu_id: int, sm_id: int, max_events: int = 100):
        """Return detailed information for ``sm_id`` of GPU ``gpu_id``."""

        from ..api.schemas import (
            SMDetailed,
            BlockSummary,
            WarpSummary,
            DivergenceRecord,
            BlockEventRecord,
        )

        gpu = self.get_gpu(gpu_id)
        if sm_id < 0 or sm_id >= len(gpu.sms):
            raise IndexError("Invalid SM id")
        sm = gpu.sms[sm_id]

        # ``StreamingMultiprocessor.block_queue`` is a ``queue.Queue`` and
        # exposes the underlying ``queue`` attribute for inspection.
        pending_blocks = list(sm.block_queue.queue)

        blocks: list[BlockSummary] = []
        for tb in pending_blocks:
            blocks.append(BlockSummary(block_idx=tb.block_idx, status="pending"))

        queued_warps = list(sm.warp_queue.queue)

        warps: list[WarpSummary] = []
        for warp in queued_warps:
            warps.append(WarpSummary(id=warp.id, active_threads=sum(warp.active_mask)))

        log = [DivergenceRecord(**asdict(ev)) for ev in sm.divergence_log[-max_events:]]
        block_log = [BlockEventRecord(**asdict(ev)) for ev in sm.block_log[-max_events:]]

        return SMDetailed(
            id=sm.id,
            blocks=blocks,
            warps=warps,
            divergence_log=log,
            counters=sm.counters.copy(),
            block_event_log=block_log,
        )

    def get_global_memory_slice(self, id: int, offset: int, size: int) -> bytes:
        """Return ``size`` bytes from global memory of GPU ``id`` starting at ``offset``."""

        gpu = self.get_gpu(id)
        return gpu.global_memory.read(offset, size)

    def get_constant_memory_slice(self, id: int, offset: int, size: int) -> bytes:
        """Return ``size`` bytes from constant memory of GPU ``id`` starting at ``offset``."""

        gpu = self.get_gpu(id)
        return gpu.constant_memory.read(offset, size)

    def get_kernel_log(self, id: int):
        """Return a list of kernel launch records for GPU ``id``."""

        from ..api.schemas import KernelLaunchRecord

        gpu = self.get_gpu(id)
        return [KernelLaunchRecord(**asdict(ev)) for ev in gpu.get_kernel_log()]

    def get_gpu_allocations(self, id: int):
        """Return active allocations for GPU ``id``."""

        from ..api.schemas import AllocationRecord

        gpu = self.get_gpu(id)
        return [
            AllocationRecord(
                offset=off,
                size=meta[1],
                dtype=meta[2].__name__ if meta[2] is not None else None,
                label=meta[3],
            )
            for off, meta in gpu.alloc_metadata.items()
        ]

    def get_event_feed(self, since_cycle: int | None = None, limit: int = 100):
        """Return a global event feed aggregated from all GPUs."""

        events = []
        for gpu_id, gpu in enumerate(self._gpus):
            for ev in gpu.get_kernel_log():
                if since_cycle is None or ev.start_cycle >= since_cycle:
                    item = asdict(ev)
                    item["gpu_id"] = gpu_id
                    item["type"] = "kernel"
                    item["id"] = str(uuid4())
                    item["timestamp"] = ev.timestamp
                    item["message"] = (
                        f"Kernel '{ev.name}' launched grid {ev.grid_dim} block {ev.block_dim}"
                    )
                    events.append(item)
            for ev in gpu.get_transfer_log():
                if since_cycle is None or ev.start_cycle >= since_cycle:
                    item = asdict(ev)
                    item["gpu_id"] = gpu_id
                    item["type"] = "transfer"
                    item["id"] = str(uuid4())
                    item["timestamp"] = ev.timestamp
                    item["message"] = f"{ev.direction} transfer of {ev.size} bytes"
                    events.append(item)
            for sm in gpu.sms:
                for ev in sm.get_divergence_log():
                    if since_cycle is None or ev.start_cycle >= since_cycle:
                        item = asdict(ev)
                        item["gpu_id"] = gpu_id
                        item["type"] = "divergence"
                        item["id"] = str(uuid4())
                        item["timestamp"] = ev.timestamp
                        item["message"] = (
                            f"Warp {ev.warp_id} divergence at PC {ev.pc}"
                        )
                        events.append(item)
                for ev in sm.get_block_event_log():
                    if since_cycle is None or ev.start_cycle >= since_cycle:
                        item = asdict(ev)
                        item["gpu_id"] = gpu_id
                        item["type"] = f"BLOCK_{ev.phase.upper()}"
                        item["id"] = str(uuid4())
                        item["timestamp"] = ev.timestamp
                        item["message"] = (
                            f"Block {ev.block_idx} {ev.phase} on SM {ev.sm_id}"
                        )
                        events.append(item)

        events.sort(key=lambda e: e.get("start_cycle", 0))
        return events[:limit]


def get_gpu_manager() -> GPUManager:
    """FastAPI dependency returning the singleton :class:`GPUManager`."""

    return GPUManager()
