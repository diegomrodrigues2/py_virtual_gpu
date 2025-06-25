from __future__ import annotations

from typing import List
from dataclasses import asdict

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

        return GPUState(
            id=id,
            name=f"GPU {id}",
            config=config,
            global_memory=gm_state,
            transfer_log=transfer_log,
            sms=sms,
            overall_load=overall_load,
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

        from ..api.schemas import SMDetailed, BlockSummary, WarpSummary, DivergenceRecord

        gpu = self.get_gpu(gpu_id)
        if sm_id < 0 or sm_id >= len(gpu.sms):
            raise IndexError("Invalid SM id")
        sm = gpu.sms[sm_id]

        try:
            pending_blocks = list(sm.block_queue.queue)
        except AttributeError:  # multiprocessing.Queue may not expose 'queue'
            pending_blocks = []

        blocks: list[BlockSummary] = []
        for tb in pending_blocks:
            blocks.append(BlockSummary(block_idx=tb.block_idx, status="pending"))

        try:
            queued_warps = list(sm.warp_queue.queue)
        except AttributeError:
            queued_warps = []

        warps: list[WarpSummary] = []
        for warp in queued_warps:
            warps.append(WarpSummary(id=warp.id, active_threads=sum(warp.active_mask)))

        log = [DivergenceRecord(**asdict(ev)) for ev in sm.divergence_log[-max_events:]]

        return SMDetailed(
            id=sm.id,
            blocks=blocks,
            warps=warps,
            divergence_log=log,
            counters=sm.counters.copy(),
        )

    def get_event_feed(self, since_cycle: int | None = None, limit: int = 100):
        """Return a global event feed aggregated from all GPUs."""

        events = []
        for gpu_id, gpu in enumerate(self._gpus):
            for ev in gpu.get_kernel_log():
                if since_cycle is None or ev.start_cycle >= since_cycle:
                    item = asdict(ev)
                    item["gpu_id"] = gpu_id
                    item["type"] = "kernel"
                    events.append(item)
            for ev in gpu.get_transfer_log():
                if since_cycle is None or ev.start_cycle >= since_cycle:
                    item = asdict(ev)
                    item["gpu_id"] = gpu_id
                    item["type"] = "transfer"
                    events.append(item)
            for sm in gpu.sms:
                for ev in sm.get_divergence_log():
                    if since_cycle is None or ev.start_cycle >= since_cycle:
                        item = asdict(ev)
                        item["gpu_id"] = gpu_id
                        item["type"] = "divergence"
                        events.append(item)

        events.sort(key=lambda e: e.get("start_cycle", 0))
        return events[:limit]


def get_gpu_manager() -> GPUManager:
    """FastAPI dependency returning the singleton :class:`GPUManager`."""

    return GPUManager()
