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

        from ..api.schemas import GPUState, SMState, GlobalMemState, TransferRecord

        gpu = self.get_gpu(id)

        gm = gpu.global_memory
        gm_state = GlobalMemState(size=gm.size, used=sum(gm.allocations.values()))

        transfer_log = [TransferRecord(**asdict(ev)) for ev in gpu.get_transfer_log()]

        sms = []
        for sm in gpu.sms:
            status = "idle" if sm.block_queue.empty() and sm.warp_queue.empty() else "busy"
            sms.append(SMState(id=sm.id, status=status, counters=sm.counters.copy()))

        return GPUState(id=id, global_memory=gm_state, transfer_log=transfer_log, sms=sms)


def get_gpu_manager() -> GPUManager:
    """FastAPI dependency returning the singleton :class:`GPUManager`."""

    return GPUManager()
