from __future__ import annotations

from fastapi import APIRouter, Depends

from ...services import GPUManager, get_gpu_manager
from ..schemas import GPUSummary, GPUState

router = APIRouter()


@router.get("/gpus", response_model=list[GPUSummary])
def list_gpus(manager: GPUManager = Depends(get_gpu_manager)) -> list[GPUSummary]:
    """Return a summary of all registered GPUs."""
    summaries = []
    for idx, gpu in enumerate(manager.list_gpus()):
        summaries.append(
            GPUSummary(
                id=idx,
                num_sms=len(gpu.sms),
                global_mem_size=gpu.global_memory.size,
                shared_mem_size=gpu.shared_mem_size,
            )
        )
    return summaries


@router.get("/gpus/{id}/state", response_model=GPUState)
def gpu_state(id: int, manager: GPUManager = Depends(get_gpu_manager)) -> GPUState:
    """Return a detailed snapshot for the GPU with ``id``."""

    return manager.get_gpu_state(id)
