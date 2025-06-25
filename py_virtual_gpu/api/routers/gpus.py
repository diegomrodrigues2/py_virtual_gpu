from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ...services import GPUManager, get_gpu_manager
from ..schemas import GPUSummary, GPUState, GPUMetrics, SMDetailed

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


@router.get("/gpus/{id}/metrics", response_model=GPUMetrics)
def gpu_metrics(id: int, manager: GPUManager = Depends(get_gpu_manager)) -> GPUMetrics:
    """Return aggregated metrics for the GPU with ``id``."""

    return manager.get_gpu_metrics(id)


@router.get("/gpus/{id}/sm/{sm_id}", response_model=SMDetailed)
def sm_detail(
    id: int,
    sm_id: int,
    max_events: int = Query(100, ge=1),
    manager: GPUManager = Depends(get_gpu_manager),
) -> SMDetailed:
    """Return detailed state for ``sm_id`` of GPU ``id``."""

    try:
        return manager.get_sm_detail(id, sm_id, max_events)
    except IndexError:
        raise HTTPException(status_code=404, detail="SM not found")
