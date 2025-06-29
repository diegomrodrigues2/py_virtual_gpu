from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ...services import GPUManager, get_gpu_manager
from ..schemas import (
    GPUSummary,
    GPUState,
    GPUMetrics,
    SMDetailed,
    MemorySlice,
    KernelLaunchRecord,
    AllocationRecord,
)

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


@router.get("/gpus/{id}/global_mem", response_model=MemorySlice)
def global_mem_slice(
    id: int,
    offset: int = Query(..., ge=0),
    size: int = Query(..., ge=0),
    dtype: str | None = Query(None, pattern="^(half|float32|float64)$"),
    manager: GPUManager = Depends(get_gpu_manager),
) -> MemorySlice:
    """Return a slice of global memory for GPU ``id``."""

    try:
        data = manager.get_global_memory_slice(id, offset, size)
    except IndexError:
        raise HTTPException(status_code=404, detail="Invalid GPU id or bounds")

    values: list[float] | None = None
    if dtype:
        from ...types import Half, Float32, Float64
        import numpy as np

        type_map = {
            "half": Half,
            "float32": Float32,
            "float64": Float64,
        }
        cls = type_map[dtype]
        arr = np.frombuffer(data, dtype=cls.dtype)
        values = [float(cls(v)) for v in arr]

    return MemorySlice(
        offset=offset,
        size=len(data),
        dtype=dtype,
        data=data.hex(),
        values=values,
    )


@router.get("/gpus/{id}/constant_mem", response_model=MemorySlice)
def constant_mem_slice(
    id: int,
    offset: int = Query(..., ge=0),
    size: int = Query(..., ge=0),
    dtype: str | None = Query(None, pattern="^(half|float32|float64)$"),
    manager: GPUManager = Depends(get_gpu_manager),
) -> MemorySlice:
    """Return a slice of constant memory for GPU ``id``."""

    try:
        data = manager.get_constant_memory_slice(id, offset, size)
    except IndexError:
        raise HTTPException(status_code=404, detail="Invalid GPU id or bounds")

    values: list[float] | None = None
    if dtype:
        from ...types import Half, Float32, Float64
        import numpy as np

        type_map = {
            "half": Half,
            "float32": Float32,
            "float64": Float64,
        }
        cls = type_map[dtype]
        arr = np.frombuffer(data, dtype=cls.dtype)
        values = [float(cls(v)) for v in arr]

    return MemorySlice(
        offset=offset,
        size=len(data),
        dtype=dtype,
        data=data.hex(),
        values=values,
    )


@router.get("/gpus/{id}/kernel_log", response_model=list[KernelLaunchRecord])
def kernel_log(
    id: int, manager: GPUManager = Depends(get_gpu_manager)
) -> list[KernelLaunchRecord]:
    """Return the kernel launch log for GPU ``id``."""

    return manager.get_kernel_log(id)


@router.get("/gpus/{id}/allocations", response_model=list[AllocationRecord])
def gpu_allocations(
    id: int, manager: GPUManager = Depends(get_gpu_manager)
) -> list[AllocationRecord]:
    """Return active memory allocations for GPU ``id``."""

    return manager.get_gpu_allocations(id)
