from __future__ import annotations

from fastapi import Depends, FastAPI

from ..virtualgpu import VirtualGPU
from ..services import GPUManager, get_gpu_manager
from .routers import gpus as gpus_router

app = FastAPI(title="Py Virtual GPU API")
app.include_router(gpus_router.router)


@app.on_event("startup")
def startup_event() -> None:
    """Register a default GPU instance on application startup."""

    manager = get_gpu_manager()
    if not manager.list_gpus():
        manager.add_gpu(VirtualGPU(num_sms=1, global_mem_size=1024))


@app.get("/status")
def read_status(
    manager: GPUManager = Depends(get_gpu_manager),
) -> dict[str, int]:
    """Return basic information about the simulated GPU."""

    gpu = manager.get_gpu(0)
    return {
        "num_sms": len(gpu.sms),
        "global_mem_size": gpu.global_memory.size,
        "shared_mem_size": gpu.shared_mem_size,
    }


@app.get("/events")
def get_events(
    since_cycle: int | None = None,
    limit: int = 100,
    manager: GPUManager = Depends(get_gpu_manager),
) -> list[dict]:
    """Return a consolidated, time-ordered event feed."""

    return manager.get_event_feed(since_cycle, limit)
