from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..virtualgpu import VirtualGPU
from ..services import GPUManager, get_gpu_manager
from .routers import gpus as gpus_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Register a default GPU instance when the application starts."""

    manager = get_gpu_manager()
    if not manager.list_gpus():
        manager.add_gpu(VirtualGPU(num_sms=1, global_mem_size=1024))
    yield


app = FastAPI(title="Py Virtual GPU API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(gpus_router.router)


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
