from __future__ import annotations

from fastapi import FastAPI

from ..virtualgpu import VirtualGPU

app = FastAPI(title="Py Virtual GPU API")

gpu = VirtualGPU(num_sms=1, global_mem_size=1024)


@app.get("/status")
def read_status() -> dict[str, int]:
    """Return basic information about the simulated GPU."""

    return {
        "num_sms": len(gpu.sms),
        "global_mem_size": gpu.global_memory.size,
        "shared_mem_size": gpu.shared_mem_size,
    }
