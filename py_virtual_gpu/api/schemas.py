from __future__ import annotations

from pydantic import BaseModel


class GPUSummary(BaseModel):
    """Basic information about a simulated GPU."""

    id: int
    num_sms: int
    global_mem_size: int
    shared_mem_size: int
