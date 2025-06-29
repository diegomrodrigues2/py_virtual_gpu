import os
import sys
import queue
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.api import app
from py_virtual_gpu.services import get_gpu_manager
from py_virtual_gpu.virtualgpu import VirtualGPU
from py_virtual_gpu.types import Float32


def _setup_gpu():
    manager = get_gpu_manager()
    manager._gpus.clear()
    gpu = VirtualGPU(num_sms=1, global_mem_size=64)
    for sm in gpu.sms:
        sm.block_queue = queue.Queue()
    manager.add_gpu(gpu)
    return gpu


def test_allocations_endpoint_and_state():
    gpu = _setup_gpu()
    ptr = gpu.malloc(4, dtype=Float32, label="buf")
    with TestClient(app) as client:
        resp = client.get("/gpus/0/allocations")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        alloc = data[0]
        assert alloc["offset"] == ptr.offset
        assert alloc["size"] == 4
        assert alloc["dtype"] == "Float32"
        assert alloc["label"] == "buf"

        state = client.get("/gpus/0/state").json()
        assert state["allocations"][0]["offset"] == ptr.offset

    gpu.free(ptr)
    with TestClient(app) as client:
        resp = client.get("/gpus/0/allocations")
        assert resp.status_code == 200
        assert resp.json() == []
    manager = get_gpu_manager()
    manager._gpus.clear()
