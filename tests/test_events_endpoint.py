import os
import sys
import queue
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.api import app
from py_virtual_gpu.services import get_gpu_manager
from py_virtual_gpu.virtualgpu import VirtualGPU
from py_virtual_gpu.warp import Warp
from py_virtual_gpu.thread import Thread


def _setup_gpu():
    manager = get_gpu_manager()
    manager._gpus.clear()
    gpu = VirtualGPU(num_sms=1, global_mem_size=64)
    for sm in gpu.sms:
        sm.block_queue = queue.Queue()
    manager.add_gpu(gpu)
    return gpu


def test_events_endpoint_returns_all():
    gpu = _setup_gpu()
    ptr = gpu.malloc(8)
    gpu.memcpy_host_to_device(b"a" * 8, ptr)
    gpu.memcpy_device_to_host(ptr, 8)
    sm = gpu.sms[0]
    sm.record_divergence(Warp(0, [Thread(), Thread()], sm), 0, [True, True], [True, False])

    def dummy():
        pass

    gpu.launch_kernel(dummy, (1, 1, 1), (1, 1, 1))
    gpu.synchronize()

    with TestClient(app) as client:
        resp = client.get("/events")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 6
        assert data == sorted(data, key=lambda e: e["start_cycle"])
        types = {ev["type"] for ev in data}
        assert {"kernel", "transfer", "divergence", "BLOCK_START", "BLOCK_END"} <= types
