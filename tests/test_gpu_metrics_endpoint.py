import os
import sys
import queue
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.api import app
from py_virtual_gpu.services import get_gpu_manager
from py_virtual_gpu.virtualgpu import VirtualGPU


def _setup_gpu():
    manager = get_gpu_manager()
    manager._gpus.clear()
    gpu = VirtualGPU(num_sms=1, global_mem_size=64)
    for sm in gpu.sms:
        sm.block_queue = queue.Queue()
    manager.add_gpu(gpu)
    return gpu


def test_metrics_endpoint_transfer_bytes():
    gpu = _setup_gpu()
    ptr = gpu.malloc(16)
    gpu.memcpy_host_to_device(b"a" * 16, ptr)

    with TestClient(app) as client:
        resp = client.get("/gpus/0/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["transfer_stats"]["transfer_bytes"] == 16
