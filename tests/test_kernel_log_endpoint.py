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


def test_kernel_log_endpoint():
    gpu = _setup_gpu()

    def dummy():
        pass

    gpu.launch_kernel(dummy, (1, 1, 1), (2, 2, 1))

    with TestClient(app) as client:
        resp = client.get("/gpus/0/kernel_log")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "dummy"
        assert data[0]["grid_dim"] == [1, 1, 1]
        assert data[0]["block_dim"] == [2, 2, 1]
