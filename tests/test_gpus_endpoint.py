import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.api import app
from py_virtual_gpu.services import GPUManager, get_gpu_manager
from py_virtual_gpu.virtualgpu import VirtualGPU


def test_gpus_endpoint_multi_gpu():
    manager = get_gpu_manager()
    manager._gpus.clear()
    manager.add_gpu(VirtualGPU(num_sms=2, global_mem_size=256))
    manager.add_gpu(VirtualGPU(num_sms=4, global_mem_size=512))

    with TestClient(app) as client:
        resp = client.get("/gpus")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["id"] == 0
        assert data[0]["num_sms"] == 2
        assert data[0]["global_mem_size"] == 256
        assert data[1]["id"] == 1
        assert data[1]["num_sms"] == 4
        assert data[1]["global_mem_size"] == 512
