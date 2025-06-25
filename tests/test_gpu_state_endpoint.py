import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.api import app
import queue

from py_virtual_gpu.services import get_gpu_manager
from py_virtual_gpu.virtualgpu import VirtualGPU
from py_virtual_gpu.warp import Warp


def _setup_gpu(num_sms=1):
    manager = get_gpu_manager()
    manager._gpus.clear()
    gpu = VirtualGPU(num_sms=num_sms, global_mem_size=64)
    for sm in gpu.sms:
        sm.block_queue = queue.Queue()
    manager.add_gpu(gpu)
    return gpu


def test_state_endpoint_idle():
    _setup_gpu(num_sms=2)
    with TestClient(app) as client:
        resp = client.get("/gpus/0/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["global_memory"]["used"] == 0
        assert len(data["sms"]) == 2
        for sm in data["sms"]:
            assert sm["status"] == "idle"
            assert sm["counters"]["warps_executed"] == 0


@pytest.mark.parametrize("policy", ["sequential", "round_robin"])
def test_state_endpoint_reports_warps(policy):
    gpu = _setup_gpu()
    gpu.sms[0].schedule_policy = policy

    def dummy():
        pass

    mp = pytest.MonkeyPatch()

    def _nop(self):
        self.active_mask = [False] * len(self.active_mask)

    mp.setattr(Warp, "execute", _nop)

    gpu.launch_kernel(dummy, (1, 1, 1), (64, 1, 1))
    gpu.synchronize()
    mp.undo()

    with TestClient(app) as client:
        resp = client.get("/gpus/0/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["sms"][0]["counters"]["warps_executed"] == 2


def test_state_endpoint_includes_config_and_load():
    _setup_gpu(num_sms=1)
    with TestClient(app) as client:
        resp = client.get("/gpus/0/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "GPU 0"
        assert data["config"]["num_sms"] == 1
        assert data["config"]["global_mem_size"] == 64
        assert "overall_load" in data


