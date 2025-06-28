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


def _setup_gpu(num_sms=1):
    manager = get_gpu_manager()
    manager._gpus.clear()
    gpu = VirtualGPU(num_sms=num_sms, global_mem_size=64)
    for sm in gpu.sms:
        sm.block_queue = queue.Queue()
    manager.add_gpu(gpu)
    return gpu


def test_sm_detail_not_found():
    _setup_gpu()
    with TestClient(app) as client:
        resp = client.get("/gpus/0/sm/99")
        assert resp.status_code == 404


def test_sm_detail_divergence_log_size():
    gpu = _setup_gpu()
    sm = gpu.sms[0]
    sm.record_divergence(Warp(0, [Thread(), Thread()], sm), 0, [True, True], [True, False])
    sm.record_divergence(Warp(0, [Thread(), Thread()], sm), 1, [True, False], [True, True])

    with TestClient(app) as client:
        resp = client.get("/gpus/0/sm/0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["counters"]["warp_divergences"] == len(data["divergence_log"])
        assert "block_event_log" in data
        assert "barrier_wait_ms" in data["counters"]
