import os
import sys
import queue
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.api import app
from py_virtual_gpu.services import get_gpu_manager
from py_virtual_gpu.virtualgpu import VirtualGPU
from py_virtual_gpu import Float32, Float64, Half
import numpy as np


def _setup_gpu():
    manager = get_gpu_manager()
    manager._gpus.clear()
    gpu = VirtualGPU(num_sms=1, global_mem_size=64)
    for sm in gpu.sms:
        sm.block_queue = queue.Queue()
    manager.add_gpu(gpu)
    return gpu


def test_global_memory_slice_endpoint():
    gpu = _setup_gpu()
    ptr = gpu.malloc(8)
    gpu.memcpy_host_to_device(b"abcdefgh", ptr)

    with TestClient(app) as client:
        resp = client.get(f"/gpus/0/global_mem?offset={ptr.offset}&size=8")
        assert resp.status_code == 200
        data = resp.json()
        assert bytes.fromhex(data["data"]) == b"abcdefgh"
        assert data["offset"] == ptr.offset
        assert data["size"] == 8
        assert data["dtype"] is None


def test_constant_memory_slice_endpoint():
    gpu = _setup_gpu()
    gpu.set_constant(b"xyz", 0)

    with TestClient(app) as client:
        resp = client.get("/gpus/0/constant_mem?offset=0&size=3")
        assert resp.status_code == 200
        data = resp.json()
        assert bytes.fromhex(data["data"]) == b"xyz"
        assert data["offset"] == 0
        assert data["size"] == 3
        assert data["dtype"] is None


def test_global_mem_slice_decoding_float32():
    gpu = _setup_gpu()
    ptr = gpu.malloc_type(2, Float32)
    gpu.memcpy_host_to_device(np.array([1.5, -2.0], dtype=np.float32).tobytes(), ptr)

    with TestClient(app) as client:
        resp = client.get(
            f"/gpus/0/global_mem?offset={ptr.offset}&size=8&dtype=float32"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["values"] == [1.5, -2.0]
        assert data["dtype"] == "float32"


def test_constant_mem_slice_decoding_half():
    gpu = _setup_gpu()
    arr = np.array([2.0, 3.0], dtype=np.float16)
    gpu.set_constant(arr.tobytes(), 0)

    with TestClient(app) as client:
        resp = client.get("/gpus/0/constant_mem?offset=0&size=4&dtype=half")
        assert resp.status_code == 200
        data = resp.json()
        assert data["values"] == [float(Half(v)) for v in arr]
        assert data["dtype"] == "half"
