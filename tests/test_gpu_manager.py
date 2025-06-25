import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.api import app
from py_virtual_gpu.services import GPUManager, get_gpu_manager
from py_virtual_gpu.virtualgpu import VirtualGPU


def test_startup_registers_gpu():
    with TestClient(app):  # triggers startup event
        manager = get_gpu_manager()
        gpus = manager.list_gpus()
        assert len(gpus) >= 1
        assert isinstance(manager.get_gpu(0), VirtualGPU)


def test_gpu_manager_add_and_get():
    manager = GPUManager()
    start = len(manager.list_gpus())
    gpu = VirtualGPU(num_sms=2, global_mem_size=512)
    idx = manager.add_gpu(gpu)
    assert idx == start
    assert manager.get_gpu(idx) is gpu
    assert len(manager.list_gpus()) == start + 1


def test_get_gpu_invalid_id():
    manager = GPUManager()
    with pytest.raises(IndexError):
        manager.get_gpu(-1)
    with pytest.raises(IndexError):
        manager.get_gpu(999)
