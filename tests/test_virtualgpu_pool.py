import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.virtualgpu import VirtualGPU, _execute_block_worker
from py_virtual_gpu.thread_block import ThreadBlock


class DummyPool:
    def __init__(self):
        self.apply_async_calls = []
        self.calls = []

    def apply_async(self, func, args=()):
        self.apply_async_calls.append((func, args))

    def close(self):
        self.calls.append("close")

    def join(self):
        self.calls.append("join")


class DummySM:
    def __init__(self):
        import queue
        self.block_queue = queue.Queue()


def test_launch_kernel_uses_pool_apply_async(monkeypatch):
    pool = DummyPool()
    monkeypatch.setattr("py_virtual_gpu.virtualgpu.Pool", lambda processes: pool)
    gpu = VirtualGPU(num_sms=2, global_mem_size=32, use_pool=True)

    def dummy():
        pass

    gpu.launch_kernel(dummy, (3, 1, 1), (1, 1, 1))
    assert len(pool.apply_async_calls) == 3
    for func, args in pool.apply_async_calls:
        assert func is _execute_block_worker
        tb, fn, fn_args = args
        assert isinstance(tb, ThreadBlock)
        assert fn is dummy
        assert fn_args == ()


def test_synchronize_closes_and_joins_pool(monkeypatch):
    pool = DummyPool()
    monkeypatch.setattr("py_virtual_gpu.virtualgpu.Pool", lambda processes: pool)
    gpu = VirtualGPU(num_sms=1, global_mem_size=16, use_pool=True, sync_on_launch=True)

    def dummy():
        pass

    gpu.launch_kernel(dummy, (1, 1, 1), (1, 1, 1))
    assert pool.calls == ["close", "join"]

