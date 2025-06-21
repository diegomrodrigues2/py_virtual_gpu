import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.streaming_multiprocessor import StreamingMultiprocessor


class DummyThread:
    pass


class DummyBlock:
    def __init__(self, num_threads: int):
        self.threads = [DummyThread() for _ in range(num_threads)]


def test_sm_instantiation_and_queue():
    sm = StreamingMultiprocessor(id=0, shared_mem_size=256, max_registers_per_thread=16)
    assert sm.block_queue.empty()
    sm.block_queue.put(DummyBlock(1))
    assert sm.block_queue.qsize() == 1


def test_execute_block_warp_count():
    sm = StreamingMultiprocessor(id=0, shared_mem_size=128, max_registers_per_thread=16, warp_size=32)
    block = DummyBlock(65)
    sm.execute_block(block)
    assert sm.counters["warps_executed"] == 3


def test_reset_counters_and_queue():
    sm = StreamingMultiprocessor(id=0, shared_mem_size=128, max_registers_per_thread=16)
    sm.block_queue.put(DummyBlock(2))
    sm.counters["warps_executed"] = 5
    sm.counters["warp_divergences"] = 2
    sm.reset()
    assert sm.block_queue.empty()
    assert sm.counters["warps_executed"] == 0
    assert sm.counters["warp_divergences"] == 0


def test_repr_contains_info():
    sm = StreamingMultiprocessor(id=7, shared_mem_size=64, max_registers_per_thread=8)
    text = repr(sm)
    assert "id=7" in text
    assert "queue_size=0" in text
    assert "warps=0" in text
