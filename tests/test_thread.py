import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.thread import Thread, RegisterMemory
from py_virtual_gpu.shared_memory import SharedMemory
from py_virtual_gpu.global_memory import GlobalMemory


def test_register_memory_read_write_clear():
    rm = RegisterMemory(1024)
    rm.write("a", 123)
    assert rm.read("a") == 123
    rm.clear()
    assert rm.read("a") is None


def test_thread_attributes():
    sm = SharedMemory(16)
    gm = GlobalMemory(32)
    t = Thread((0, 0, 0), (1, 2, 3), (4, 4, 1), (8, 8, 1), 256, sm, gm)
    assert t.thread_idx == (0, 0, 0)
    assert t.block_idx == (1, 2, 3)
    assert t.block_dim == (4, 4, 1)
    assert t.grid_dim == (8, 8, 1)
    assert isinstance(t.registers, RegisterMemory)
    assert t.registers.size == 256


def test_run_passes_indices_and_args():
    sm = SharedMemory(1)
    gm = GlobalMemory(1)
    t = Thread((1, 0, 0), (0, 0, 0), (2, 1, 1), (1, 1, 1), 4, sm, gm)

    received = {}

    def k(threadIdx, blockIdx, blockDim, gridDim, a, b):
        received["vals"] = (threadIdx, blockIdx, blockDim, gridDim, a, b)
        threadIdx = (9, 9, 9)  # should not affect stored attribute

    t.run(
        k,
        t.thread_idx,
        t.block_idx,
        t.block_dim,
        t.grid_dim,
        7,
        8,
    )

    assert received["vals"] == (
        (1, 0, 0),
        (0, 0, 0),
        (2, 1, 1),
        (1, 1, 1),
        7,
        8,
    )
    assert t.threadIdx == (1, 0, 0)
    assert t.blockIdx == (0, 0, 0)


def test_repr_contains_indices_and_register_count():
    sm = SharedMemory(1)
    gm = GlobalMemory(1)
    t = Thread((0, 0, 0), (0, 0, 0), (1, 1, 1), (1, 1, 1), 0, sm, gm)
    text = repr(t)
    assert "idx=(0,0,0)" in text
    assert "blk=(0,0,0)" in text
    assert "regs=0" in text

