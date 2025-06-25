import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.thread import Thread
from py_virtual_gpu.memory_hierarchy import RegisterFile
from py_virtual_gpu.shared_memory import SharedMemory
from py_virtual_gpu.global_memory import GlobalMemory


def test_register_file_read_write():
    rf = RegisterFile(16)
    rf.write(0, b"abcd")
    assert rf.read(0, 4) == b"abcd"


def test_thread_attributes():
    sm = SharedMemory(16)
    gm = GlobalMemory(32)
    t = Thread(
        (0, 0, 0),
        (1, 2, 3),
        (4, 4, 1),
        (8, 8, 1),
        256,
        shared_mem=sm,
        global_mem=gm,
    )
    assert t.thread_idx == (0, 0, 0)
    assert t.block_idx == (1, 2, 3)
    assert t.block_dim == (4, 4, 1)
    assert t.grid_dim == (8, 8, 1)
    assert isinstance(t.registers, RegisterFile)
    assert t.registers.size == 256


def test_run_passes_indices_and_args():
    sm = SharedMemory(1)
    gm = GlobalMemory(1)
    t = Thread(
        (1, 0, 0),
        (0, 0, 0),
        (2, 1, 1),
        (1, 1, 1),
        4,
        shared_mem=sm,
        global_mem=gm,
    )

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
    t = Thread(
        (0, 0, 0),
        (0, 0, 0),
        (1, 1, 1),
        (1, 1, 1),
        0,
        shared_mem=sm,
        global_mem=gm,
    )
    text = repr(t)
    assert "idx=(0,0,0)" in text
    assert "blk=(0,0,0)" in text
    assert "regs=0" in text


def test_alloc_local_and_latency():
    gm = GlobalMemory(16)
    t = Thread(register_mem_size=4, local_mem_size=8)
    off1 = t.alloc_local(4)
    off2 = t.alloc_local(2)
    assert (off1, off2) == (0, 4)
    assert t.local_ptr == 6
    assert t.local_mem.latency_cycles == gm.latency_cycles
    assert t.local_mem.bandwidth_bpc == gm.bandwidth_bpc


def test_alloc_local_overflow():
    t = Thread(register_mem_size=4, local_mem_size=4)
    t.alloc_local(4)
    with pytest.raises(IndexError):
        t.alloc_local(1)

