import os
import sys
import math
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.thread import Thread


def _kernel_alloc_write(threadIdx, blockIdx, blockDim, gridDim, th, data):
    off = th.alloc_local(len(data))
    th.local_mem.write(off, data)
    return off


def test_kernel_alloc_persistence_and_latency():
    t = Thread(register_mem_size=0, local_mem_size=16)
    data = b"abcd"
    off = t.run(
        _kernel_alloc_write,
        t.thread_idx,
        t.block_idx,
        t.block_dim,
        t.grid_dim,
        t,
        data,
    )
    stats = dict(t.local_mem.stats)
    assert off == 0
    assert t.local_ptr == len(data)
    assert t.local_mem.read(off, len(data)) == data
    expected_cycles = t.local_mem.latency_cycles + math.ceil(len(data) / t.local_mem.bandwidth_bpc)
    assert stats == {"reads": 0, "writes": 1, "cycles": expected_cycles}


def _kernel_overflow(threadIdx, blockIdx, blockDim, gridDim, th):
    th.alloc_local(4)
    th.alloc_local(4)
    th.alloc_local(1)


def test_alloc_local_overflow_inside_kernel():
    t = Thread(register_mem_size=0, local_mem_size=8)
    with pytest.raises(IndexError):
        t.run(
            _kernel_overflow,
            t.thread_idx,
            t.block_idx,
            t.block_dim,
            t.grid_dim,
            t,
        )
