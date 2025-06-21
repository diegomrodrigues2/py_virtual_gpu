import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.thread_block import ThreadBlock


def test_block_initialization_attributes():
    tb = ThreadBlock((0, 0, 0), (4, 4, 1), (16, 16, 1), shared_mem_size=256)
    assert tb.block_idx == (0, 0, 0)
    assert tb.block_dim == (4, 4, 1)
    assert tb.grid_dim == (16, 16, 1)
    assert tb.shared_mem.size == 256
    assert tb.barrier.parties == 16


def test_initialize_threads_creates_all():
    tb = ThreadBlock((0, 0, 0), (2, 2, 1), (8, 8, 1), shared_mem_size=64)
    tb.initialize_threads(lambda *a: None)
    assert len(tb.threads) == 4


def test_execute_invokes_kernel_for_each_thread():
    tb = ThreadBlock((0, 0, 0), (2, 1, 1), (1, 1, 1), shared_mem_size=1)
    called = []

    def kernel(tidx, bidx, bdim, gdim, value):
        called.append((tidx, bidx, value))

    tb.execute(kernel, 42)

    assert len(tb.threads) == 2
    assert len(called) == 2
    expected = [((0, 0, 0), (0, 0, 0), 42), ((1, 0, 0), (0, 0, 0), 42)]
    assert called == expected


def test_repr_contains_info():
    tb = ThreadBlock((1, 2, 3), (1, 1, 1), (1, 1, 1), shared_mem_size=1)
    text = repr(tb)
    assert "idx=(1, 2, 3)" in text
    assert "threads=0" in text or "threads=1" in text
    assert "block_dim=(1, 1, 1)" in text
