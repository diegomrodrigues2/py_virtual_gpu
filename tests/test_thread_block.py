import os
import sys

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


def test_execute_runs_without_error():
    tb = ThreadBlock((0, 0, 0), (1, 1, 1), (1, 1, 1), shared_mem_size=1)
    tb.execute(lambda *a: None)
    assert len(tb.threads) == 1


def test_repr_contains_info():
    tb = ThreadBlock((1, 2, 3), (1, 1, 1), (1, 1, 1), shared_mem_size=1)
    text = repr(tb)
    assert "idx=(1, 2, 3)" in text
    assert "threads=0" in text or "threads=1" in text
    assert "block_dim=(1, 1, 1)" in text
