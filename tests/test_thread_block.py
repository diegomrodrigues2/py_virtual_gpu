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
    assert sorted(called) == sorted(expected)


def test_repr_contains_info():
    tb = ThreadBlock((1, 2, 3), (1, 1, 1), (1, 1, 1), shared_mem_size=1)
    text = repr(tb)
    assert "idx=(1, 2, 3)" in text
    assert "threads=0" in text or "threads=1" in text
    assert "block_dim=(1, 1, 1)" in text


def test_initialize_threads_sets_same_barrier_instance():
    tb = ThreadBlock((0, 0, 0), (2, 1, 1), (1, 1, 1), shared_mem_size=1)
    tb.initialize_threads(lambda *a: None)

    barrier_ids = {id(t.barrier) for t in tb.threads}
    assert len(barrier_ids) == 1
    assert tb.barrier is tb.threads[0].barrier


class DummyBarrier:
    def __init__(self) -> None:
        self.calls = 0

    def wait(self) -> None:  # pragma: no cover - simple counter
        self.calls += 1


@pytest.mark.parametrize("block_dim", [(3, 1, 1), (2, 2, 1), (2, 2, 2)])
def test_barrier_sync_invokes_wait(block_dim):
    tb = ThreadBlock((0, 0, 0), block_dim, (1, 1, 1), shared_mem_size=0)
    dummy = DummyBarrier()
    tb.barrier = dummy
    tb.initialize_threads(lambda *a: None)

    tb.barrier_sync()
    assert dummy.calls == 1


def test_barrier_wait_orders_entries():
    block_dim = (4, 1, 1)
    tb = ThreadBlock((0, 0, 0), block_dim, (1, 1, 1), shared_mem_size=0)
    records = []

    def kernel(tidx, bidx, bdim, gdim, barrier, log):
        log.append(("before", tidx))
        barrier.wait()
        log.append(("after", tidx))

    tb.execute(kernel, tb.barrier, records)

    total_threads = block_dim[0] * block_dim[1] * block_dim[2]
    assert len(records) == 2 * total_threads

    before_indices = [i for i, r in enumerate(records) if r[0] == "before"]
    after_indices = [i for i, r in enumerate(records) if r[0] == "after"]

    assert len(before_indices) == total_threads
    assert len(after_indices) == total_threads
    assert min(after_indices) > max(before_indices)


def test_initialize_threads_custom_register_and_local_size():
    tb = ThreadBlock((0, 0, 0), (2, 1, 1), (1, 1, 1), shared_mem_size=0)
    tb.initialize_threads(lambda *a: None, register_mem_size=8, local_mem_size=16)
    for t in tb.threads:
        assert t.registers.size == 8
        assert t.local_mem.size == 16
