import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.thread import Thread
from py_virtual_gpu.shared_memory import SharedMemory
from py_virtual_gpu.global_memory import GlobalMemory
from py_virtual_gpu.fence import threadfence_block, threadfence, threadfence_system


def test_threadfence_block_acquires_lock():
    sm = SharedMemory(4)
    t = Thread((0, 0, 0), (0, 0, 0), (1, 1, 1), (1, 1, 1), 0, shared_mem=sm)

    mock = MagicMock()
    mock.__enter__.return_value = None
    mock.__exit__.return_value = None
    sm.lock = mock

    def kernel(tidx, bidx, bdim, gdim):
        threadfence_block()

    t.run(kernel, t.thread_idx, t.block_idx, t.block_dim, t.grid_dim)
    assert mock.__enter__.called
    assert mock.__exit__.called


def test_threadfence_acquires_global_lock():
    gm = GlobalMemory(4)
    t = Thread((0, 0, 0), (0, 0, 0), (1, 1, 1), (1, 1, 1), 0, global_mem=gm)

    mock = MagicMock()
    mock.__enter__.return_value = None
    mock.__exit__.return_value = None
    gm.lock = mock

    def kernel(tidx, bidx, bdim, gdim):
        threadfence()

    t.run(kernel, t.thread_idx, t.block_idx, t.block_dim, t.grid_dim)
    assert mock.__enter__.called
    assert mock.__exit__.called


def test_threadfence_system_alias():
    gm = GlobalMemory(4)
    t = Thread((0, 0, 0), (0, 0, 0), (1, 1, 1), (1, 1, 1), 0, global_mem=gm)

    def kernel(tidx, bidx, bdim, gdim):
        gm.write(0, b"\x05")
        threadfence_system()

    t.run(kernel, t.thread_idx, t.block_idx, t.block_dim, t.grid_dim)
    assert gm.read(0, 1) == b"\x05"

from py_virtual_gpu.thread_block import ThreadBlock


def test_threadfence_block_orders_writes_between_threads():
    tb = ThreadBlock((0, 0, 0), (2, 1, 1), (1, 1, 1), shared_mem_size=1)
    from multiprocessing import Manager

    results = Manager().list([0, 0])

    def kernel(tidx, bidx, bdim, gdim, barrier, out):
        if tidx[0] == 0:
            tb.shared_mem.write(0, b"\x07")
            threadfence_block()
        barrier.wait()
        out[tidx[0]] = tb.shared_mem.read(0, 1)[0]

    tb.execute(kernel, tb.barrier, results)
    assert list(results)[1] == 7
