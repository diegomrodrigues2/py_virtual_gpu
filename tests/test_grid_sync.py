import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.virtualgpu import VirtualGPU
from py_virtual_gpu.sync import sync_grid
from py_virtual_gpu.errors import SynchronizationError


def test_sync_grid_success():
    gpu = VirtualGPU(0, 32, barrier_timeout=0.1)

    from multiprocessing import Manager

    records = Manager().list()

    def kernel(tidx, bidx, bdim, gdim, log):
        log.append(("before", tidx))
        sync_grid()
        log.append(("after", tidx))

    gpu.launch_kernel(kernel, (1, 1, 1), (2, 1, 1), records, cooperative=True)

    total_threads = 2
    assert len(records) == 2 * total_threads
    before = [i for i, r in enumerate(list(records)) if r[0] == "before"]
    after = [i for i, r in enumerate(list(records)) if r[0] == "after"]
    assert len(before) == total_threads
    assert len(after) == total_threads
    assert min(after) > max(before)


def test_sync_grid_missing_thread_raises():
    gpu = VirtualGPU(0, 32, barrier_timeout=0.05)
    from multiprocessing import Manager

    errors = Manager().list()

    def kernel(tidx, bidx, bdim, gdim, log):
        if tidx[0] == 0:
            return
        try:
            sync_grid()
        except SynchronizationError:
            log.append(tidx)

    gpu.launch_kernel(kernel, (1, 1, 1), (2, 1, 1), errors, cooperative=True)

    assert list(errors) == [(1, 0, 0)]
