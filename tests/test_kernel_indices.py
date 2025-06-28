import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.kernel import kernel
from py_virtual_gpu.virtualgpu import VirtualGPU


def test_kernel_receives_thread_and_block_indices():
    gpu = VirtualGPU(0, 32)
    gpu.sms = []  # execute synchronously
    VirtualGPU.set_current(gpu)

    from multiprocessing import Manager

    results = Manager().list()

    @kernel(grid_dim=(2, 1, 1), block_dim=(2, 1, 1))
    def collect(threadIdx, blockIdx, blockDim, gridDim, val):
        results.append((threadIdx, blockIdx, blockDim, gridDim, val))

    collect(5)

    assert len(results) == 4
    combos = set((t, b) for t, b, _, _, _ in list(results))
    assert combos == {
        ((0, 0, 0), (0, 0, 0)),
        ((1, 0, 0), (0, 0, 0)),
        ((0, 0, 0), (1, 0, 0)),
        ((1, 0, 0), (1, 0, 0)),
    }
    for _, _, bdim, gdim, v in list(results):
        assert bdim == (2, 1, 1)
        assert gdim == (2, 1, 1)
        assert v == 5
