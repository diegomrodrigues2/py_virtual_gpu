import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.thread_block import ThreadBlock
from py_virtual_gpu.warp_utils import shfl_sync, ballot_sync


def test_shfl_sync_exchanges_values():
    block = ThreadBlock((0, 0, 0), (4, 1, 1), (1, 1, 1), shared_mem_size=0)
    from multiprocessing import Manager

    results = Manager().list([None] * 4)

    def kernel(tidx, bidx, bdim, gdim, out):
        val = tidx[0] + 1
        out[tidx[0]] = shfl_sync(val, 0)

    block.execute(kernel, results)
    assert list(results) == [1, 1, 1, 1]


def test_ballot_sync_collects_predicates():
    block = ThreadBlock((0, 0, 0), (4, 1, 1), (1, 1, 1), shared_mem_size=0)
    from multiprocessing import Manager

    results = Manager().list([None] * 4)

    def kernel(tidx, bidx, bdim, gdim, out):
        pred = tidx[0] % 2 == 0
        out[tidx[0]] = ballot_sync(pred)

    block.execute(kernel, results)
    assert list(results) == [5, 5, 5, 5]
