import os
import sys
import multiprocessing as mp
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.global_memory import GlobalMemory
from py_virtual_gpu.virtualgpu import VirtualGPU
from py_virtual_gpu.atomics import atomicAdd, atomicCAS



def test_atomic_add_and_bounds():
    gm = GlobalMemory(8)
    gm.write(0, (5).to_bytes(4, "little", signed=True))
    old = gm.atomic_add(0, 3)
    assert old == 5
    assert int.from_bytes(gm.read(0, 4), "little", signed=True) == 8
    with pytest.raises(IndexError):
        gm.atomic_add(5, 1)


def test_atomic_cas_and_min():
    gm = GlobalMemory(4)
    gm.write(0, (7).to_bytes(4, "little", signed=True))
    old = gm.atomic_min(0, 3)
    assert old == 7
    assert int.from_bytes(gm.read(0, 4), "little", signed=True) == 3
    swapped = gm.atomic_cas(0, 3, 9)
    assert swapped is True
    assert int.from_bytes(gm.read(0, 4), "little", signed=True) == 9


GM = None

def _worker_sub(loops):
    for _ in range(loops):
        GM.atomic_sub(0, 1)


def test_atomic_sub_concurrent():
    gm = GlobalMemory(4)
    gm.write(0, (0).to_bytes(4, "little", signed=True))
    global GM
    GM = gm
    ctx = mp.get_context("fork")
    with ctx.Pool(4) as pool:
        pool.map(_worker_sub, [1000] * 4)
    raw = memoryview(gm.buffer)[0:4].tobytes()
    result = int.from_bytes(raw, "little", signed=True)
    assert result == -4000


def test_atomic_methods_use_lock():
    gm = GlobalMemory(4)
    gm.write(0, (0).to_bytes(4, "little", signed=True))
    mock_lock = MagicMock()
    mock_lock.__enter__.return_value = None
    mock_lock.__exit__.return_value = None
    gm.lock = mock_lock
    gm.atomic_add(0, 1)
    gm.atomic_sub(0, 1)
    gm.atomic_max(0, 0)
    gm.atomic_cas(0, 0, 1)
    gm.atomic_exchange(0, 0)
    assert mock_lock.__enter__.call_count == 5
    assert mock_lock.__exit__.call_count == 5


def test_atomic_wrappers_through_virtualgpu():
    gpu = VirtualGPU(0, 16)
    VirtualGPU.set_current(gpu)
    ptr = gpu.malloc(4)
    gpu.global_mem.write(ptr.offset, (1).to_bytes(4, "little", signed=True))
    old = atomicAdd(ptr, 2)
    assert old == 1
    assert int.from_bytes(gpu.global_mem.read(ptr.offset, 4), "little", signed=True) == 3
    swapped = atomicCAS(ptr, 3, 5)
    assert swapped is True
    assert int.from_bytes(gpu.global_mem.read(ptr.offset, 4), "little", signed=True) == 5
