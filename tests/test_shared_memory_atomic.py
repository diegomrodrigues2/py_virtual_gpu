import os
import sys
import multiprocessing as mp
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.shared_memory import SharedMemory


def test_atomic_sub_basic():
    sm = SharedMemory(4)
    sm.write(0, (10).to_bytes(4, "little", signed=True))
    old = sm.atomic_sub(0, 3)
    assert old == 10
    val = int.from_bytes(sm.read(0, 4), "little", signed=True)
    assert val == 7


def test_atomic_max_and_cas():
    sm = SharedMemory(4)
    sm.write(0, (5).to_bytes(4, "little", signed=True))
    old = sm.atomic_max(0, 8)
    assert old == 5
    assert int.from_bytes(sm.read(0, 4), "little", signed=True) == 8
    swapped = sm.atomic_cas(0, 8, 1)
    assert swapped is True
    assert int.from_bytes(sm.read(0, 4), "little", signed=True) == 1
    swapped = sm.atomic_cas(0, 8, 2)
    assert swapped is False


SHARED = None


def _worker_sub(loops):
    for _ in range(loops):
        SHARED.atomic_sub(0, 1)


def test_atomic_sub_concurrent():
    sm = SharedMemory(4)
    sm.write(0, (0).to_bytes(4, "little", signed=True))
    global SHARED
    SHARED = sm
    ctx = mp.get_context("fork")
    with ctx.Pool(4) as pool:
        pool.map(_worker_sub, [1000] * 4)
    result = int.from_bytes(sm.read(0, 4), "little", signed=True)
    assert result == -4000


def test_atomic_methods_use_lock():
    sm = SharedMemory(4)
    sm.write(0, (0).to_bytes(4, "little", signed=True))
    mock_lock = MagicMock()
    mock_lock.__enter__.return_value = None
    mock_lock.__exit__.return_value = None
    sm.lock = mock_lock
    sm.atomic_add(0, 1)
    sm.atomic_sub(0, 1)
    sm.atomic_max(0, 0)
    sm.atomic_cas(0, 0, 1)
    assert mock_lock.__enter__.call_count == 4
    assert mock_lock.__exit__.call_count == 4
