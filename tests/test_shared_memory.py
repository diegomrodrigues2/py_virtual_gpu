import os, sys, multiprocessing as mp, pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.shared_memory import SharedMemory


def test_read_write_basic():
    sm = SharedMemory(16)
    sm.write(4, b"abc")
    assert sm.read(4, 3) == b"abc"


def test_bounds_check():
    sm = SharedMemory(8)
    with pytest.raises(IndexError):
        sm.read(7, 2)
    with pytest.raises(IndexError):
        sm.write(7, b"\x00\x01")


def test_atomic_add():
    sm = SharedMemory(4)
    sm.write(0, (5).to_bytes(4, "little", signed=True))
    old = sm.atomic_add(0, 7)
    assert old == 5
    val = int.from_bytes(sm.read(0, 4), "little", signed=True)
    assert val == 12


def _worker_add(shared, loops):
    for _ in range(loops):
        shared.atomic_add(0, 1)


def test_atomic_add_concurrent():
    sm = SharedMemory(4)
    sm.write(0, (0).to_bytes(4, "little", signed=True))
    processes = [mp.Process(target=_worker_add, args=(sm, 1000)) for _ in range(4)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    result = int.from_bytes(sm.read(0, 4), "little", signed=True)
    assert result == 4000
