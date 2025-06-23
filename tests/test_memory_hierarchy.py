import os
import sys
import math
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.memory_hierarchy import RegisterFile, SharedMemory


def test_register_file_defaults():
    rf = RegisterFile()
    assert rf.latency_cycles == 1
    assert rf.bandwidth_bpc == rf.size


def test_shared_memory_latency_cycles():
    sm = SharedMemory()
    sm.read(0, 128)
    expected = 10 + math.ceil(128 / (96 * 1024))
    assert sm.stats["cycles"] == expected


def test_stats_account_and_reset():
    rf = RegisterFile(1024)
    rf.write(0, b"a" * 100)
    rf.read(0, 50)
    cycles = rf.latency_cycles + math.ceil(100 / rf.bandwidth_bpc)
    cycles += rf.latency_cycles + math.ceil(50 / rf.bandwidth_bpc)
    assert rf.stats == {
        "reads": 1,
        "writes": 1,
        "cycles": cycles,
        "spill_events": 0,
        "spill_bytes": 0,
        "spill_cycles": 0,
    }
    rf.reset_stats()
    assert rf.stats == {
        "reads": 0,
        "writes": 0,
        "cycles": 0,
        "spill_events": 0,
        "spill_bytes": 0,
        "spill_cycles": 0,
    }


def test_bounds_checks():
    rf = RegisterFile(32)
    with pytest.raises(IndexError):
        rf.read(16, 20)
    with pytest.raises(RuntimeError):
        rf.write(20, b"\x00" * 20)
    sm = SharedMemory(64)
    with pytest.raises(IndexError):
        sm.read(32, 40)
    with pytest.raises(IndexError):
        sm.write(40, b"\x00" * 40)

