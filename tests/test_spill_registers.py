import os
import sys
from unittest import mock
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.thread import Thread


def test_no_spill_when_within_capacity():
    t = Thread(register_mem_size=8)
    t.registers.write(0, b"abcd")
    assert t.get_spill_stats() == {"spill_events": 0, "spill_bytes": 0, "spill_cycles": 0}
    assert t.local_mem.stats["writes"] == 0


def test_spill_exact_overflow():
    t = Thread(register_mem_size=8)
    with mock.patch.object(t.local_mem, "write", wraps=t.local_mem.write) as w:
        t.registers.write(0, b"abcdefghij")
        assert w.call_count == 1
    assert t.registers.read(0, 8) == b"abcdefgh"
    assert t.local_mem.read(0, 2) == b"ij"
    stats = t.get_spill_stats()
    assert stats["spill_events"] == 1
    assert stats["spill_bytes"] == 2
    assert stats["spill_cycles"] == 50


def test_multiple_spill_events():
    t = Thread(register_mem_size=8)
    with mock.patch.object(t.local_mem, "write", wraps=t.local_mem.write) as w:
        t.registers.write(0, b"AAAAAA")  # fits
        t.registers.write(6, b"BBBB")    # 2 spill
        t.registers.write(10, b"CCCCC")  # 5 spill
        assert w.call_count == 2
    assert t.local_mem.read(0, 2) == b"BB"
    assert t.local_mem.read(2, 5) == b"CCCCC"
    stats = t.get_spill_stats()
    assert stats["spill_events"] == 2
    assert stats["spill_bytes"] == 7
    expected_cycles = math.ceil(2/4)*50 + math.ceil(5/4)*50
    assert stats["spill_cycles"] == expected_cycles
