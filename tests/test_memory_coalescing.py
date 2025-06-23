import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.warp import Warp, is_coalesced
from py_virtual_gpu.shared_memory import SharedMemory
from py_virtual_gpu.thread import Thread
from py_virtual_gpu.streaming_multiprocessor import StreamingMultiprocessor


def test_is_coalesced_true():
    assert is_coalesced([0, 4, 8, 12], 4) is True


def test_is_coalesced_false():
    assert is_coalesced([0, 8, 20, 32], 4) is False


def test_detect_bank_conflicts():
    sm = SharedMemory(64, num_banks=4, bank_stride=4)
    addrs = [0, 4, 8, 0]
    assert sm.detect_bank_conflicts(addrs) == 1


def test_warp_memory_access_updates_counters():
    sm = StreamingMultiprocessor(id=0, shared_mem_size=64, max_registers_per_thread=0, warp_size=2)
    w = Warp(0, [Thread(), Thread()], sm)

    w.memory_access([0, 8], 4)
    assert sm.counters["non_coalesced_accesses"] == 1
    assert sm.stats["extra_cycles"] == 1

    w.memory_access([0, 0], 4, space="shared")
    assert sm.counters["bank_conflicts"] == 1
    assert sm.stats["extra_cycles"] == 2
