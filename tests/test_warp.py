import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.warp import Warp
from py_virtual_gpu.thread import Thread
from py_virtual_gpu.streaming_multiprocessor import StreamingMultiprocessor


def test_warp_attributes_and_execute_basic(monkeypatch):
    threads = [Thread() for _ in range(3)]
    sm = StreamingMultiprocessor(id=0, shared_mem_size=0, max_registers_per_thread=0, warp_size=3)
    w = Warp(id=0, threads=threads, sm=sm)
    assert w.id == 0
    assert len(w.threads) == 3
    assert w.active_mask == [True, True, True]
    from py_virtual_gpu.dispatch import Instruction
    monkeypatch.setattr(Warp, "fetch_next_instruction", lambda self: Instruction("NOP", tuple()))
    monkeypatch.setattr(Warp, "evaluate_predicate", lambda self, inst: [True, True, True])
    w.execute()
    assert w.pc == 1
    assert sm.counters["warp_divergences"] == 0
    inst = Instruction("NOP", tuple())
    w.issue_instruction(inst)
    assert w.pc == 2
