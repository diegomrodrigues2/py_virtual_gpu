import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.streaming_multiprocessor import StreamingMultiprocessor
from py_virtual_gpu.warp import Warp
from py_virtual_gpu.thread import Thread
from py_virtual_gpu.dispatch import Instruction


def test_divergence_events_and_log(monkeypatch):
    sm = StreamingMultiprocessor(id=0, shared_mem_size=0, max_registers_per_thread=0, warp_size=2)
    w = Warp(0, [Thread(), Thread()], sm)

    monkeypatch.setattr(Warp, "fetch_next_instruction", lambda self: Instruction("BR", tuple()))
    masks = [[True, False], [True, True]]
    monkeypatch.setattr(Warp, "evaluate_predicate", lambda self, inst: masks.pop(0))

    w.execute()  # diverge
    w.execute()  # reconverge

    log = sm.get_divergence_log()
    assert sm.counters["warp_divergences"] == 2
    assert len(log) == 2
    first, second = log
    assert first.warp_id == 0
    assert first.pc == 0
    assert first.mask_before == [True, True]
    assert first.mask_after == [True, False]
    assert second.mask_before == [True, False]
    assert second.mask_after == [True, True]

    sm.clear_divergence_log()
    assert sm.get_divergence_log() == []
    assert sm.counters["warp_divergences"] == 2
