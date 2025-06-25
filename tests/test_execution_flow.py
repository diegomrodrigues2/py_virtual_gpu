import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.streaming_multiprocessor import StreamingMultiprocessor
from py_virtual_gpu.warp import Warp
from py_virtual_gpu.thread import Thread
from py_virtual_gpu.dispatch import Instruction


class Block:
    def __init__(self, num_threads):
        self.threads = [Thread() for _ in range(num_threads)]


def test_round_robin_reenqueues_active_warp():
    sm = StreamingMultiprocessor(id=0, shared_mem_size=0, max_registers_per_thread=0, warp_size=2)
    sm.schedule_policy = "round_robin"
    block = Block(4)

    mp = pytest.MonkeyPatch()

    mp.setattr(Warp, "fetch_next_instruction", lambda self: Instruction("NOP", tuple()))

    calls = {0: 0, 1: 0}

    def eval_pred(self, inst):
        idx = self.id
        cnt = calls[idx]
        calls[idx] += 1
        if idx == 0:
            return [True, False] if cnt == 0 else [False, False]
        return [False, False]

    mp.setattr(Warp, "evaluate_predicate", eval_pred)
    sm.execute_block(block)
    mp.undo()

    assert calls[0] == 2
    assert calls[1] == 1
    assert sm.warp_queue.empty()


def test_sm_records_divergence_events():
    sm = StreamingMultiprocessor(id=1, shared_mem_size=0, max_registers_per_thread=0, warp_size=2)
    block = Block(2)

    mp = pytest.MonkeyPatch()
    mp.setattr(Warp, "fetch_next_instruction", lambda self: Instruction("BR", tuple()))
    masks = {0: [[True, False], [True, True], [False, False]]}

    def eval_pred(self, inst):
        seq = masks[self.id]
        return seq.pop(0)

    mp.setattr(Warp, "evaluate_predicate", eval_pred)

    sm.execute_block(block)
    mp.undo()

    log = sm.get_divergence_log()
    assert len(log) >= 2
    first, second = log[:2]
    assert first.mask_before == [True, True]
    assert first.mask_after == [True, False]
    assert second.mask_before == [True, False]
    assert second.mask_after == [True, True]


def test_memory_instructions_update_counters():
    sm = StreamingMultiprocessor(id=2, shared_mem_size=64, max_registers_per_thread=0, warp_size=2)
    w = Warp(0, [Thread(), Thread()], sm)

    mp = pytest.MonkeyPatch()
    instructions = [
        Instruction("LD", ("addr_a", 4, "global")),
        Instruction("LD", ("addr_b", 4, "shared")),
    ]

    mp.setattr(Warp, "fetch_next_instruction", lambda self: instructions[self.pc])
    mp.setattr(Warp, "evaluate_predicate", lambda self, inst: [False, False] if self.pc == 2 else [True, True])

    w.threads[0].addr_a = 0
    w.threads[1].addr_a = 8

    w.threads[0].addr_b = 0
    w.threads[1].addr_b = 0

    # execute first instruction (non-coalesced global access)
    w.execute()
    # execute second instruction (shared memory bank conflict)
    w.execute()
    mp.undo()

    assert sm.counters["non_coalesced_accesses"] == 2
    assert sm.counters["bank_conflicts"] == 1
