import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.streaming_multiprocessor import StreamingMultiprocessor
from py_virtual_gpu.warp import Warp
from py_virtual_gpu.dispatch import Instruction
from py_virtual_gpu.types import Half, Float32, Float64


class DummyThread:
    pass


class DummyBlock:
    def __init__(self, num_threads: int):
        self.threads = [DummyThread() for _ in range(num_threads)]


def test_sm_instantiation_and_queue():
    sm = StreamingMultiprocessor(id=0, shared_mem_size=256, max_registers_per_thread=16)
    assert sm.block_queue.empty()
    assert sm.warp_queue.empty()
    assert sm.schedule_policy == "round_robin"
    sm.block_queue.put(DummyBlock(1))
    assert sm.block_queue.qsize() == 1


def test_execute_block_warp_count():
    sm = StreamingMultiprocessor(id=0, shared_mem_size=128, max_registers_per_thread=16, warp_size=32)
    block = DummyBlock(65)
    # Prevent NotImplementedError from Warp.execute
    def _nop(self):
        self.active_mask = [False] * len(self.active_mask)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(Warp, "execute", _nop)
    sm.execute_block(block)
    monkeypatch.undo()
    assert sm.warp_queue.empty()
    assert sm.counters["warps_executed"] == 3


def test_sequential_policy_executes_each_warp_once():
    sm = StreamingMultiprocessor(id=1, shared_mem_size=128, max_registers_per_thread=16, warp_size=2)
    sm.schedule_policy = "sequential"
    block = DummyBlock(4)

    order = []

    def _record(self):
        order.append(self.id)
        self.active_mask = [False] * len(self.active_mask)

    mp = pytest.MonkeyPatch()
    mp.setattr(Warp, "execute", _record)
    sm.execute_block(block)
    mp.undo()

    assert order == [0, 1]
    assert sm.warp_queue.empty()


def test_round_robin_reenqueues_active_warps():
    sm = StreamingMultiprocessor(id=2, shared_mem_size=128, max_registers_per_thread=16, warp_size=2)
    sm.schedule_policy = "round_robin"
    block = DummyBlock(4)

    call_count = {0: 0, 1: 0}

    def _toggle(self):
        call_count[self.id] += 1
        if self.id == 0 and call_count[self.id] == 1:
            # remain active so it will be re-enqueued
            self.active_mask[0] = False
            self.active_mask[1] = True
        else:
            self.active_mask = [False] * len(self.active_mask)

    mp = pytest.MonkeyPatch()
    mp.setattr(Warp, "execute", _toggle)
    sm.execute_block(block)
    mp.undo()

    assert call_count[0] == 2
    assert call_count[1] == 1
    assert sm.warp_queue.empty()


def test_reset_counters_and_queue():
    sm = StreamingMultiprocessor(id=0, shared_mem_size=128, max_registers_per_thread=16)
    sm.block_queue.put(DummyBlock(2))
    sm.warp_queue.put(Warp(0, [], sm))
    sm.counters["warps_executed"] = 5
    sm.counters["warp_divergences"] = 2
    sm.reset()
    assert sm.block_queue.empty()
    assert sm.warp_queue.empty()
    assert sm.counters["warps_executed"] == 0
    assert sm.counters["warp_divergences"] == 0


def test_repr_contains_info():
    sm = StreamingMultiprocessor(id=7, shared_mem_size=64, max_registers_per_thread=8)
    text = repr(sm)
    assert "id=7" in text
    assert "queue_size=0" in text
    assert "warps=0" in text


def test_instruction_cycle_accounting():
    sm = StreamingMultiprocessor(
        id=3,
        shared_mem_size=0,
        max_registers_per_thread=0,
        fp16_cycles=1,
        fp32_cycles=2,
        fp64_cycles=3,
    )
    w = Warp(0, [DummyThread(), DummyThread()], sm)

    w.issue_instruction(Instruction("ADD", (Half(1.0),)))
    assert sm.counters["cycles"] == 1

    w.issue_instruction(Instruction("ADD", (Float32(1.0),)))
    assert sm.counters["cycles"] == 3

    w.issue_instruction(Instruction("ADD", (Float64(1.0),)))
    assert sm.counters["cycles"] == 6
