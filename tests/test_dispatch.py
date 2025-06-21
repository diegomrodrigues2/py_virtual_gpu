import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.dispatch import Instruction, SIMTStack
from py_virtual_gpu.warp import Warp
from py_virtual_gpu.thread import Thread
from py_virtual_gpu.streaming_multiprocessor import StreamingMultiprocessor


def test_instruction_and_simtstack_basic():
    inst = Instruction("ADD", ("r1", "r2"))
    assert inst.opcode == "ADD"
    stack = SIMTStack()
    stack.push([True, False], 5)
    assert stack.top() == ([True, False], 5)
    mask, pc = stack.pop()
    assert (mask, pc) == ([True, False], 5)


def test_warp_issue_instruction_stub():
    sm = StreamingMultiprocessor(id=0, shared_mem_size=0, max_registers_per_thread=0, warp_size=2)
    w = Warp(0, [Thread(), Thread()], sm)
    inst = Instruction("NOP", tuple())
    with pytest.raises(NotImplementedError):
        w.issue_instruction(inst)


def test_sm_dispatch_round_robin_stub():
    sm = StreamingMultiprocessor(id=0, shared_mem_size=8, max_registers_per_thread=4, warp_size=2)
    w1 = Warp(0, [Thread(), Thread()], sm)
    w2 = Warp(1, [Thread(), Thread()], sm)
    sm.warp_queue.put(w1)
    sm.warp_queue.put(w2)
    with pytest.raises(NotImplementedError):
        sm.dispatch()


def test_sm_record_divergence_counter():
    sm = StreamingMultiprocessor(id=1, shared_mem_size=8, max_registers_per_thread=4)
    assert sm.counters["warp_divergences"] == 0
    sm.record_divergence(Warp(0, [Thread(), Thread()], sm), 0, [True, True], [False, True])
    assert sm.counters["warp_divergences"] == 1

