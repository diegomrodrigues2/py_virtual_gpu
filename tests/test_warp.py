import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.warp import Warp
from py_virtual_gpu.thread import Thread


def test_warp_attributes_and_execute_stub():
    threads = [Thread() for _ in range(3)]
    w = Warp(id=0, threads=threads)
    assert w.id == 0
    assert len(w.threads) == 3
    assert w.active_mask == [True, True, True]
    with pytest.raises(NotImplementedError):
        w.execute()
    from py_virtual_gpu.dispatch import Instruction
    inst = Instruction("NOP", tuple())
    with pytest.raises(NotImplementedError):
        w.issue_instruction(inst)
