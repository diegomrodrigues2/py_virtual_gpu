import os
import sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import VirtualGPU



class DummySM:
    def __init__(self):
        import queue
        self.block_queue = queue.Queue()


def test_constant_memory_default_size_and_set():
    gpu = VirtualGPU(0, 32)
    assert gpu.const_memory.size == 64 * 1024
    gpu.set_constant(b"abc")
    assert gpu.const_memory.read(0, 3) == b"abc"


def test_set_constant_bounds():
    gpu = VirtualGPU(0, 16)
    with pytest.raises(ValueError):
        gpu.set_constant(b"a" * (gpu.const_memory.size + 1))


def test_launch_kernel_exposes_const_mem():
    gpu = VirtualGPU(0, 32)
    gpu.sms = [DummySM()]

    def dummy():
        pass

    gpu.launch_kernel(dummy, (1, 1, 1), (1, 1, 1))
    tb = gpu.sms[0].block_queue.get()
    t = tb.threads[0]
    assert t.const_mem is gpu.const_memory


def test_read_constant_wrapper():
    gpu = VirtualGPU(0, 32)
    gpu.set_constant(b"xyz")
    assert gpu.read_constant(0, 3) == b"xyz"

