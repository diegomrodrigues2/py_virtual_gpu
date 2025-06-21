import pytest

from py_virtual_gpu.virtualgpu import VirtualGPU
import queue


class DummySM:
    def __init__(self):
        self.block_queue = queue.Queue()


def test_launch_kernel_round_robin_queueing():
    gpu = VirtualGPU(0, 32)
    gpu.sms = [DummySM(), DummySM()]

    def dummy():
        pass

    gpu.launch_kernel(dummy, (2, 1, 1), (1, 1, 1))
    assert gpu.sms[0].block_queue.qsize() == 1
    assert gpu.sms[1].block_queue.qsize() == 1
    b0 = gpu.sms[0].block_queue.get()
    b1 = gpu.sms[1].block_queue.get()
    assert b0.block_idx == (0, 0, 0)
    assert b1.block_idx == (1, 0, 0)


def test_launch_kernel_thread_context():
    gpu = VirtualGPU(0, 32, shared_mem_size=8)
    gpu.sms = [DummySM()]

    def dummy():
        pass

    gpu.launch_kernel(dummy, (1, 1, 1), (2, 1, 1))
    tb = gpu.sms[0].block_queue.get()
    assert len(tb.threads) == 2
    t0 = tb.threads[0]
    assert t0.block_idx == (0, 0, 0)
    assert t0.thread_idx == (0, 0, 0)
    assert t0.block_dim == (2, 1, 1)
    assert t0.grid_dim == (1, 1, 1)
    assert t0.global_mem is gpu.global_memory
    assert t0.shared_mem is tb.shared_mem
