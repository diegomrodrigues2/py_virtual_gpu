import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.virtualgpu import VirtualGPU


def test_virtualgpu_presets_configure_sm_latencies():
    gpu = VirtualGPU(num_sms=1, global_mem_size=32, preset="A100")
    sm = gpu.sms[0]
    assert (sm.fp16_cycles, sm.fp32_cycles, sm.fp64_cycles) == (1, 2, 4)

    gpu2 = VirtualGPU(num_sms=1, global_mem_size=32, preset="RTX3080")
    sm2 = gpu2.sms[0]
    assert (sm2.fp16_cycles, sm2.fp32_cycles, sm2.fp64_cycles) == (2, 4, 8)
