import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import VirtualGPU, Float32
from py_virtual_gpu.optimizers import adam_step
from py_virtual_gpu.services import get_gpu_manager


def test_adam_step_helper():
    n = 4
    params = [1.0, 2.0, 3.0, 4.0]
    grads = [0.1, -0.2, 0.3, -0.4]

    gpu = VirtualGPU(num_sms=0, global_mem_size=256)
    get_gpu_manager().add_gpu(gpu)
    VirtualGPU.set_current(gpu)

    param_ptr = gpu.malloc(n, dtype=Float32)
    grad_ptr = gpu.malloc(n, dtype=Float32)
    m_ptr = gpu.malloc(n, dtype=Float32)
    v_ptr = gpu.malloc(n, dtype=Float32)

    for i, v in enumerate(params):
        param_ptr[i] = Float32(v)
        grad_ptr[i] = Float32(grads[i])
        m_ptr[i] = Float32(0.0)
        v_ptr[i] = Float32(0.0)

    lr = Float32(0.01)
    beta1 = Float32(0.9)
    beta2 = Float32(0.999)
    eps = Float32(1e-8)
    num_steps = 3

    for t in range(1, num_steps + 1):
        adam_step(
            param_ptr,
            grad_ptr,
            m_ptr,
            v_ptr,
            lr,
            beta1,
            beta2,
            eps,
            t,
            n,
            grid_dim=(1, 1, 1),
            block_dim=(n, 1, 1),
        )
        gpu.synchronize()

    kernel_res = [float(param_ptr[i]) for i in range(n)]

    host_params = params.copy()
    host_m = [0.0] * n
    host_v = [0.0] * n
    for t in range(1, num_steps + 1):
        corr1 = 1.0 - beta1.value ** t
        corr2 = 1.0 - beta2.value ** t
        for i in range(n):
            g = grads[i]
            host_m[i] = beta1.value * host_m[i] + (1.0 - beta1.value) * g
            host_v[i] = beta2.value * host_v[i] + (1.0 - beta2.value) * (g * g)
            m_hat = host_m[i] / corr1
            v_hat = host_v[i] / corr2
            denom = (v_hat ** 0.5) + eps.value
            host_params[i] = host_params[i] - lr.value * (m_hat / denom)

    assert kernel_res == pytest.approx(host_params)


def test_adam_step_kernel_logging():
    """Ensure adam_step records a kernel launch for each invocation."""
    n = 2
    params = [1.0, 2.0]
    grads = [0.5, -0.25]

    gpu = VirtualGPU(num_sms=0, global_mem_size=64)
    get_gpu_manager().add_gpu(gpu)
    VirtualGPU.set_current(gpu)

    param_ptr = gpu.malloc(n, dtype=Float32)
    grad_ptr = gpu.malloc(n, dtype=Float32)
    m_ptr = gpu.malloc(n, dtype=Float32)
    v_ptr = gpu.malloc(n, dtype=Float32)

    for i in range(n):
        param_ptr[i] = Float32(params[i])
        grad_ptr[i] = Float32(grads[i])
        m_ptr[i] = Float32(0.0)
        v_ptr[i] = Float32(0.0)

    lr = Float32(0.01)
    beta1 = Float32(0.9)
    beta2 = Float32(0.999)
    eps = Float32(1e-8)

    num_steps = 3
    prev_len = 0
    for t in range(1, num_steps + 1):
        adam_step(
            param_ptr,
            grad_ptr,
            m_ptr,
            v_ptr,
            lr,
            beta1,
            beta2,
            eps,
            t,
            n,
            grid_dim=(1, 1, 1),
            block_dim=(n, 1, 1),
        )
        gpu.synchronize()

        log = gpu.get_kernel_log()
        assert len(log) == prev_len + 1
        prev_len = len(log)

    assert len(gpu.get_kernel_log()) == num_steps
