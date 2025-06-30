import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import VirtualGPU, Float32, sqrt_numeric
from py_virtual_gpu.kernel import kernel
from py_virtual_gpu.services import get_gpu_manager
from py_virtual_gpu.api.server import start_background_api


@kernel
def adam_step(threadIdx, blockIdx, blockDim, gridDim,
               param_ptr, grad_ptr, m_ptr, v_ptr,
               lr, beta1, beta2, eps, corr1, corr2, n):
    i = blockIdx[0] * blockDim[0] + threadIdx[0]
    if i < n:
        g = grad_ptr[i]
        m = m_ptr[i]
        v = v_ptr[i]
        p = param_ptr[i]

        m = beta1 * m + (Float32(1.0) - beta1) * g
        v = beta2 * v + (Float32(1.0) - beta2) * (g * g)

        m_ptr[i] = m
        v_ptr[i] = v

        m_hat = m / corr1
        v_hat = v / corr2
        denom = sqrt_numeric(v_hat) + eps
        param_ptr[i] = p - lr * (m_hat / denom)


def main(with_api: bool = False) -> None:
    if with_api:
        api_thread, stop_api = start_background_api()
    else:
        api_thread = stop_api = None

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
    for i, v in enumerate(grads):
        grad_ptr[i] = Float32(v)
    for i in range(n):
        m_ptr[i] = Float32(0.0)
        v_ptr[i] = Float32(0.0)

    lr = Float32(0.01)
    beta1 = Float32(0.9)
    beta2 = Float32(0.999)
    eps = Float32(1e-8)
    num_steps = 3

    for t in range(1, num_steps + 1):
        corr1 = Float32(1.0 - beta1.value ** t)
        corr2 = Float32(1.0 - beta2.value ** t)
        adam_step(
            param_ptr,
            grad_ptr,
            m_ptr,
            v_ptr,
            lr,
            beta1,
            beta2,
            eps,
            corr1,
            corr2,
            n,
            grid_dim=(1, 1, 1),
            block_dim=(n, 1, 1),
        )
        gpu.synchronize()

    result = [float(param_ptr[i]) for i in range(n)]

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

    print("Kernel result:", result)
    print("Host result:", host_params)

    if stop_api:
        stop_api()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Basic Adam optimizer example")
    parser.add_argument("--api", action="store_true", help="start API server while running")
    args = parser.parse_args()
    main(with_api=args.api)
