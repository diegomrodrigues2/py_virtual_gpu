import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import VirtualGPU, Float32, atomicAdd_float32
from py_virtual_gpu.kernel import kernel
from py_virtual_gpu.optimizers import adam_step
from py_virtual_gpu.services import get_gpu_manager
from py_virtual_gpu.api.server import start_background_api


@kernel
def compute_grads(threadIdx, blockIdx, blockDim, gridDim, x_ptr, y_ptr, params_ptr, grad_ptr, n):
    i = blockIdx[0] * blockDim[0] + threadIdx[0]
    if i < n:
        w = params_ptr[0]
        b = params_ptr[1]
        x = x_ptr[i]
        y = y_ptr[i]
        y_hat = w * x + b
        diff = y_hat - y
        atomicAdd_float32(grad_ptr, (diff * x) / Float32(n))
        atomicAdd_float32(grad_ptr + 1, diff / Float32(n))


def main(with_api: bool = False) -> None:
    if with_api:
        api_thread, stop_api = start_background_api()
    else:
        api_thread = stop_api = None

    np.random.seed(0)
    n_samples = 32
    x_vals = np.linspace(-1.0, 1.0, n_samples, dtype=np.float32)
    true_w = 2.0
    true_b = 1.0
    y_vals = true_w * x_vals + true_b + np.random.normal(scale=0.1, size=n_samples).astype(np.float32)

    gpu = VirtualGPU(num_sms=0, global_mem_size=1024)
    get_gpu_manager().add_gpu(gpu)
    VirtualGPU.set_current(gpu)

    x_ptr = gpu.malloc(n_samples, dtype=Float32, label="x")
    y_ptr = gpu.malloc(n_samples, dtype=Float32, label="y")
    params_ptr = gpu.malloc(2, dtype=Float32, label="params")
    grad_ptr = gpu.malloc(2, dtype=Float32, label="grads")
    m_ptr = gpu.malloc(2, dtype=Float32, label="m")
    v_ptr = gpu.malloc(2, dtype=Float32, label="v")

    for i, v in enumerate(x_vals):
        x_ptr[i] = Float32(float(v))
    for i, v in enumerate(y_vals):
        y_ptr[i] = Float32(float(v))

    params_ptr[0] = Float32(0.0)
    params_ptr[1] = Float32(0.0)
    for i in range(2):
        m_ptr[i] = Float32(0.0)
        v_ptr[i] = Float32(0.0)

    lr = Float32(0.01)
    beta1 = Float32(0.9)
    beta2 = Float32(0.999)
    eps = Float32(1e-8)
    epochs = 50

    host_params = [0.0, 0.0]
    host_m = [0.0, 0.0]
    host_v = [0.0, 0.0]

    for t in range(1, epochs + 1):
        grad_ptr[0] = Float32(0.0)
        grad_ptr[1] = Float32(0.0)

        compute_grads(
            x_ptr,
            y_ptr,
            params_ptr,
            grad_ptr,
            n_samples,
            grid_dim=(1, 1, 1),
            block_dim=(n_samples, 1, 1),
        )
        gpu.synchronize()

        adam_step(
            params_ptr,
            grad_ptr,
            m_ptr,
            v_ptr,
            lr,
            beta1,
            beta2,
            eps,
            t,
            2,
            grid_dim=(1, 1, 1),
            block_dim=(2, 1, 1),
        )
        gpu.synchronize()

        # host computation
        g_w = 0.0
        g_b = 0.0
        for i in range(n_samples):
            y_hat = host_params[0] * x_vals[i] + host_params[1]
            diff = y_hat - y_vals[i]
            g_w += (diff * x_vals[i]) / n_samples
            g_b += diff / n_samples

        corr1 = 1.0 - beta1.value ** t
        corr2 = 1.0 - beta2.value ** t
        host_m[0] = beta1.value * host_m[0] + (1.0 - beta1.value) * g_w
        host_m[1] = beta1.value * host_m[1] + (1.0 - beta1.value) * g_b
        host_v[0] = beta2.value * host_v[0] + (1.0 - beta2.value) * (g_w * g_w)
        host_v[1] = beta2.value * host_v[1] + (1.0 - beta2.value) * (g_b * g_b)
        m_hat_w = host_m[0] / corr1
        v_hat_w = host_v[0] / corr2
        m_hat_b = host_m[1] / corr1
        v_hat_b = host_v[1] / corr2
        host_params[0] -= lr.value * (m_hat_w / ((v_hat_w ** 0.5) + eps.value))
        host_params[1] -= lr.value * (m_hat_b / ((v_hat_b ** 0.5) + eps.value))

    result = [float(params_ptr[i]) for i in range(2)]
    print("Kernel result:", result)
    print("Host result:", host_params)

    if stop_api:
        stop_api()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="1D linear regression with Adam")
    parser.add_argument("--api", action="store_true", help="start API server while running")
    args = parser.parse_args()
    main(with_api=args.api)
