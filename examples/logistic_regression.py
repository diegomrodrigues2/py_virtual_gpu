import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import VirtualGPU, Float32, atomicAdd_float32
from py_virtual_gpu.kernel import kernel
from py_virtual_gpu.optimizers import adam_step
from py_virtual_gpu.types import exp_numeric
from py_virtual_gpu.services import get_gpu_manager
from py_virtual_gpu.api.server import start_background_api


@kernel
def compute_grads(threadIdx, blockIdx, blockDim, gridDim, x_ptr, y_ptr, params_ptr, grad_ptr, n):
    i = blockIdx[0] * blockDim[0] + threadIdx[0]
    if i < n:
        w1 = params_ptr[0]
        w2 = params_ptr[1]
        b = params_ptr[2]
        x1 = x_ptr[i * 2]
        x2 = x_ptr[i * 2 + 1]
        label = y_ptr[i]
        z = w1 * x1 + w2 * x2 + b
        p = Float32(1.0) / (Float32(1.0) + exp_numeric(Float32(0.0) - z))
        loss_grad = p - label
        atomicAdd_float32(grad_ptr, (loss_grad * x1) / Float32(n))
        atomicAdd_float32(grad_ptr + 1, (loss_grad * x2) / Float32(n))
        atomicAdd_float32(grad_ptr + 2, loss_grad / Float32(n))


def main(with_api: bool = False) -> None:
    if with_api:
        api_thread, stop_api = start_background_api()
    else:
        api_thread = stop_api = None

    np.random.seed(0)
    n_samples = 32
    half = n_samples // 2
    class0 = np.random.normal(loc=(-1.0, -1.0), scale=0.2, size=(half, 2)).astype(np.float32)
    class1 = np.random.normal(loc=(1.0, 1.0), scale=0.2, size=(half, 2)).astype(np.float32)
    x_vals = np.vstack([class0, class1])
    y_vals = np.concatenate([
        np.zeros(half, dtype=np.float32),
        np.ones(half, dtype=np.float32),
    ])

    gpu = VirtualGPU(num_sms=0, global_mem_size=1024)
    get_gpu_manager().add_gpu(gpu)
    VirtualGPU.set_current(gpu)

    x_ptr = gpu.malloc(n_samples * 2, dtype=Float32, label="x")
    y_ptr = gpu.malloc(n_samples, dtype=Float32, label="y")
    params_ptr = gpu.malloc(3, dtype=Float32, label="params")
    grad_ptr = gpu.malloc(3, dtype=Float32, label="grads")
    m_ptr = gpu.malloc(3, dtype=Float32, label="m")
    v_ptr = gpu.malloc(3, dtype=Float32, label="v")

    for i in range(n_samples):
        x_ptr[i * 2] = Float32(float(x_vals[i, 0]))
        x_ptr[i * 2 + 1] = Float32(float(x_vals[i, 1]))
        y_ptr[i] = Float32(float(y_vals[i]))

    for i in range(3):
        params_ptr[i] = Float32(0.0)
        m_ptr[i] = Float32(0.0)
        v_ptr[i] = Float32(0.0)

    lr = Float32(0.01)
    beta1 = Float32(0.9)
    beta2 = Float32(0.999)
    eps = Float32(1e-8)
    epochs = 50

    host_params = [np.float32(0.0), np.float32(0.0), np.float32(0.0)]
    host_m = [np.float32(0.0), np.float32(0.0), np.float32(0.0)]
    host_v = [np.float32(0.0), np.float32(0.0), np.float32(0.0)]

    for t in range(1, epochs + 1):
        grad_ptr[0] = Float32(0.0)
        grad_ptr[1] = Float32(0.0)
        grad_ptr[2] = Float32(0.0)

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
            3,
            grid_dim=(1, 1, 1),
            block_dim=(3, 1, 1),
        )
        gpu.synchronize()

        # host computation
        g_w1 = np.float32(0.0)
        g_w2 = np.float32(0.0)
        g_b = np.float32(0.0)
        for i in range(n_samples):
            x1 = np.float32(x_vals[i, 0])
            x2 = np.float32(x_vals[i, 1])
            label = np.float32(y_vals[i])
            z = host_params[0] * x1 + host_params[1] * x2 + host_params[2]
            p = np.float32(1.0) / (np.float32(1.0) + np.exp(-z))
            loss_grad = p - label
            g_w1 += (loss_grad * x1) / n_samples
            g_w2 += (loss_grad * x2) / n_samples
            g_b += loss_grad / n_samples

        corr1 = 1.0 - beta1.value ** t
        corr2 = 1.0 - beta2.value ** t
        host_m[0] = beta1.value * host_m[0] + (1.0 - beta1.value) * g_w1
        host_m[1] = beta1.value * host_m[1] + (1.0 - beta1.value) * g_w2
        host_m[2] = beta1.value * host_m[2] + (1.0 - beta1.value) * g_b
        host_v[0] = beta2.value * host_v[0] + (1.0 - beta2.value) * (g_w1 * g_w1)
        host_v[1] = beta2.value * host_v[1] + (1.0 - beta2.value) * (g_w2 * g_w2)
        host_v[2] = beta2.value * host_v[2] + (1.0 - beta2.value) * (g_b * g_b)
        m_hat_w1 = host_m[0] / corr1
        v_hat_w1 = host_v[0] / corr2
        m_hat_w2 = host_m[1] / corr1
        v_hat_w2 = host_v[1] / corr2
        m_hat_b = host_m[2] / corr1
        v_hat_b = host_v[2] / corr2
        host_params[0] = np.float32(
            host_params[0]
            - lr.value * (m_hat_w1 / ((v_hat_w1 ** 0.5) + eps.value))
        )
        host_params[1] = np.float32(
            host_params[1]
            - lr.value * (m_hat_w2 / ((v_hat_w2 ** 0.5) + eps.value))
        )
        host_params[2] = np.float32(
            host_params[2]
            - lr.value * (m_hat_b / ((v_hat_b ** 0.5) + eps.value))
        )

    result = [float(params_ptr[i]) for i in range(3)]
    print("Kernel result:", result)
    print("Host result:", host_params)
    print(
        f"Decision boundary: {host_params[0]:.3f} * x1 + {host_params[1]:.3f} * x2 + {host_params[2]:.3f} = 0"
    )

    if stop_api:
        stop_api()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="2D logistic regression with Adam")
    parser.add_argument("--api", action="store_true", help="start API server while running")
    args = parser.parse_args()
    main(with_api=args.api)
