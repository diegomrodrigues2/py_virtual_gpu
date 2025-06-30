# Approximating sin(x) with a single hidden layer MLP using Adam optimizer.
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import Float32, VirtualGPU, atomicAdd_float32
from py_virtual_gpu.kernel import kernel
from py_virtual_gpu.optimizers import adam_step
from py_virtual_gpu.services import get_gpu_manager
from py_virtual_gpu.types import exp_numeric
from py_virtual_gpu.api.server import start_background_api


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def tanh_numeric(x: Float32) -> Float32:
    """Typed hyperbolic tangent."""
    two_x = Float32(2.0) * x
    e = exp_numeric(two_x)
    return (e - Float32(1.0)) / (e + Float32(1.0))


@kernel
def compute_grads(
    threadIdx,
    blockIdx,
    blockDim,
    gridDim,
    x_ptr,
    y_ptr,
    params_ptr,
    grad_ptr,
    n,
):
    i = blockIdx[0] * blockDim[0] + threadIdx[0]
    if i < n:
        w1 = params_ptr[0]
        b1 = params_ptr[1]
        w2 = params_ptr[2]

        x = x_ptr[i]
        y = y_ptr[i]

        h = tanh_numeric(w1 * x + b1)
        y_hat = w2 * h
        diff = y_hat - y

        grad_w2 = (diff * h) / Float32(n)
        grad_b1 = (diff * w2 * (Float32(1.0) - h * h)) / Float32(n)
        grad_w1 = grad_b1 * x

        atomicAdd_float32(grad_ptr, grad_w1)
        atomicAdd_float32(grad_ptr + 1, grad_b1)
        atomicAdd_float32(grad_ptr + 2, grad_w2)


# -----------------------------------------------------------------------------


def main(with_api: bool = False, plot: bool = False) -> None:
    if with_api:
        api_thread, stop_api = start_background_api()
    else:
        api_thread = stop_api = None

    np.random.seed(0)
    n_samples = 64
    x_vals = np.linspace(-np.pi, np.pi, n_samples, dtype=np.float32)
    y_vals = np.sin(x_vals).astype(np.float32)

    gpu = VirtualGPU(num_sms=0, global_mem_size=4096)
    get_gpu_manager().add_gpu(gpu)
    VirtualGPU.set_current(gpu)

    x_ptr = gpu.malloc(n_samples, dtype=Float32, label="x")
    y_ptr = gpu.malloc(n_samples, dtype=Float32, label="y")
    params_ptr = gpu.malloc(3, dtype=Float32, label="params")
    grad_ptr = gpu.malloc(3, dtype=Float32, label="grads")
    m_ptr = gpu.malloc(3, dtype=Float32, label="m")
    v_ptr = gpu.malloc(3, dtype=Float32, label="v")

    for i, v in enumerate(x_vals):
        x_ptr[i] = Float32(float(v))
    for i, v in enumerate(y_vals):
        y_ptr[i] = Float32(float(v))

    for i in range(3):
        params_ptr[i] = Float32(0.0)
        grad_ptr[i] = Float32(0.0)
        m_ptr[i] = Float32(0.0)
        v_ptr[i] = Float32(0.0)

    lr = Float32(0.01)
    beta1 = Float32(0.9)
    beta2 = Float32(0.999)
    eps = Float32(1e-8)
    epochs = 200

    host_params = [np.float32(0.0), np.float32(0.0), np.float32(0.0)]
    host_m = [np.float32(0.0), np.float32(0.0), np.float32(0.0)]
    host_v = [np.float32(0.0), np.float32(0.0), np.float32(0.0)]

    if plot:
        try:
            import matplotlib.pyplot as plt
        except Exception:  # pragma: no cover - optional dep
            plt = None
        if plt is not None:
            plt.ion()
            fig, ax = plt.subplots()
            ax.plot(x_vals, y_vals, label="sin(x)")
            line, = ax.plot([], [], label="approx")
            ax.legend()

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

        # host computation for verification
        g_w1 = np.float32(0.0)
        g_b1 = np.float32(0.0)
        g_w2 = np.float32(0.0)
        for i in range(n_samples):
            x = x_vals[i]
            y = y_vals[i]
            h = np.tanh(host_params[0] * x + host_params[1])
            y_hat = host_params[2] * h
            diff = y_hat - y
            grad_w2 = diff * h / n_samples
            grad_b1 = diff * host_params[2] * (1 - h * h) / n_samples
            grad_w1 = grad_b1 * x
            g_w1 += grad_w1
            g_b1 += grad_b1
            g_w2 += grad_w2

        corr1 = 1.0 - beta1.value ** t
        corr2 = 1.0 - beta2.value ** t
        host_m[0] = beta1.value * host_m[0] + (1.0 - beta1.value) * g_w1
        host_m[1] = beta1.value * host_m[1] + (1.0 - beta1.value) * g_b1
        host_m[2] = beta1.value * host_m[2] + (1.0 - beta1.value) * g_w2
        host_v[0] = beta2.value * host_v[0] + (1.0 - beta2.value) * (g_w1 * g_w1)
        host_v[1] = beta2.value * host_v[1] + (1.0 - beta2.value) * (g_b1 * g_b1)
        host_v[2] = beta2.value * host_v[2] + (1.0 - beta2.value) * (g_w2 * g_w2)
        m_hat_w1 = host_m[0] / corr1
        v_hat_w1 = host_v[0] / corr2
        m_hat_b1 = host_m[1] / corr1
        v_hat_b1 = host_v[1] / corr2
        m_hat_w2 = host_m[2] / corr1
        v_hat_w2 = host_v[2] / corr2
        host_params[0] = np.float32(
            host_params[0] - lr.value * (m_hat_w1 / (np.sqrt(v_hat_w1) + eps.value))
        )
        host_params[1] = np.float32(
            host_params[1] - lr.value * (m_hat_b1 / (np.sqrt(v_hat_b1) + eps.value))
        )
        host_params[2] = np.float32(
            host_params[2] - lr.value * (m_hat_w2 / (np.sqrt(v_hat_w2) + eps.value))
        )

        if plot and plt is not None and t % 50 == 0:
            preds = [
                host_params[2] * np.tanh(host_params[0] * xv + host_params[1])
                for xv in x_vals
            ]
            line.set_data(x_vals, preds)
            ax.set_title(f"Epoch {t}")
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.001)

    result = [float(params_ptr[i]) for i in range(3)]
    print("Kernel result:", result)
    print("Host result:", host_params)

    if stop_api:
        stop_api()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Approximate sin(x) with a single hidden layer MLP"
    )
    parser.add_argument("--api", action="store_true", help="start API server while running")
    parser.add_argument("--plot", action="store_true", help="plot approximation during training")
    args = parser.parse_args()
    main(with_api=args.api, plot=args.plot)
