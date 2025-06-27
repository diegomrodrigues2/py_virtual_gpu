import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import VirtualGPU
from py_virtual_gpu.services import get_gpu_manager
from py_virtual_gpu.api.server import start_background_api
from py_virtual_gpu.thread_block import ThreadBlock


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
def reduce_partial_kernel(threadIdx, blockIdx, blockDim, gridDim,
                          in_ptr, partial_ptr, shared_mem, barrier, n):
    tx = threadIdx[0]
    idx = blockIdx[0] * blockDim[0] + tx
    if idx < n:
        shared_mem.write(tx * 4, in_ptr[idx])
    else:
        shared_mem.write(tx * 4, (0).to_bytes(4, "little", signed=True))
    barrier.wait()

    stride = blockDim[0] // 2
    while stride > 0:
        if tx < stride:
            a = int.from_bytes(shared_mem.read(tx * 4, 4), "little", signed=True)
            b = int.from_bytes(shared_mem.read((tx + stride) * 4, 4), "little", signed=True)
            shared_mem.write(tx * 4, (a + b).to_bytes(4, "little", signed=True))
        barrier.wait()
        stride //= 2

    if tx == 0:
        partial_ptr[blockIdx[0]] = shared_mem.read(0, 4)


def final_reduce_kernel(threadIdx, blockIdx, blockDim, gridDim,
                        partial_ptr, out_ptr, shared_mem, barrier, num_partials):
    tx = threadIdx[0]
    if tx < num_partials:
        shared_mem.write(tx * 4, partial_ptr[tx])
    else:
        shared_mem.write(tx * 4, (0).to_bytes(4, "little", signed=True))
    barrier.wait()

    stride = blockDim[0] // 2
    while stride > 0:
        if tx < stride:
            a = int.from_bytes(shared_mem.read(tx * 4, 4), "little", signed=True)
            b = int.from_bytes(shared_mem.read((tx + stride) * 4, 4), "little", signed=True)
            shared_mem.write(tx * 4, (a + b).to_bytes(4, "little", signed=True))
        barrier.wait()
        stride //= 2

    if tx == 0:
        out_ptr[0] = shared_mem.read(0, 4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def host_reduce(vals):
    total = 0
    for v in vals:
        total += v
    return total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(with_api: bool = False) -> None:
    if with_api:
        api_thread, stop_api = start_background_api()
    else:
        api_thread = stop_api = None

    gpu = VirtualGPU(num_sms=0, global_mem_size=512)
    get_gpu_manager().add_gpu(gpu)
    VirtualGPU.set_current(gpu)

    values = [1 for _ in range(32)]
    in_bytes = b"".join(v.to_bytes(4, "little", signed=True) for v in values)
    n = len(values)

    in_ptr = gpu.malloc(len(in_bytes))
    partial_ptr = gpu.malloc(16)  # 4 partial sums
    out_ptr = gpu.malloc(4)

    gpu.memcpy_host_to_device(in_bytes, in_ptr)

    grid_dim = (4, 1, 1)
    block_dim = (8, 1, 1)

    # Stage 1: compute partial sums
    for bx in range(grid_dim[0]):
        tb = ThreadBlock((bx, 0, 0), block_dim, grid_dim, shared_mem_size=block_dim[0] * 4)
        tb.initialize_threads(reduce_partial_kernel)
        for t in tb.threads:
            setattr(t, "global_mem", gpu.global_memory)
        tb.execute(reduce_partial_kernel, in_ptr, partial_ptr, tb.shared_mem, tb.barrier, n)

    # Stage 2: reduce partial sums
    tb_final = ThreadBlock((0, 0, 0), block_dim, (1, 1, 1), shared_mem_size=block_dim[0] * 4)
    tb_final.initialize_threads(final_reduce_kernel)
    for t in tb_final.threads:
        setattr(t, "global_mem", gpu.global_memory)
    tb_final.execute(final_reduce_kernel, partial_ptr, out_ptr, tb_final.shared_mem, tb_final.barrier, grid_dim[0])

    out = gpu.memcpy_device_to_host(out_ptr, 4)
    result = int.from_bytes(out, "little", signed=True)
    expected = host_reduce(values)

    print("Kernel result:", result)
    print("Host result:", expected)
    if result == expected:
        print("Hierarchical reduction successful!")
    else:
        print("Mismatch detected")

    if stop_api:
        stop_api()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hierarchical reduction example")
    parser.add_argument("--api", action="store_true", help="start API server while running")
    args = parser.parse_args()
    main(with_api=args.api)
