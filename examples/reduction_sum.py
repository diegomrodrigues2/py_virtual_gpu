import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import VirtualGPU
from py_virtual_gpu.services import get_gpu_manager
from py_virtual_gpu.api.server import start_background_api
from py_virtual_gpu.kernel import kernel
from py_virtual_gpu.thread import get_current_thread


@kernel
def reduce_sum_kernel(threadIdx, blockIdx, blockDim, gridDim, in_ptr, out_ptr):
    ctx = get_current_thread()
    shared_mem = ctx.shared_mem
    barrier = ctx.barrier
    tx = threadIdx[0]
    shared_mem.write(tx * 4, in_ptr[tx])
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


def host_reduce(vals):
    total = 0
    for v in vals:
        total += v
    return total


def main(with_api: bool = False) -> None:
    if with_api:
        api_thread, stop_api = start_background_api()
    else:
        api_thread = stop_api = None

    values = [1, 2, 3, 4, 5, 6, 7, 8]
    gpu = VirtualGPU(num_sms=0, global_mem_size=128, shared_mem_size=len(values) * 4)
    get_gpu_manager().add_gpu(gpu)
    VirtualGPU.set_current(gpu)
    in_bytes = b"".join(v.to_bytes(4, "little", signed=True) for v in values)

    in_ptr = gpu.malloc(len(in_bytes))
    out_ptr = gpu.malloc(4)

    gpu.memcpy_host_to_device(in_bytes, in_ptr)

    reduce_sum_kernel(
        in_ptr,
        out_ptr,
        grid_dim=(1, 1, 1),
        block_dim=(len(values), 1, 1),
    )
    gpu.synchronize()

    out = gpu.memcpy_device_to_host(out_ptr, 4)
    result = int.from_bytes(out, "little", signed=True)
    expected = host_reduce(values)

    print("Kernel result:", result)
    print("Host result:", expected)
    if result == expected:
        print("Reduction successful!")
    else:
        print("Mismatch detected")

    if stop_api:
        stop_api()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vector reduction example")
    parser.add_argument("--api", action="store_true", help="start API server while running")
    args = parser.parse_args()
    main(with_api=args.api)
