import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import VirtualGPU
from py_virtual_gpu.services import get_gpu_manager
from py_virtual_gpu.api.server import start_background_api
from py_virtual_gpu.kernel import kernel
from py_virtual_gpu.thread import get_current_thread


@kernel
def conv2d_kernel(threadIdx, blockIdx, blockDim, gridDim,
                  in_ptr, out_ptr, width, height):
    ctx = get_current_thread()
    shared_mem = ctx.shared_mem
    barrier = ctx.barrier
    tx, ty, _ = threadIdx
    idx = ty * width + tx
    # load element to shared memory
    val = int.from_bytes(in_ptr[idx], "little", signed=True)
    shared_mem.write(idx * 4, val.to_bytes(4, "little", signed=True))
    barrier.wait()

    if 0 < tx < width - 1 and 0 < ty < height - 1:
        acc = 0
        for j in range(3):
            for i in range(3):
                coeff_bytes = VirtualGPU.get_current().read_constant((j * 3 + i) * 4, 4)
                coeff = int.from_bytes(coeff_bytes, "little", signed=True)
                sx = tx + i - 1
                sy = ty + j - 1
                s_idx = sy * width + sx
                val = int.from_bytes(shared_mem.read(s_idx * 4, 4), "little", signed=True)
                acc += val * coeff
        out_ptr[idx] = acc.to_bytes(4, "little", signed=True)
    else:
        out_ptr[idx] = (0).to_bytes(4, "little", signed=True)


def host_convolution(inp, kernel):
    h = len(inp)
    w = len(inp[0])
    pad = len(kernel) // 2
    out = [[0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if pad <= x < w - pad and pad <= y < h - pad:
                acc = 0
                for j in range(len(kernel)):
                    for i in range(len(kernel[0])):
                        acc += inp[y + j - pad][x + i - pad] * kernel[j][i]
                out[y][x] = acc
    return out


def main(with_api: bool = False) -> None:
    if with_api:
        api_thread, stop_api = start_background_api()
    else:
        api_thread = stop_api = None

    width = height = 4
    gpu = VirtualGPU(
        num_sms=0,
        global_mem_size=256,
        shared_mem_size=width * height * 4,
    )
    get_gpu_manager().add_gpu(gpu)
    VirtualGPU.set_current(gpu)

    input_matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]
    kernel = [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ]
    filter_flat = [v for row in kernel for v in row]
    const_data = b"".join(int(v).to_bytes(4, "little", signed=True) for v in filter_flat)
    gpu.set_constant(const_data)

    in_bytes = b"".join(int(v).to_bytes(4, "little", signed=True) for row in input_matrix for v in row)
    in_ptr = gpu.malloc(len(in_bytes))
    out_ptr = gpu.malloc(len(in_bytes))
    gpu.memcpy_host_to_device(in_bytes, in_ptr)

    conv2d_kernel(
        in_ptr,
        out_ptr,
        width,
        height,
        grid_dim=(1, 1, 1),
        block_dim=(width, height, 1),
    )
    gpu.synchronize()

    out = gpu.memcpy_device_to_host(out_ptr, len(in_bytes))
    result = [int.from_bytes(out[i * 4:(i + 1) * 4], "little", signed=True) for i in range(width * height)]
    expected_matrix = host_convolution(input_matrix, kernel)
    expected = [v for row in expected_matrix for v in row]

    print("Kernel result:", result)
    print("Host result:", expected)
    if result == expected:
        print("2D convolution successful!")
    else:
        print("Mismatch detected")

    if stop_api:
        stop_api()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="2D convolution example")
    parser.add_argument("--api", action="store_true", help="start API server while running")
    args = parser.parse_args()
    main(with_api=args.api)
