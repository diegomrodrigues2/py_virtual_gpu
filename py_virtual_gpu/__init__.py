"""Top-level package for the virtual GPU simulator."""

from .virtualgpu import VirtualGPU, KernelLaunchEvent
from .global_memory import GlobalMemory
from .memory import DevicePointer
from .streaming_multiprocessor import StreamingMultiprocessor, DivergenceEvent
from .thread_block import ThreadBlock
from .warp import Warp, is_coalesced
from .sync import syncthreads
from .fence import threadfence_block, threadfence, threadfence_system
from .warp_utils import shfl_sync, ballot_sync
from .types import (
    Half,
    Float32,
    Float64,
    sqrt_numeric,
    sin_numeric,
    cos_numeric,
    exp_numeric,
    log_numeric,
)
from .dispatch import Instruction, SIMTStack
from .transfer import TransferEvent
from .errors import SynchronizationError
from .atomics import (
    atomicAdd,
    atomicAdd_float32,
    atomicAdd_float64,
    atomicSub,
    atomicCAS,
    atomicMax,
    atomicMin,
    atomicExchange,
)
from .memory_hierarchy import (
    MemorySpace,
    RegisterFile,
    SharedMemory as HierSharedMemory,
    L1Cache,
    L2Cache,
    GlobalMemorySpace,
    ConstantMemory,
    LocalMemory,
    HostMemory,
)

const_memory = ConstantMemory
constant_memory = ConstantMemory

__all__ = [
    "VirtualGPU",
    "GlobalMemory",
    "StreamingMultiprocessor",
    "ThreadBlock",
    "Warp",
    "is_coalesced",
    "Instruction",
    "SIMTStack",
    "DevicePointer",
    "DivergenceEvent",
    "MemorySpace",
    "RegisterFile",
    "HierSharedMemory",
    "L1Cache",
    "L2Cache",
    "GlobalMemorySpace",
    "ConstantMemory",
    "const_memory",
    "constant_memory",
    "LocalMemory",
    "HostMemory",
    "TransferEvent",
    "KernelLaunchEvent",
    "SynchronizationError",
    "atomicAdd",
    "atomicAdd_float32",
    "atomicAdd_float64",
    "atomicSub",
    "atomicCAS",
    "atomicMax",
    "atomicMin",
    "atomicExchange",
    "syncthreads",
    "shfl_sync",
    "ballot_sync",
    "threadfence_block",
    "threadfence",
    "threadfence_system",
    "Half",
    "Float32",
    "Float64",
    "sqrt_numeric",
    "sin_numeric",
    "cos_numeric",
    "exp_numeric",
    "log_numeric",
]

