"""Top-level package for the virtual GPU simulator."""

from .virtualgpu import VirtualGPU
from .global_memory import GlobalMemory
from .memory import DevicePointer
from .streaming_multiprocessor import StreamingMultiprocessor, DivergenceEvent
from .thread_block import ThreadBlock
from .warp import Warp, is_coalesced
from .dispatch import Instruction, SIMTStack
from .transfer import TransferEvent
from .atomics import (
    atomicAdd,
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
    "LocalMemory",
    "HostMemory",
    "TransferEvent",
    "atomicAdd",
    "atomicSub",
    "atomicCAS",
    "atomicMax",
    "atomicMin",
    "atomicExchange",
]

