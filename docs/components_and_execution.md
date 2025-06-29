# Components and Execution

This document provides a brief overview of the main components of the virtual GPU and how they interact during a kernel run.

## VirtualGPU

The `VirtualGPU` object represents the whole device. It aggregates multiple `StreamingMultiprocessor` (SM) units and the `GlobalMemory`. Its role is to divide the execution grid into `ThreadBlock`s and distribute them among the SMs.

## StreamingMultiprocessor

Each `StreamingMultiprocessor` has its own `SharedMemory` and a queue of `ThreadBlock`s. An SM executes blocks sequentially or in round-robin, creating `Warp`s to drive the `Thread`s in lock-step. The `dispatch()` method issues one instruction per warp each "cycle" and requeues those still active. The counter `warps_executed` is incremented on every issue.

## ThreadBlock

A `ThreadBlock` is a group of threads that share a region of `SharedMemory` and can synchronize via a barrier. When dispatched to an SM the block instantiates its threads and starts kernel execution.

## Thread and Warp

`Thread` is the smallest execution unit. It has private registers and references to the shared and global memories. Threads are grouped into `Warp`s to simulate the SIMT model. The warp controls the active mask of threads and handles possible control flow divergence.

## GlobalMemory and SharedMemory

`GlobalMemory` is accessible by all SMs and blocks. It is implemented with `multiprocessing.Array` so processes can share it. `SharedMemory` is limited to a block or SM and is used for fast thread communication.

## Memory Hierarchy

The different levels are represented by subclasses of ``MemorySpace``. Each has conceptual latency and bandwidth counters stored in ``stats`` whenever ``read`` or ``write`` are called. These include ``RegisterFile`` (private registers), ``SharedMemory`` (on-chip), ``L1Cache``/``L2Cache`` (caches), ``GlobalMemorySpace`` and the specialized ``ConstantMemory`` and ``LocalMemory``. ``reset_stats()`` clears the counters for new measurements.

### ConstantMemory

``ConstantMemory`` is read-only (``64 KiB`` by default) and shared by all threads. The host copies values with ``gpu.set_constant`` and kernels can read via ``thread.const_mem.read`` or ``VirtualGPU.get_current().read_constant``.

```python
gpu.set_constant(b"hello")                     # host -> constant memory
value = VirtualGPU.get_current().read_constant(0, 3)
```

## Execution Flow

1. The user calls `launch_kernel` on the `VirtualGPU` specifying the grid and block dimensions.
2. The virtual GPU creates the required `ThreadBlock`s and places them in the SM queues.
3. Each SM consumes its queue, dividing blocks into `Warp`s that execute in lock-step.
4. During execution threads access `SharedMemory` and `GlobalMemory` as defined by the kernel.
5. After all blocks finish execution results can be copied back to the host application.

## Memory and Execution Features

- `VirtualGPU.malloc` and `VirtualGPU.free` manage spaces in `GlobalMemory` using `DevicePointer`.
- `memcpy_host_to_device`, `memcpy_device_to_host` and `memcpy` copy data between CPU and GPU.
- Example copy:
```python
ptr = gpu.malloc(256)
gpu.memcpy_host_to_device(b"\x00" * 256, ptr)
data = gpu.memcpy_device_to_host(ptr, 256)
```
- Pointers returned by `malloc` are instances of `DevicePointer` that support arithmetic and indexing (`ptr + n`, `ptr[i]`, etc.), allowing syntax similar to CUDA C++. Below is a kernel that multiplies vectors using `ptr[i]`:
- You can pass a numeric type such as `Half`, `Float32` or `Float64` to `malloc` or use `malloc_type(count, dtype)` to obtain typed pointers. Elements read from those pointers return instances of the chosen type and operations between them automatically promote to the wider dtype.
- `malloc` also accepts a `label` string so allocations can be identified in API responses and in the dashboard.
```python
@kernel(grid_dim=(1, 1, 1), block_dim=(4, 1, 1))
def vec_mul(threadIdx, blockIdx, blockDim, gridDim, a_ptr, b_ptr, out_ptr):
    i = threadIdx[0]
    a = int.from_bytes(a_ptr[i], "little")
    b = int.from_bytes(b_ptr[i], "little")
    out_ptr[i] = (a * b).to_bytes(4, "little")
```
- The `@kernel` decorator turns Python functions into kernels and uses the device set by `VirtualGPU.set_current`.
- `launch_kernel` splits the grid into `ThreadBlock`s and distributes them across the SMs, exposing `threadIdx`, `blockIdx`, `blockDim` and `gridDim` to the kernel.
- `ThreadBlock.barrier_sync()` lets threads in a block wait on each other, mirroring CUDA's ``__syncthreads()``.
- The `SharedMemory` class exposes atomic operations (`atomic_add`, `atomic_sub`, `atomic_cas`, `atomic_max`, `atomic_min`, `atomic_exchange`) for safely updating shared values.

## Memory Fences

Three functions mimic CUDA memory fences to control write visibility:

- `threadfence_block()` ensures data written to a block's `SharedMemory` is visible to its other threads after the fence.
- `threadfence()` propagates updates to all memories within the device, covering both `SharedMemory` and `GlobalMemory`.
- `threadfence_system()` extends the effect so the host application can observe the writes; in the simulation it is equivalent to `threadfence()`.

In all cases these functions simply acquire and release the respective memory locks to emulate ordering.

## Atomic Operations

The library also provides high level helpers for performing atomic operations directly on ``DevicePointer`` in ``GlobalMemory``. The functions ``atomicAdd``, ``atomicSub``, ``atomicCAS``, ``atomicMax``, ``atomicMin`` and ``atomicExchange`` take the pointer and the desired value, returning the previous value.

```python
from py_virtual_gpu import kernel, atomicAdd

@kernel(grid_dim=(1, 1, 1), block_dim=(4, 1, 1))
def incr(threadIdx, blockIdx, blockDim, gridDim, counter_ptr):
    atomicAdd(counter_ptr, 1)
```

These helpers internally use ``GlobalMemory.atomic_*``. The original methods remain available directly via ``SharedMemory`` and ``GlobalMemory`` instances.

## Divergence Monitoring

With the new execution flow each warp advances in lock-step. The SM's `dispatch()` method selects a warp from the queue and calls `warp.execute()` to issue an instruction. During this step the instruction is fetched, branch predicates are evaluated for all threads and the ``SIMTStack`` is updated for reconvergence. Whenever the thread mask changes ``record_divergence`` stores a ``DivergenceEvent`` in ``divergence_log`` and increments ``counters['warp_divergences']``. The total instructions issued accumulate in ``counters['warps_executed']`` and any extra cycles from memory access are added to ``stats['extra_cycles']``.

The log can be consulted for later analysis. The example below builds a simple chart showing the cumulative number of divergences over the ``pc`` of each event:

```python
log = sm.get_divergence_log()
pcs = [e.pc for e in log]
divs = list(range(1, len(log) + 1))
plt.plot(pcs, divs)
plt.xlabel("PC")
plt.ylabel("Accumulated divergences")
plt.show()
print("Warps executed:", sm.counters["warps_executed"])
print("Recorded divergences:", sm.counters["warp_divergences"])
```

## Memory Access Patterns

The ``Warp.memory_access`` method can record whether a set of addresses is **coalesced** and if there are **bank conflicts** in ``SharedMemory``. When addresses are not contiguous ``counters['non_coalesced_accesses']`` increments and ``stats['extra_cycles']`` receives +1 conceptual cycle. Bank conflicts reported by ``SharedMemory.detect_bank_conflicts`` increase ``counters['bank_conflicts']`` and add ``conflicts - 1`` extra cycles.

```python
warp.memory_access([0, 8], 4)            # non-coalesced
warp.memory_access([0, 0], 4, "shared")  # bank conflict
coalescing = sm.report_coalescing_stats()
conflicts = sm.report_bank_conflict_stats()
print(coalescing)
print(conflicts)
```

## Register Spill

When a thread exceeds its ``RegisterFile`` capacity during a write the extra bytes are redirected to its private ``LocalMemory``. Each spill logs events and adds cycles computed from ``spill_granularity`` and ``spill_latency_cycles``. Statistics can be retrieved per thread with ``thread.get_spill_stats()`` or aggregated via ``VirtualGPU.get_memory_stats()``.

## Per-Thread Local Memory

``LocalMemory`` is private to each ``Thread`` and has the same latency as ``GlobalMemory``. It is used for large local variables and for automatic register spill. The region is limited and kernels can reserve sections with ``Thread.alloc_local``:

```python
@kernel(grid_dim=(1,1,1), block_dim=(1,1,1))
def example(threadIdx, blockIdx, blockDim, gridDim):
    off = thread.alloc_local(4)
    thread.local_mem.write(off, b"data")
    value = thread.local_mem.read(off, 4)
```

## Warp Operations

Two functions assist exchanging information between threads of the same warp. They use the block's shared barrier for synchronization.

- ``shfl_sync(value, src_lane)`` returns the value from the ``src_lane`` to all threads:

```python
from py_virtual_gpu import kernel, shfl_sync

@kernel(grid_dim=(1,1,1), block_dim=(4,1,1))
def copy_first(threadIdx, blockIdx, blockDim, gridDim, out):
    v = threadIdx[0]
    out[threadIdx[0]] = shfl_sync(v, 0)
```

- ``ballot_sync(predicate)`` gathers a predicate from each lane and returns a bit mask where each position represents a thread's result:

```python
@kernel(grid_dim=(1,1,1), block_dim=(4,1,1))
def masks(threadIdx, blockIdx, blockDim, gridDim, out):
    m = ballot_sync(threadIdx[0] % 2 == 0)
    out[threadIdx[0]] = m
```

## Barrier Synchronization

The ``syncthreads()`` method exposes the ``ThreadBlock`` barrier. All threads must reach the call for execution to continue. If only a subset calls ``syncthreads()`` the block enters deadlock. This usually happens when there is **warp divergence** inside a conditional branch. Always ensure code paths reconverge before the barrier or move the synchronization outside the divergent section.

### Reductions and Scans

Reduction and scan operations are classic examples that interleave writes to ``SharedMemory`` with barriers. The snippet below shows the pattern for a per-block sum reduction:

```python
from py_virtual_gpu import kernel, syncthreads
from py_virtual_gpu.thread import get_current_thread

@kernel(grid_dim=(1,1,1), block_dim=(8,1,1))
def reduce_sum(threadIdx, blockIdx, blockDim, gridDim, data):
    tx = threadIdx[0]
    shared = get_current_thread().shared_mem
    shared.write(tx * 4, data[tx])
    syncthreads()

    stride = blockDim[0] // 2
    while stride > 0:
        if tx < stride:
            a = int.from_bytes(shared.read(tx * 4, 4), "little")
            b = int.from_bytes(shared.read((tx + stride) * 4, 4), "little")
            shared.write(tx * 4, (a + b).to_bytes(4, "little"))
        syncthreads()
        stride //= 2
```

A prefix-sum (scan) follows a similar logic: each iteration reads from the previous position, writes the result and synchronizes so all threads see the intermediates before the next step.

### Atomics or Barriers?

Use atomic operations when multiple threads need to update the same address independently, such as incrementing a global counter. When values are accumulated cooperatively in ``SharedMemory`` prefer ``syncthreads()`` to synchronize each step. If results must be visible beyond the block also call ``threadfence_block()``, ``threadfence()`` or ``threadfence_system()`` depending on the memory scope.
