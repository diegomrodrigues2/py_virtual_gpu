# Class Structure

This document summarizes the responsibilities and basic interface of the main classes in the virtual GPU. For an overview of how programming model elements map to the simulated structures see **Table 2** in [RESEARCH.md](../RESEARCH.md).

Implementation details of these classes and their APIs are addressed in issues **2.1 through 3.3** in the repository.

## `VirtualGPU`

**Responsibility:** represents the full parallel device containing the `StreamingMultiprocessor`s (SMs) and global memory. It coordinates kernel launches and manages memory.

**Key attributes**
- `sms`: list of available simulated SMs.
- `global_memory`: memory area accessible by all blocks and threads.

**Main methods**
- `launch_kernel(func, grid_dim, block_dim, *args)`: divides the grid into blocks, schedules them on the SMs and passes the arguments to each thread.
- `malloc(size) / free(ptr)`: interface for allocating and freeing space in `global_memory`.
- `malloc(size, dtype)` or `malloc_type(count, dtype)` return typed pointers that handle elements of the given numeric type (`Half`, `Float32`, `Float64`).
- `memcpy_host_to_device(...)` and `memcpy_device_to_host(...)`: simulate transfers between CPU and GPU.

## `StreamingMultiprocessor`

**Responsibility:** simulates an SM, scheduling `ThreadBlock`s and managing the execution of threads in each block.

**Key attributes**
- `blocks`: list of executing `ThreadBlock`s.
- `shared_memory`: fast on-chip memory used by executing blocks.

**Main methods**
- `execute_block(block)`: runs a `ThreadBlock`, controlling internal synchronization and thread progress.
- `schedule(blocks)`: distribution logic for blocks assigned to this SM.

## `ThreadBlock`

**Responsibility:** represents a group of threads that share memory and can synchronize.

**Key attributes**
- `threads`: collection of `Thread`s belonging to the block.
- `shared_memory`: memory area accessible only by threads in this block.
- `barrier`: synchronization mechanism (e.g. `multiprocessing.Barrier`).

**Main methods**
- `run()`: starts execution of all threads in the block.
- `syncthreads()`: makes all threads wait on the barrier, mimicking `__syncthreads()`.

## `Thread`

**Responsibility:** smallest execution unit of the kernel, containing private registers and the thread and block indices.

**Key attributes**
- `registers`: private memory of each thread.
- `local_mem`: `LocalMemory` area for large local variables and register spill.
- `thread_idx` and `block_idx`: indices identifying this thread's position in the grid.

**Main methods**
- `execute(kernel_func, *args)`: runs the kernel function with the provided arguments.
- `alloc_local(size)`: reserves space in `local_mem` for kernel use.
- `read_write_memory(...)`: utility operations to access global or shared memory as needed.

## `@kernel` decorator and context

These utilities facilitate dispatching kernels written in Python.

- `kernel`: decorator that creates a ``KernelFunction`` object from the original function.
- `VirtualGPU.set_current(gpu)` and `VirtualGPU.get_current()`: set and retrieve the active device that will receive executions.
- `KernelFunction.__call__`: invokes ``VirtualGPU.launch_kernel`` using ``grid_dim`` and ``block_dim``.

## `Instruction` and `SIMTStack`

Support components for the SIMT model and divergence handling.

- `Instruction`: simplified instruction with opcode and operands.
- `SIMTStack`: keeps thread masks and reconvergence PCs while a warp executes.
