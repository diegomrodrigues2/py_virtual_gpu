### Project Goal

This project aims to develop a **full virtual GPU in Python** for educational purposes and deep learning about parallel computer architecture. The simulator faithfully reproduces the central components of a modern GPU (Streaming Multiprocessors, memory hierarchy, SIMT model) and its programming model (kernels, grids, blocks, threads) using `multiprocessing` to overcome Python's GIL and achieve real parallelism.

**Main Objectives:**
- **Hands-on Learning**: Demystify GPU computing through interactive functional simulation without expensive hardware.
- **Conceptual Accuracy**: Focus on functional correctness and fidelity to the CUDA/OpenCL programming model, prioritizing educational clarity over cycle-level precision.
- **Data Parallelism**: Demonstrate how workloads are distributed and processed in parallel, including multi-GPU scenarios.
- **Safe Environment**: Provide a controlled platform to experiment with GPU concepts without risking hardware.

The simulator allows users to concretely visualize how high-level abstractions (kernels, synchronization, memory hierarchy) translate into underlying hardware operations, bridging the gap between theory and practice in parallel programming.

---

## Issue Organization

### 1. Initial Structuring and Architecture

- **1.3** Outline of the Main Classes (VirtualGPU, SM, Block, Thread)

---

### 2. Basic Component Implementation

- **2.1** Implement the VirtualGPU class (basic structure)
- **2.2** Implement GlobalMemory using multiprocessing.Array
- **2.3** Implement the StreamingMultiprocessor (SM) class (basic structure)
- **2.4** Implement the ThreadBlock class (basic structure)
- **2.5** Implement the Thread class with RegisterMemory (private)
- **2.6** Implement SharedMemory inside ThreadBlock (internal communication)

---

### 3. Execution Model and API

- **2.7** Simulate the SIMT Model: warp groups and sequential/parallel execution
- **2.8** Instruction Dispatch Mechanism and Warp Divergence (conceptual)
- **3.1** Develop the Memory Management API (malloc, free)
- **3.2** Implement the Data Transfer API (memcpy_host_to_device, device_to_host)
- **3.3** Develop the @kernel decorator for Python functions
- **3.4** Implement Kernel Launch (launch_kernel) and Execution Context
- **3.5** Access Thread/Block Indices and Dimensions inside the Kernel

---

### 4. Parallel Execution and Synchronization

- **4.1** Integrate with multiprocessing.Pool for ThreadBlock scheduling
- **4.2** Implement Barrier Synchronization (multiprocessing.Barrier) in ThreadBlock
- **4.3** Implement Simulated Atomic Operations (with Lock/Value)
- **4.4** Refine the Warp Divergence Model (conceptual monitoring and visualization)

---

### 5. Multi-GPU and Scalability

- **5.1** Adapt for Multiple VirtualGPU Instances (Independent Processes)
- **5.2** Implement Data Distribution Strategies for Multi-GPU
- **5.3** Simulate Inter-GPU Communication (via Queue/Pipe)
- **5.5** Scalability Tests with Multiple SMs and VirtualGPUs

---

### 6. Support and Monitoring Tools

- **6.1** Implement Detailed Logging and Event Tracking System
- **6.2** Develop Conceptual Performance Counters (memory accesses, divergences)
- **6.3** Basic Debugging Tools (inspect memory/register state)

---

### 7. Validation with Educational Examples

- **7.1** Implement the vector multiplication example using indexable pointers
- **7.2** Implement the 2D convolution example using shared and constant memory
- **7.3** Include tests to ensure these examples run successfully
- **7.4** Create the `examples/` folder containing the example scripts

**Link to issues:**
https://github.com/diegomrodrigues2/py_virtual_gpu/issues
