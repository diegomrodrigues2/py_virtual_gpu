import { GPUState, SimulatorEvent, GpuSummary, BackendData, StreamingMultiprocessorState } from './types';

const generateSMState = (smId: number, totalSMs: number, gpuId: string): StreamingMultiprocessorState => {
  const isActive = Math.random() > 0.3;
  const load = isActive ? Math.floor(Math.random() * 70 + 30) : Math.floor(Math.random() * 20);
  return {
    id: smId,
    blocks_active: isActive ? Math.floor(Math.random() * 2) + 1 : 0,
    blocks_pending: Math.floor(Math.random() * 5),
    warps_executed: Math.floor(Math.random() * 1000),
    warp_divergences: Math.floor(Math.random() * 50),
    non_coalesced_accesses: Math.floor(Math.random() * 100),
    shared_mem_usage_kb: parseFloat((Math.random() * 48).toFixed(1)),
    shared_mem_total_kb: 48,
    registers_used: Math.floor(Math.random() * 32768),
    registers_total: 65536,
    bank_conflicts: Math.floor(Math.random() * 10),
    active_block_idx: isActive ? `(${smId % 4},${Math.floor(smId / 4)},0)` : undefined,
    status: isActive ? (Math.random() > 0.8 ? 'waiting' : 'running') : 'idle',
    load_percentage: load,
  };
};

const generateGPUState = (gpuIndex: number): GPUState => {
  const numSMs = 8; //Math.random() > 0.5 ? 8 : 16;
  const globalMemTotalBytes = (Math.random() > 0.5 ? 1024 : 2048) * 1024 * 1024; // 1GB or 2GB
  const globalMemUsedBytes = Math.floor(Math.random() * globalMemTotalBytes * 0.8);
  const activeSMs = Math.floor(Math.random() * (numSMs + 1));
  const overallLoad = numSMs > 0 ? Math.floor((activeSMs / numSMs) * 70 + Math.random() * 30) : 0;

  return {
    id: `GPU${gpuIndex}`,
    name: `Simulated GPU ${String.fromCharCode(65 + gpuIndex)}`, // Alpha, Bravo, etc.
    config: {
      num_sms: numSMs,
      global_mem_size: globalMemTotalBytes,
      shared_mem_per_sm_kb: 48,
      registers_per_sm_total: 65536,
    },
    global_memory: {
      used: globalMemUsedBytes,
      total: globalMemTotalBytes,
      allocations: [{address: '0x0000', size: globalMemUsedBytes/2}, {address: '0x8000', size: globalMemUsedBytes/2}]
    },
    transfers: {
      H2D: Math.floor(Math.random() * 100),
      D2H: Math.floor(Math.random() * 80),
      bytes_transferred: Math.floor(Math.random() * 1024 * 1024 * 50), // 50MB
    },
    sms: Array.from({ length: numSMs }, (_, i) => generateSMState(i, numSMs, `GPU${gpuIndex}`)),
    overall_load: Math.min(100, overallLoad),
    temperature: parseFloat((Math.random() * 40 + 30).toFixed(1)), // 30-70C
    power_draw_watts: parseFloat((Math.random() * 150 + 50).toFixed(1)), // 50-200W
  };
};

const mockGPUStates: GPUState[] = Array.from({ length: 2 }, (_, i) => generateGPUState(i));

const mockEvents: SimulatorEvent[] = [
  { id: 'evt1', timestamp: new Date(Date.now() - 5000).toISOString(), type: 'KERNEL_LAUNCH', message: 'Kernel "vectorAdd" launched on GPU0.', gpuId: 'GPU0', details: { kernelName: 'vectorAdd', gridSize: '(4,1,1)', blockSize: '(256,1,1)'} },
  { id: 'evt2', timestamp: new Date(Date.now() - 4500).toISOString(), type: 'BLOCK_START', message: 'Block (0,0,0) started on SM0.', gpuId: 'GPU0', smId: 0, blockIdx: '(0,0,0)' },
  { id: 'evt3', timestamp: new Date(Date.now() - 3000).toISOString(), type: 'WARP_DIVERGENCE', message: 'Warp divergence detected in SM1, warp 3.', gpuId: 'GPU0', smId: 1, details: { warpId: 3, pc: '0x1A4' } },
  { id: 'evt4', timestamp: new Date(Date.now() - 2000).toISOString(), type: 'MEM_COPY_H2D', message: '512KB copied from Host to GPU1.', gpuId: 'GPU1', details: { size: '512KB' } },
  { id: 'evt5', timestamp: new Date(Date.now() - 1000).toISOString(), type: 'BLOCK_END', message: 'Block (0,0,0) finished on SM0.', gpuId: 'GPU0', smId: 0, blockIdx: '(0,0,0)' },
  { id: 'evt6', timestamp: new Date(Date.now() - 500).toISOString(), type: 'INFO_EVENT', message: 'GPU0 temperature stable at 55°C.', gpuId: 'GPU0' },
];


export const fetchBackendData = async (): Promise<BackendData> => {
  // Simulate API call delay
  await new Promise(resolve => setTimeout(resolve, 500));

  // Simulate dynamic data by regenerating states
  const currentGpuStates = Array.from({ length: Math.random() > 0.2 ? 2: 1 }, (_, i) => generateGPUState(i));
  const currentGpuSummaries: GpuSummary[] = currentGpuStates.map(gpu => ({
    id: gpu.id,
    name: gpu.name,
    globalMemoryUsed: gpu.global_memory.used,
    globalMemoryTotal: gpu.global_memory.total,
    activeSMs: gpu.sms.filter(sm => sm.status === 'running' || sm.status === 'waiting').length,
    totalSMs: gpu.config.num_sms,
    overallLoad: gpu.overall_load,
    status: Math.random() > 0.1 ? 'online' : 'error',
  }));
  
  // Add a new event occasionally
  const newEvents = [...mockEvents];
  if (Math.random() > 0.7) {
    newEvents.unshift({
      id: `evt${Date.now()}`,
      timestamp: new Date().toISOString(),
      type: 'INFO_EVENT',
      message: `Periodic health check for GPU${Math.floor(Math.random()*currentGpuStates.length)}.`,
      gpuId: `GPU${Math.floor(Math.random()*currentGpuStates.length)}`,
    });
  }
  
  return {
    gpuSummaries: currentGpuSummaries,
    gpuStates: currentGpuStates,
    events: newEvents.slice(0, 20), // Keep log somewhat limited
  };
};