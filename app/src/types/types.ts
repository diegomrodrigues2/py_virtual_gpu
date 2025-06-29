
export interface GPUConfig {
  num_sms: number;
  global_mem_size: number; // in bytes
  shared_mem_per_sm_kb: number; 
  registers_per_sm_total: number; 
}

export interface GlobalMemoryState {
  used: number; // in bytes
  total: number; // in bytes
  allocations?: { address: number | string; size: number }[];
}

export interface TransfersState {
  H2D: number; // count
  D2H: number; // count
  InterGPU?: number; // count
  bytes_transferred: number;
  cycles_transferred?: number; 
}

export interface StreamingMultiprocessorState {
  id: number;
  blocks_active: number;
  blocks_pending: number;
  warps_executed: number;
  warp_divergences: number;
  non_coalesced_accesses: number;
  shared_mem_usage_kb: number; 
  shared_mem_total_kb: number; 
  registers_used?: number;
  registers_total?: number;
  bank_conflicts: number;
  barrier_wait_ms: number;
  active_block_idx?: string;
  status: 'running' | 'idle' | 'waiting' | 'error';
  load_percentage?: number; // 0-100
  active_warps?: { id: number; active_threads: number; total_threads: number; pc?: number; status?: string }[];
}

export interface BlockSummary {
  block_idx: [number, number, number];
  status: string;
}

export interface WarpSummary {
  id: number;
  active_threads: number;
}

export interface DivergenceRecord {
  warp_id: number;
  pc: number;
  mask_before: boolean[];
  mask_after: boolean[];
}

export interface BlockEventRecord {
  block_idx: [number, number, number];
  sm_id: number;
  phase: string;
  start_cycle: number;
}

export interface KernelLaunchRecord {
  name: string;
  grid_dim: [number, number, number];
  block_dim: [number, number, number];
  start_cycle: number;
}

export interface MemorySlice {
  offset: number;
  size: number;
  data: string; // hex-encoded
  values?: number[];
}

export interface AllocationRecord {
  offset: number;
  size: number;
  dtype?: string | null;
  label?: string | null;
}

export interface SMDetailed {
  id: number;
  blocks: BlockSummary[];
  warps: WarpSummary[];
  divergence_log: DivergenceRecord[];
  counters: Record<string, number>;
  block_event_log: BlockEventRecord[];
}

export interface GPUState {
  id: string; 
  name: string; // e.g., "Simulated GPU Alpha"
  config: GPUConfig;
  global_memory: GlobalMemoryState;
  transfers: TransfersState;
  sms: StreamingMultiprocessorState[];
  overall_load: number; // 0-100%
  communication_log?: { from_gpu: string; to_gpu: string; data_size_bytes: number; timestamp: string }[];
  temperature?: number; // Conceptual temperature
  power_draw_watts?: number; // Conceptual power draw
}

export interface SimulatorEvent {
  id: string;
  timestamp: string;
  type: 'KERNEL_LAUNCH' | 'BLOCK_START' | 'BLOCK_END' | 'WARP_DIVERGENCE' | 'MEM_COPY_H2D' | 'MEM_COPY_D2H' | 'MEM_COPY_INTERGPU' | 'SYNC_EVENT' | 'ERROR_EVENT' | 'INFO_EVENT';
  message: string;
  gpuId?: string;
  smId?: number;
  blockIdx?: string; 
  details?: Record<string, any>;
}

export interface GpuSummary {
  id: string;
  name: string;
  globalMemoryUsed: number;
  globalMemoryTotal: number;
  activeSMs: number;
  totalSMs: number;
  overallLoad: number; // Percentage
  status: 'online' | 'offline' | 'error';
}

export interface BackendData {
  gpuSummaries: GpuSummary[];
  gpuStates: GPUState[];
  events: SimulatorEvent[];
}