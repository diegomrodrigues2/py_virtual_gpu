import {
  GPUState,
  SimulatorEvent,
  GpuSummary,
  BackendData,
  StreamingMultiprocessorState,
  GPUConfig,
  TransfersState,
  SMDetailed,
  MemorySlice,
  KernelLaunchRecord,
} from '../types/types';

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000';

async function fetchJSON<T>(url: string): Promise<T> {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Request failed: ${resp.status}`);
  }
  return resp.json() as Promise<T>;
}

function mapSmState(sm: any): StreamingMultiprocessorState {
  const counters = sm.counters || {};
  return {
    id: sm.id,
    blocks_active: counters.blocks_active ?? 0,
    blocks_pending: counters.blocks_pending ?? 0,
    warps_executed: counters.warps_executed ?? 0,
    warp_divergences: counters.warp_divergences ?? 0,
    non_coalesced_accesses: counters.non_coalesced_accesses ?? 0,
    shared_mem_usage_kb: 0,
    shared_mem_total_kb: 0,
    registers_used: undefined,
    registers_total: undefined,
    bank_conflicts: counters.bank_conflicts ?? 0,
    active_block_idx: undefined,
    status: (sm.status || 'idle') as 'running' | 'idle' | 'waiting' | 'error',
    load_percentage: undefined,
    active_warps: undefined,
  };
}

function aggregateTransfers(log: any[]): TransfersState {
  const transfers: TransfersState = { H2D: 0, D2H: 0, bytes_transferred: 0 };
  for (const t of log) {
    if (t.direction === 'H2D') transfers.H2D += 1;
    if (t.direction === 'D2H') transfers.D2H += 1;
    transfers.bytes_transferred += t.size || 0;
  }
  return transfers;
}

export const fetchGpuState = async (id: string): Promise<GPUState> => {
  const state = await fetchJSON<any>(`${API_BASE}/gpus/${id}/state`);
  const sms = (state.sms || []).map(mapSmState);
  const transfers = aggregateTransfers(state.transfer_log || []);
  const config: GPUConfig = {
    num_sms: sms.length,
    global_mem_size: state.global_memory.size,
    shared_mem_per_sm_kb: 0,
    registers_per_sm_total: 0,
  };

  return {
    id: String(state.id),
    name: `GPU ${state.id}`,
    config,
    global_memory: {
      used: state.global_memory.used,
      total: state.global_memory.size,
    },
    transfers,
    sms,
    overall_load:
      sms.length > 0
        ? Math.round(
            (sms.filter((s) => s.status !== 'idle').length / sms.length) * 100,
          )
        : 0,
  };
};

export const fetchSmDetail = async (
  gpuId: string,
  smId: string,
): Promise<SMDetailed> => {
  const detail = await fetchJSON<any>(`${API_BASE}/gpus/${gpuId}/sm/${smId}`);
  return {
    id: detail.id,
    blocks: detail.blocks ?? [],
    warps: detail.warps ?? [],
    divergence_log: detail.divergence_log ?? [],
    counters: detail.counters ?? {},
    block_event_log: detail.block_event_log ?? [],
  };
};

export const fetchBackendData = async (): Promise<BackendData> => {
  const gpuList = await fetchJSON<any[]>(`${API_BASE}/gpus`);
  const gpuStates: GPUState[] = [];
  const gpuSummaries: GpuSummary[] = [];

  for (const g of gpuList) {
    const gpuState = await fetchGpuState(String(g.id));
    gpuStates.push(gpuState);
    gpuSummaries.push({
      id: gpuState.id,
      name: gpuState.name,
      globalMemoryUsed: gpuState.global_memory.used,
      globalMemoryTotal: gpuState.global_memory.total,
      activeSMs: gpuState.sms.filter(s => s.status !== 'idle').length,
      totalSMs: gpuState.sms.length,
      overallLoad: gpuState.overall_load,
      status: 'online',
    });
  }

  const events = await fetchJSON<SimulatorEvent[]>(`${API_BASE}/events`);

  return { gpuSummaries, gpuStates, events };
};

export const fetchGlobalMemorySlice = async (
  gpuId: string,
  offset: number,
  size: number,
  dtype?: 'half' | 'float32' | 'float64',
): Promise<MemorySlice> => {
  const dtypeParam = dtype ? `&dtype=${dtype}` : '';
  return fetchJSON<MemorySlice>(
    `${API_BASE}/gpus/${gpuId}/global_mem?offset=${offset}&size=${size}${dtypeParam}`,
  );
};

export const fetchConstantMemorySlice = async (
  gpuId: string,
  offset: number,
  size: number,
  dtype?: 'half' | 'float32' | 'float64',
): Promise<MemorySlice> => {
  const dtypeParam = dtype ? `&dtype=${dtype}` : '';
  return fetchJSON<MemorySlice>(
    `${API_BASE}/gpus/${gpuId}/constant_mem?offset=${offset}&size=${size}${dtypeParam}`,
  );
};

export const fetchKernelLog = async (
  gpuId: string,
): Promise<KernelLaunchRecord[]> => {
  return fetchJSON<KernelLaunchRecord[]>(`${API_BASE}/gpus/${gpuId}/kernel_log`);
};

