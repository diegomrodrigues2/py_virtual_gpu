import { GPUState, SimulatorEvent, GpuSummary, BackendData, StreamingMultiprocessorState, GPUConfig, TransfersState } from '../types/types';

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

export const fetchBackendData = async (): Promise<BackendData> => {
  const gpuList = await fetchJSON<any[]>(`${API_BASE}/gpus`);
  const gpuStates: GPUState[] = [];
  const gpuSummaries: GpuSummary[] = [];

  for (const g of gpuList) {
    const state = await fetchJSON<any>(`${API_BASE}/gpus/${g.id}/state`);
    const sms = (state.sms || []).map(mapSmState);
    const transfers = aggregateTransfers(state.transfer_log || []);
    const config: GPUConfig = {
      num_sms: sms.length,
      global_mem_size: state.global_memory.size,
      shared_mem_per_sm_kb: 0,
      registers_per_sm_total: 0,
    };
    const gpuState: GPUState = {
      id: String(state.id),
      name: `GPU ${state.id}`,
      config,
      global_memory: {
        used: state.global_memory.used,
        total: state.global_memory.size,
      },
      transfers,
      sms,
      overall_load: sms.length > 0 ? Math.round((sms.filter(s => s.status !== 'idle').length / sms.length) * 100) : 0,
    };
    gpuStates.push(gpuState);
    gpuSummaries.push({
      id: gpuState.id,
      name: gpuState.name,
      globalMemoryUsed: gpuState.global_memory.used,
      globalMemoryTotal: gpuState.global_memory.total,
      activeSMs: sms.filter(s => s.status !== 'idle').length,
      totalSMs: sms.length,
      overallLoad: gpuState.overall_load,
      status: 'online',
    });
  }

  const events = await fetchJSON<SimulatorEvent[]>(`${API_BASE}/events`);

  return { gpuSummaries, gpuStates, events };
};
