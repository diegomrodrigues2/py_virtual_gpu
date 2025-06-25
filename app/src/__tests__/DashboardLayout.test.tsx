import { render, screen } from '@testing-library/react';
import React from 'react';
import { DashboardLayout } from '../App';
import { GPUState, GpuSummary } from '../types/types';

const gpu: GPUState = {
  id: '0',
  name: 'GPU 0',
  config: { num_sms: 1, global_mem_size: 1024, shared_mem_per_sm_kb: 0, registers_per_sm_total: 0 },
  global_memory: { used: 0, total: 1024 },
  transfers: { H2D: 0, D2H: 0, bytes_transferred: 0 },
  sms: [],
  overall_load: 0,
};

const summary: GpuSummary = {
  id: '0',
  name: 'GPU 0',
  globalMemoryUsed: 0,
  globalMemoryTotal: 1024,
  activeSMs: 0,
  totalSMs: 1,
  overallLoad: 0,
  status: 'online'
};

describe('DashboardLayout view switching', () => {
  it('shows cluster view when currentView is cluster', () => {
    render(
      <DashboardLayout
        gpuSummaries={[summary]}
        selectedGpu={gpu}
        allGpuStates={[gpu]}
        events={[]}
        onSelectGpu={() => {}}
        isLoading={false}
        currentView="cluster"
        onSetView={() => {}}
      />
    );
    expect(screen.getByText('GPU Cluster Overview')).toBeInTheDocument();
  });

  it('shows detail view when currentView is detail', () => {
    render(
      <DashboardLayout
        gpuSummaries={[summary]}
        selectedGpu={gpu}
        allGpuStates={[gpu]}
        events={[]}
        onSelectGpu={() => {}}
        isLoading={false}
        currentView="detail"
        onSetView={() => {}}
      />
    );
    expect(screen.getByText(/GPU 0 \(0\) - Details/)).toBeInTheDocument();
  });
});

