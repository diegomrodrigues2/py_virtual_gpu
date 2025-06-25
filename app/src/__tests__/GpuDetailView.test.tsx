import { render, screen } from '@testing-library/react';
import React from 'react';
import { GpuDetailView } from '../App';
import { GPUState } from '../types/types';

describe('GpuDetailView', () => {
  it('shows memory usage and sm cards', () => {
    const gpu: GPUState = {
      id: '0',
      name: 'GPU 0',
      config: {
        num_sms: 1,
        global_mem_size: 1024,
        shared_mem_per_sm_kb: 0,
        registers_per_sm_total: 0,
      },
      global_memory: { used: 512, total: 1024 },
      transfers: { H2D: 1, D2H: 0, bytes_transferred: 512 },
      sms: [
        {
          id: 0,
          blocks_active: 1,
          blocks_pending: 0,
          warps_executed: 10,
          warp_divergences: 0,
          non_coalesced_accesses: 0,
          shared_mem_usage_kb: 0,
          shared_mem_total_kb: 0,
          registers_used: undefined,
          registers_total: undefined,
          bank_conflicts: 0,
          active_block_idx: undefined,
          status: 'running',
          load_percentage: 80,
          active_warps: undefined,
        },
      ],
      overall_load: 50,
      temperature: 70,
      power_draw_watts: 100,
    };

    render(<GpuDetailView gpu={gpu} />);
    expect(screen.getByText('Global Memory')).toBeInTheDocument();
    expect(screen.getByText('SM 0')).toBeInTheDocument();
    expect(screen.getByText('50%')).toBeInTheDocument();
  });
});

