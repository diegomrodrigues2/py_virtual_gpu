import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';
import { vi } from 'vitest';
import { GpuDetailView } from '../App';
import { GPUState, KernelLaunchRecord, MemorySlice } from '../types/types';
import * as service from '../services/gpuSimulatorService';

vi.mock('../services/gpuSimulatorService', () => ({
  fetchKernelLog: vi.fn(),
  fetchGlobalMemorySlice: vi.fn(),
  fetchConstantMemorySlice: vi.fn(),
}));

const mockFetch = service.fetchKernelLog as unknown as ReturnType<typeof vi.fn>;
const mockFetchGlobalSlice = service.fetchGlobalMemorySlice as unknown as ReturnType<typeof vi.fn>;
const mockFetchConstantSlice = service.fetchConstantMemorySlice as unknown as ReturnType<typeof vi.fn>;

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

  it('toggles kernel log visibility when button is clicked', async () => {
    const user = userEvent.setup();
    const gpu: GPUState = {
      id: '0',
      name: 'GPU 0',
      config: {
        num_sms: 0,
        global_mem_size: 1024,
        shared_mem_per_sm_kb: 0,
        registers_per_sm_total: 0,
      },
      global_memory: { used: 0, total: 1024 },
      transfers: { H2D: 0, D2H: 0, bytes_transferred: 0 },
      sms: [],
      overall_load: 0,
    };

    const log: KernelLaunchRecord[] = [
      { name: 'dummy', grid_dim: [1, 1, 1], block_dim: [1, 1, 1], start_cycle: 0 },
    ];

    mockFetch.mockResolvedValue(log);

    render(<GpuDetailView gpu={gpu} />);

    const button = screen.getByRole('button', { name: /show kernel log/i });
    await user.click(button);

    expect(mockFetch).toHaveBeenCalledWith('0');
    expect(await screen.findByText('dummy')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /hide kernel log/i }));
    expect(screen.queryByText('dummy')).not.toBeInTheDocument();
  });

  it('fetches and displays memory slice', async () => {
    const user = userEvent.setup();
    const gpu: GPUState = {
      id: '0',
      name: 'GPU 0',
      config: {
        num_sms: 0,
        global_mem_size: 1024,
        shared_mem_per_sm_kb: 0,
        registers_per_sm_total: 0,
      },
      global_memory: { used: 0, total: 1024 },
      transfers: { H2D: 0, D2H: 0, bytes_transferred: 0 },
      sms: [],
      overall_load: 0,
    };

    const slice: MemorySlice = { offset: 4, size: 1, data: Buffer.from('A').toString('hex') };

    mockFetchConstantSlice.mockResolvedValue(slice);

    render(<GpuDetailView gpu={gpu} />);

    await user.selectOptions(screen.getByRole('combobox'), 'constant');
    await user.clear(screen.getByPlaceholderText('Offset'));
    await user.type(screen.getByPlaceholderText('Offset'), '4');
    await user.clear(screen.getByPlaceholderText('Size'));
    await user.type(screen.getByPlaceholderText('Size'), '1');
    await user.click(screen.getByRole('button', { name: /fetch/i }));

    expect(mockFetchConstantSlice).toHaveBeenCalledWith('0', 4, 1);
    expect(await screen.findByText('41')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /clear/i }));
    expect(screen.queryByText('41')).not.toBeInTheDocument();
  });
});

