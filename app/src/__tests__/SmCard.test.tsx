import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';
import { vi } from 'vitest';
import { SmCard } from '../components/components';
import { StreamingMultiprocessorState, SMDetailed } from '../types/types';
import * as service from '../services/gpuSimulatorService';

vi.mock('../services/gpuSimulatorService', () => ({
  fetchSmDetail: vi.fn(),
}));

const mockFetch = service.fetchSmDetail as unknown as ReturnType<typeof vi.fn>;

const sm: StreamingMultiprocessorState = {
  id: 0,
  blocks_active: 0,
  blocks_pending: 0,
  warps_executed: 0,
  warp_divergences: 0,
  non_coalesced_accesses: 0,
  shared_mem_usage_kb: 0,
  shared_mem_total_kb: 0,
  registers_used: undefined,
  registers_total: undefined,
  bank_conflicts: 0,
  active_block_idx: undefined,
  status: 'idle',
  load_percentage: 0,
  active_warps: undefined,
};

const detail: SMDetailed = {
  id: 0,
  blocks: [],
  warps: [],
  divergence_log: [],
  counters: {},
};

describe('SmCard', () => {
  it('toggles detail view when button is clicked', async () => {
    mockFetch.mockResolvedValue(detail);
    render(<SmCard sm={sm} gpuId="0" />);

    const button = screen.getByRole('button', { name: /view details/i });
    await userEvent.click(button);
    expect(mockFetch).toHaveBeenCalledWith('0', '0');
    expect(await screen.findByText(/no blocks scheduled/i)).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /hide details/i }));
    expect(screen.queryByText(/no blocks scheduled/i)).not.toBeInTheDocument();
  });
});
