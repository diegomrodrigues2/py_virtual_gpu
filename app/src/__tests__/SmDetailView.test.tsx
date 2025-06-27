import { render, screen } from '@testing-library/react';
import React from 'react';
import { SmDetailView } from '../components/SmDetailView';
import { SMDetailed } from '../types/types';

describe('SmDetailView', () => {
  it('renders block and warp info', () => {
    const detail: SMDetailed = {
      id: 0,
      blocks: [{ block_idx: [0, 0, 0], status: 'pending' }],
      warps: [{ id: 1, active_threads: 32 }],
      divergence_log: [],
      counters: {},
      block_event_log: [
        { block_idx: [0, 0, 0], sm_id: 0, phase: 'start', start_cycle: 1 },
      ],
    };
    render(<SmDetailView sm={detail} />);
    expect(screen.getByText('Block [0, 0, 0]')).toBeInTheDocument();
    expect(screen.getByText('pending')).toBeInTheDocument();
    expect(screen.getByText('Warp 1')).toBeInTheDocument();
    expect(screen.getByText('32 threads active')).toBeInTheDocument();
    expect(screen.getByText(/Block \[0, 0, 0\] start/i)).toBeInTheDocument();
    expect(screen.getByText(/Cycle 1/)).toBeInTheDocument();
  });
});
