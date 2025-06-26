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
    };
    render(<SmDetailView sm={detail} />);
    expect(screen.getByText(/Block \[0, 0, 0\] - pending/)).toBeInTheDocument();
    expect(screen.getByText(/Warp 1: 32 threads active/)).toBeInTheDocument();
  });
});
