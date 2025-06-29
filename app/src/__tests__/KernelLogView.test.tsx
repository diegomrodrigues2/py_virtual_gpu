import { render, screen } from '@testing-library/react';
import React from 'react';
import { vi } from 'vitest';
import { KernelLogView } from '../components/KernelLogView';
import { KernelLaunchRecord } from '../types/types';
import * as service from '../services/gpuSimulatorService';

vi.mock('../services/gpuSimulatorService', () => ({
  fetchKernelLog: vi.fn(),
}));

const mockFetch = service.fetchKernelLog as unknown as ReturnType<typeof vi.fn>;

const log: KernelLaunchRecord[] = [
  {
    name: 'dummy',
    grid_dim: [1, 1, 1],
    block_dim: [2, 2, 1],
    start_cycle: 5,
    timestamp: '2025-06-29T12:00:00Z',
  },
];

describe('KernelLogView', () => {
  it('renders kernel log entries', async () => {
    mockFetch.mockResolvedValue(log);
    render(<KernelLogView gpuId="0" />);
    expect(await screen.findByText('dummy')).toBeInTheDocument();
    expect(screen.getByText('1x1x1')).toBeInTheDocument();
    expect(screen.getByText('2x2x1')).toBeInTheDocument();
    expect(screen.getByText('5')).toBeInTheDocument();
    const ts = new Date(log[0].timestamp).toLocaleString();
    expect(screen.getByText(ts)).toBeInTheDocument();
  });
});
