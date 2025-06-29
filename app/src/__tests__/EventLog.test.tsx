import { render, screen } from '@testing-library/react';
import React from 'react';
import { EventLog } from '../components/components';
import { SimulatorEvent } from '../types/types';

const events: SimulatorEvent[] = [
  {
    id: '1',
    timestamp: '2025-06-29T12:00:00Z',
    type: 'KERNEL_LAUNCH',
    message: 'Kernel A launched',
    gpuId: '0'
  }
];

describe('EventLog', () => {
  it('renders event information', () => {
    render(<EventLog events={events} />);
    const ts = new Date(events[0].timestamp).toLocaleString();
    expect(screen.getByText(events[0].message)).toBeInTheDocument();
    expect(screen.getByText(ts)).toBeInTheDocument();
    expect(screen.getByText('KERNEL_LAUNCH')).toBeInTheDocument();
  });
});
