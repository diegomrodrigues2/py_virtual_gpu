import { render, screen } from '@testing-library/react';
import React from 'react';
import { MemoryViewer } from '../components/MemoryViewer';
import { MemorySlice } from '../types/types';

describe('MemoryViewer', () => {
  it('renders hex and ascii view', () => {
    const slice: MemorySlice = { offset: 0, size: 4, data: Buffer.from('test').toString('hex') };
    render(<MemoryViewer slice={slice} />);
    expect(screen.getByText('00000000')).toBeInTheDocument();
    expect(screen.getAllByText('74').length).toBeGreaterThan(0);
    expect(screen.getByText('test')).toBeInTheDocument();
  });

  it('shows decoded numeric values when provided', () => {
    const slice: MemorySlice = { offset: 0, size: 4, data: Buffer.from('0100', 'hex').toString('hex'), values: [1] };
    render(<MemoryViewer slice={slice} />);
    expect(screen.getByText('Decoded:')).toBeInTheDocument();
    expect(screen.getByText('1')).toBeInTheDocument();
  });
});
