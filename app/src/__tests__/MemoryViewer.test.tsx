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
});
