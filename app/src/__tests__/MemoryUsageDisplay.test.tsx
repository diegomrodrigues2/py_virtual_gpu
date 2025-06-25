import { render, screen } from '@testing-library/react';
import React from 'react';
import { MemoryUsageDisplay } from '../components/components';

describe('MemoryUsageDisplay', () => {
  it('shows percentage and formatted values', () => {
    render(<MemoryUsageDisplay used={1024} total={2048} label="Global" />);
    expect(screen.getByText('Global')).toBeInTheDocument();
    expect(screen.getByText('50.0% Used')).toBeInTheDocument();
    expect(screen.getByText('1 KB / 2 KB')).toBeInTheDocument();
  });
});
