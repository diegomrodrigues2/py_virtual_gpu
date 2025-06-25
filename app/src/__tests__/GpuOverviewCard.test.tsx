import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';
import { vi } from 'vitest';
import { GpuOverviewCard } from '../components/components';
import { GpuSummary } from '../types/types';

describe('GpuOverviewCard', () => {
  it('calls onSelectGpu when clicked', async () => {
    const summary: GpuSummary = {
      id: '0',
      name: 'GPU 0',
      globalMemoryUsed: 0,
      globalMemoryTotal: 1024,
      activeSMs: 1,
      totalSMs: 1,
      overallLoad: 0,
      status: 'online'
    };
    const onSelect = vi.fn();
    render(<GpuOverviewCard summary={summary} onSelectGpu={onSelect} isSelected={false} />);
    await userEvent.click(screen.getByRole('button'));
    expect(onSelect).toHaveBeenCalledWith('0');
  });
});
