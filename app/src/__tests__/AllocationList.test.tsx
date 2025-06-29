import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';
import { vi } from 'vitest';
import { AllocationList } from '../components/AllocationList';
import { AllocationRecord, MemorySlice } from '../types/types';
import * as service from '../services/gpuSimulatorService';

vi.mock('../services/gpuSimulatorService', () => ({
  fetchAllocations: vi.fn(),
  fetchGlobalMemorySlice: vi.fn(),
}));

const mockFetchAllocs = service.fetchAllocations as unknown as ReturnType<typeof vi.fn>;
const mockFetchSlice = service.fetchGlobalMemorySlice as unknown as ReturnType<typeof vi.fn>;

const allocs: AllocationRecord[] = [
  { offset: 0, size: 4, dtype: 'Float32', label: 'buf' },
];

const slice: MemorySlice = { offset: 0, size: 4, data: Buffer.from('0000', 'hex').toString('hex') };

describe('AllocationList', () => {
  it('lists allocations and fetches slice on click', async () => {
    mockFetchAllocs.mockResolvedValue(allocs);
    mockFetchSlice.mockResolvedValue(slice);
    const handleSelect = vi.fn();
    const user = userEvent.setup();
    render(<AllocationList gpuId="0" onSelect={handleSelect} />);

    expect(await screen.findByText('buf')).toBeInTheDocument();

    await user.click(screen.getByText('buf'));
    expect(mockFetchSlice).toHaveBeenCalledWith('0', 0, 4, 'float32');
    expect(handleSelect).toHaveBeenCalledWith(slice);
  });
});
