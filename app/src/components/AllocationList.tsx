import React, { useEffect, useState } from 'react';
import { AllocationRecord, MemorySlice } from '../types/types';
import { fetchAllocations, fetchGlobalMemorySlice } from '../services/gpuSimulatorService';

interface AllocationListProps {
  gpuId: string;
  onSelect: (slice: MemorySlice) => void;
}

export const AllocationList: React.FC<AllocationListProps> = ({ gpuId, onSelect }) => {
  const [allocations, setAllocations] = useState<AllocationRecord[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    fetchAllocations(gpuId)
      .then((allocs) => {
        if (mounted) setAllocations(allocs);
      })
      .catch((err) => console.error('Failed to fetch allocations', err))
      .finally(() => setLoading(false));
    return () => {
      mounted = false;
    };
  }, [gpuId]);

  const dtypeMap: Record<string, 'half' | 'float32' | 'float64' | undefined> = {
    Half: 'half',
    Float32: 'float32',
    Float64: 'float64',
  };

  const handleClick = async (alloc: AllocationRecord) => {
    try {
      const dtype = alloc.dtype ? dtypeMap[alloc.dtype] : undefined;
      const slice = await fetchGlobalMemorySlice(gpuId, alloc.offset, alloc.size, dtype);
      onSelect(slice);
    } catch (err) {
      console.error('Failed to fetch memory slice', err);
    }
  };

  if (loading) {
    return <p className="text-xs text-gray-400">Loading allocations...</p>;
  }

  if (allocations.length === 0) {
    return <p className="text-xs text-gray-400">No active allocations.</p>;
  }

  return (
    <table className="text-xs w-full mt-2">
      <thead>
        <tr className="text-left">
          <th className="pr-2">Label</th>
          <th className="pr-2">Offset</th>
          <th className="pr-2">Size</th>
          <th>Dtype</th>
        </tr>
      </thead>
      <tbody>
        {allocations.map((a, idx) => (
          <tr
            key={idx}
            className="odd:bg-gray-800 cursor-pointer hover:bg-gray-700"
            onClick={() => handleClick(a)}
          >
            <td className="font-mono pr-2">{a.label ?? '-'}</td>
            <td className="pr-2">{a.offset}</td>
            <td className="pr-2">{a.size}</td>
            <td>{a.dtype ?? '-'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};
