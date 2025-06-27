import React, { useEffect, useState } from 'react';
import { KernelLaunchRecord } from '../types/types';
import { fetchKernelLog } from './gpuSimulatorService';

export const KernelLogView: React.FC<{ gpuId: string }> = ({ gpuId }) => {
  const [log, setLog] = useState<KernelLaunchRecord[]>([]);

  useEffect(() => {
    let isMounted = true;
    fetchKernelLog(gpuId)
      .then((entries) => {
        if (isMounted) setLog(entries);
      })
      .catch((err) => console.error('Failed to fetch kernel log', err));
    return () => {
      isMounted = false;
    };
  }, [gpuId]);

  if (log.length === 0) {
    return <p className="text-gray-400">No kernel launches recorded.</p>;
  }

  return (
    <table className="text-sm w-full mt-2">
      <thead>
        <tr className="text-left">
          <th className="pr-4">Name</th>
          <th className="pr-4">Grid</th>
          <th className="pr-4">Block</th>
          <th>Start Cycle</th>
        </tr>
      </thead>
      <tbody>
        {log.map((k, idx) => (
          <tr key={idx} className="odd:bg-gray-800">
            <td className="font-mono pr-4">{k.name}</td>
            <td className="pr-4">{k.grid_dim.join('x')}</td>
            <td className="pr-4">{k.block_dim.join('x')}</td>
            <td>{k.start_cycle}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};
