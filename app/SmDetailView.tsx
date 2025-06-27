import React from 'react';
import { SMDetailed } from '../types/types';

export const SmDetailView: React.FC<{ sm: SMDetailed }> = ({ sm }) => (
  <div className="mt-2 text-sm">
    <div className="mb-2">
      <h5 className="font-semibold text-sky-300">Blocks</h5>
      {sm.blocks.length === 0 ? (
        <p className="text-gray-400">No blocks scheduled.</p>
      ) : (
        <ul className="list-disc list-inside">
          {sm.blocks.map((b) => (
            <li key={b.block_idx.join(',')}>
              Block [{b.block_idx.join(', ')}] - {b.status}
            </li>
          ))}
        </ul>
      )}
    </div>
    <div>
      <h5 className="font-semibold text-sky-300">Warps</h5>
      {sm.warps.length === 0 ? (
        <p className="text-gray-400">No warps queued.</p>
      ) : (
        <ul className="list-disc list-inside">
          {sm.warps.map((w) => (
            <li key={w.id}>Warp {w.id}: {w.active_threads} threads active</li>
          ))}
        </ul>
      )}
    </div>
  </div>
);
