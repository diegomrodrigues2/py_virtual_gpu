import React from 'react';
import { SMDetailed } from '../types/types';

export const SmDetailView: React.FC<{ sm: SMDetailed }> = ({ sm }) => (
  <div className="text-sm space-y-4">
    <div>
      <h5 className="font-semibold text-sky-300 mb-2 flex items-center">
        <span className="w-2 h-2 bg-sky-400 rounded-full mr-2"></span>
        Blocks ({sm.blocks.length})
      </h5>
      {sm.blocks.length === 0 ? (
        <p className="text-gray-400 ml-4">No blocks scheduled.</p>
      ) : (
        <div className="ml-4 space-y-1">
          {sm.blocks.map((b) => (
            <div key={b.block_idx.join(',')} className="flex items-center justify-between bg-gray-600 px-3 py-1 rounded">
              <span>Block [{b.block_idx.join(', ')}]</span>
              <span className={`px-2 py-0.5 text-xs rounded ${
                b.status === 'running' ? 'bg-green-600' : 
                b.status === 'pending' ? 'bg-yellow-600' : 'bg-gray-500'
              }`}>
                {b.status}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
    
    <div>
      <h5 className="font-semibold text-sky-300 mb-2 flex items-center">
        <span className="w-2 h-2 bg-teal-400 rounded-full mr-2"></span>
        Warps ({sm.warps.length})
      </h5>
      {sm.warps.length === 0 ? (
        <p className="text-gray-400 ml-4">No warps queued.</p>
      ) : (
        <div className="ml-4 space-y-1">
          {sm.warps.map((w) => (
            <div key={w.id} className="flex items-center justify-between bg-gray-600 px-3 py-1 rounded">
              <span>Warp {w.id}</span>
              <span className="text-teal-300">{w.active_threads} threads active</span>
            </div>
          ))}
        </div>
      )}
    </div>

    {sm.counters && Object.keys(sm.counters).length > 0 && (
      <div>
        <h5 className="font-semibold text-sky-300 mb-2 flex items-center">
          <span className="w-2 h-2 bg-purple-400 rounded-full mr-2"></span>
          Counters
        </h5>
        <div className="ml-4 grid grid-cols-2 gap-2 text-xs">
          {Object.entries(sm.counters).map(([key, value]) => (
            <div key={key} className="flex justify-between bg-gray-600 px-2 py-1 rounded">
              <span className="text-gray-300">{key}:</span>
              <span className="text-white font-medium">{value}</span>
            </div>
          ))}
        </div>
      </div>
    )}

    {sm.block_event_log && sm.block_event_log.length > 0 && (
      <div>
        <h5 className="font-semibold text-sky-300 mb-2 flex items-center">
          <span className="w-2 h-2 bg-blue-400 rounded-full mr-2"></span>
          Block Events
        </h5>
        <div className="ml-4 space-y-1 text-xs">
          {sm.block_event_log.map((ev, idx) => (
            <div key={idx} className="flex justify-between bg-gray-600 px-2 py-1 rounded">
              <span>Block [{ev.block_idx.join(', ')}] {ev.phase}</span>
              <span>Cycle {ev.start_cycle}</span>
            </div>
          ))}
        </div>
      </div>
    )}
  </div>
);
