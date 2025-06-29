import React from 'react';
import { MemorySlice } from '../types/types';

export const MemoryViewer: React.FC<{ slice: MemorySlice }> = ({ slice }) => {
  const bytes: number[] = [];
  for (let i = 0; i < slice.data.length; i += 2) {
    bytes.push(parseInt(slice.data.substr(i, 2), 16));
  }
  const rowSize = 16;
  const rows: number[][] = [];
  for (let i = 0; i < bytes.length; i += rowSize) {
    rows.push(bytes.slice(i, i + rowSize));
  }

  const toAscii = (b: number) => (b >= 32 && b <= 126 ? String.fromCharCode(b) : '.');

  return (
    <div className="overflow-auto">
      <table className="font-mono text-xs border-collapse">
        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx}>
              <td className="pr-2 text-gray-400">
                {(slice.offset + idx * rowSize).toString(16).padStart(8, '0')}
              </td>
              {row.map((b, i) => (
                <td key={i} className="px-1">
                  {b.toString(16).padStart(2, '0')}
                </td>
              ))}
              <td className="pl-4 text-gray-500">{row.map(toAscii).join('')}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {slice.values && (
        <div className="mt-2 text-xs font-mono">
          <strong>Decoded:</strong> {slice.values.join(', ')}
        </div>
      )}
    </div>
  );
};
