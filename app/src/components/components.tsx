import React, { useState } from 'react';
import { GPUState, SimulatorEvent, GpuSummary, StreamingMultiprocessorState, GlobalMemoryState, TransfersState } from '../types/types';

// --- Icons ---
export const IconChip: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" {...props}>
    <path fillRule="evenodd" d="M7.5 6a4.5 4.5 0 119 0 4.5 4.5 0 01-9 0zM3.751 20.105a8.25 8.25 0 0116.498 0 .75.75 0 01-.437.695A18.683 18.683 0 0112 22.5c-2.786 0-5.433-.608-7.812-1.7a.75.75 0 01-.437-.695z" clipRule="evenodd" />
    <path d="M12.75 12.75a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm-5.25-1.5a.75.75 0 100-1.5.75.75 0 000 1.5zm9 1.5a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm-3-4.5a.75.75 0 100-1.5.75.75 0 000 1.5zm-3.75 7.5a.75.75 0 100-1.5.75.75 0 000 1.5zM15 15.75a.75.75 0 11-1.5 0 .75.75 0 011.5 0z" />
    <path fillRule="evenodd" d="M5.166 2.452A10.456 10.456 0 001.5 10.5c0 4.512 2.868 8.354 6.966 9.803A10.456 10.456 0 0012.75 21a10.456 10.456 0 004.284-9.247 10.456 10.456 0 00-4.284-9.247A10.456 10.456 0 008.466 2.452zM12.75 3a8.956 8.956 0 00-8.25 8.25c0 3.788 2.378 7.04 5.759 8.283A.75.75 0 0010.5 19.5v-2.25a.75.75 0 00-.75-.75h-1.5a.75.75 0 000 1.5h.75v1.408a8.956 8.956 0 001.5-.208 8.956 8.956 0 001.5.208v-1.408h.75a.75.75 0 000-1.5h-1.5a.75.75 0 00-.75.75V19.5a.75.75 0 00.241-.467A8.956 8.956 0 0017.25 11.25a8.956 8.956 0 00-8.25-8.25H12.75z" clipRule="evenodd" />
  </svg>
); 

export const IconMemory: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" {...props}>
    <path d="M16.5 7.5h-9v9h9v-9z" />
    <path fillRule="evenodd" d="M3 5.25A2.25 2.25 0 015.25 3h13.5A2.25 2.25 0 0121 5.25v13.5A2.25 2.25 0 0118.75 21H5.25A2.25 2.25 0 013 18.75V5.25zM4.5 18.75a.75.75 0 00.75.75h13.5a.75.75 0 00.75-.75V5.25a.75.75 0 00-.75-.75H5.25a.75.75 0 00-.75.75v13.5zM8.25 4.5V3M15.75 4.5V3M4.5 8.25H3M4.5 15.75H3M8.25 21v-1.5M15.75 21v-1.5M19.5 8.25H21M19.5 15.75H21" clipRule="evenodd" />
  </svg>
);

export const IconActivity: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 12h16.5m-16.5 3.75h16.5M3.75 19.5h16.5M5.625 4.5h12.75a1.875 1.875 0 010 3.75H5.625a1.875 1.875 0 010-3.75z" />
  </svg>
);

export const IconInfo: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" />
  </svg>
);

export const IconChevronDown: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
  </svg>
);
export const IconChevronUp: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 15.75l7.5-7.5 7.5 7.5" />
  </svg>
);
export const IconGpu: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" {...props}>
    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v2h-2zm0 4h2v6h-2z"/>
    <rect x="7" y="7" width="2" height="2" />
    <rect x="7" y="11" width="2" height="2" />
    <rect x="7" y="15" width="2" height="2" />
    <rect x="15" y="7" width="2" height="2" />
    <rect x="15" y="11" width="2" height="2" />
    <rect x="15" y="15" width="2" height="2" />
  </svg>
);
export const IconLink: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M13.19 8.688a4.5 4.5 0 011.242 7.244l-4.5 4.5a4.5 4.5 0 01-6.364-6.364l1.757-1.757m13.35-.622l1.757-1.757a4.5 4.5 0 00-6.364-6.364l-4.5 4.5a4.5 4.5 0 001.242 7.244" />
    </svg>
);

// --- Tooltip ---
interface TooltipProps { text: string; children: React.ReactNode; position?: 'top' | 'bottom' | 'left' | 'right'; }
export const Tooltip: React.FC<TooltipProps> = ({ text, children, position = 'top' }) => {
  const [visible, setVisible] = useState(false);
  const positionClasses = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
  };
  return (
    <div className="relative inline-block" onMouseEnter={() => setVisible(true)} onMouseLeave={() => setVisible(false)}>
      {children}
      {visible && (
        <div className={`absolute z-10 px-3 py-2 text-sm font-medium text-white bg-gray-700 rounded-lg shadow-sm whitespace-nowrap ${positionClasses[position]}`}>
          {text}
        </div>
      )}
    </div>
  );
};

// --- Helper: ProgressBar ---
interface ProgressBarProps { value: number; maxValue: number; colorClass?: string; heightClass?: string; }
const ProgressBar: React.FC<ProgressBarProps> = ({ value, maxValue, colorClass = 'bg-sky-500', heightClass = 'h-2.5' }) => {
  const percentage = maxValue > 0 ? (value / maxValue) * 100 : 0;
  return (
    <div className={`w-full bg-gray-700 rounded-full ${heightClass} overflow-hidden`}>
      <div className={`${colorClass} ${heightClass} rounded-full`} style={{ width: `${percentage}%` }}></div>
    </div>
  );
};

// --- Helper: StatDisplay ---
interface StatDisplayProps { 
  label: string; 
  value: string | number; 
  unit?: string; 
  icon?: React.ReactElement<React.SVGProps<SVGSVGElement>>; // Type refined for icon prop
  className?: string;
}
export const StatDisplay: React.FC<StatDisplayProps> = ({ label, value, unit, icon, className }) => (
  <div className={`flex flex-col ${className}`}>
    <div className="text-xs text-gray-400 flex items-center">
      {/* icon prop is optional, React.isValidElement is a good check before cloning */}
      {icon && React.isValidElement(icon) && React.cloneElement(icon, { className: 'w-3 h-3 mr-1'})}
      {label}
    </div>
    <div className="text-lg font-semibold text-sky-300">
      {value}
      {unit && <span className="text-xs text-gray-400 ml-1">{unit}</span>}
    </div>
  </div>
);

// --- MemoryUsageDisplay ---
interface MemoryUsageDisplayProps {
  used: number;
  total: number;
  label: string;
}
export const MemoryUsageDisplay: React.FC<MemoryUsageDisplayProps> = ({ used, total, label }) => {
  const percentage = total > 0 ? ((used / total) * 100).toFixed(1) : 0;
  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="bg-gray-800 p-4 rounded-lg shadow-md">
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-md font-semibold text-sky-400">{label}</h3>
        <span className="text-sm text-gray-300">{percentage}% Used</span>
      </div>
      <ProgressBar value={used} maxValue={total} colorClass="bg-teal-500" />
      <div className="mt-2 text-xs text-gray-400 text-right">
        {formatBytes(used)} / {formatBytes(total)}
      </div>
    </div>
  );
};

// --- SmCard ---
interface SmCardProps { sm: StreamingMultiprocessorState; }
export const SmCard: React.FC<SmCardProps> = ({ sm }) => {
  const statusColors: Record<StreamingMultiprocessorState['status'], string> = {
    running: 'bg-green-500',
    idle: 'bg-gray-500',
    waiting: 'bg-yellow-500',
    error: 'bg-red-500',
  };
  const loadPercentage = sm.load_percentage || 0;

  return (
    <div className="bg-gray-800 p-4 rounded-lg shadow-lg hover:shadow-sky-500/30 transition-shadow duration-300">
      <div className="flex justify-between items-center mb-3">
        <h4 className="text-lg font-bold text-sky-400 flex items-center">
          <IconChip className="w-5 h-5 mr-2 text-sky-500" /> SM {sm.id}
        </h4>
        <Tooltip text={`Status: ${sm.status.toUpperCase()}`}>
          <span className={`px-2 py-0.5 text-xs font-semibold rounded-full text-white ${statusColors[sm.status]}`}>
            {sm.status.toUpperCase()}
          </span>
        </Tooltip>
      </div>
      
      <div className="mb-3">
        <div className="text-xs text-gray-400 mb-1">Load</div>
        <ProgressBar value={loadPercentage} maxValue={100} colorClass={loadPercentage > 75 ? 'bg-red-500' : loadPercentage > 50 ? 'bg-yellow-500' : 'bg-green-500'} heightClass="h-2" />
        <div className="text-right text-xs mt-1 text-gray-300">{loadPercentage}%</div>
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
        <StatDisplay label="Active Blocks" value={sm.blocks_active} />
        <StatDisplay label="Pending Blocks" value={sm.blocks_pending} />
        <StatDisplay label="Warps Executed" value={sm.warps_executed.toLocaleString()} />
        <StatDisplay label="Warp Divergences" value={sm.warp_divergences} />
        <StatDisplay label="Bank Conflicts" value={sm.bank_conflicts} />
        <StatDisplay label="Non-Coalesced" value={sm.non_coalesced_accesses} />
        {sm.shared_mem_total_kb > 0 && (
          <StatDisplay label="Shared Mem" value={`${sm.shared_mem_usage_kb}/${sm.shared_mem_total_kb}`} unit="KB"/>
        )}
        {sm.active_block_idx && <StatDisplay label="Active Block" value={sm.active_block_idx} className="col-span-2"/>}
      </div>
    </div>
  );
};

// --- GpuOverviewCard --- (Used in GpuClusterView)
interface GpuOverviewCardProps { summary: GpuSummary; onSelectGpu: (id: string) => void; isSelected: boolean; }
export const GpuOverviewCard: React.FC<GpuOverviewCardProps> = ({ summary, onSelectGpu, isSelected }) => {
  const statusColor = summary.status === 'online' ? 'border-green-500' : summary.status === 'error' ? 'border-red-500' : 'border-gray-600';
  const bgColor = isSelected ? 'bg-sky-700' : 'bg-gray-800 hover:bg-gray-700';
  
  return (
    <button 
      onClick={() => onSelectGpu(summary.id)}
      className={`p-4 rounded-lg shadow-md transition-all duration-200 w-full text-left border-2 ${bgColor} ${isSelected ? 'border-sky-400' : statusColor}`}
    >
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-xl font-semibold text-sky-300 flex items-center">
          <IconGpu className="w-6 h-6 mr-2"/> {summary.name} ({summary.id})
        </h3>
        <span className={`px-2 py-1 text-xs font-bold rounded-full ${summary.status === 'online' ? 'bg-green-600 text-white' : 'bg-red-600 text-white'}`}>
          {summary.status.toUpperCase()}
        </span>
      </div>
      <div className="text-sm text-gray-300 mb-1">Load: {summary.overallLoad}%</div>
      <ProgressBar value={summary.overallLoad} maxValue={100} colorClass={summary.overallLoad > 75 ? 'bg-red-400' : summary.overallLoad > 50 ? 'bg-yellow-400' : 'bg-green-400'} heightClass="h-2"/>
      <div className="mt-2 text-xs text-gray-400">
        Global Mem: { (summary.globalMemoryUsed / (1024*1024)).toFixed(1) }MB / { (summary.globalMemoryTotal / (1024*1024)).toFixed(1) }MB
      </div>
      <div className="text-xs text-gray-400">
        SMs: {summary.activeSMs} / {summary.totalSMs} active
      </div>
    </button>
  );
};


// --- TransfersDisplay ---
interface TransfersDisplayProps { transfers: TransfersState; }
export const TransfersDisplay: React.FC<TransfersDisplayProps> = ({ transfers }) => (
  <div className="bg-gray-800 p-4 rounded-lg shadow-md">
    <h3 className="text-md font-semibold text-sky-400 mb-3 flex items-center">
      <IconLink className="w-5 h-5 mr-2 text-sky-500" /> Data Transfers
    </h3>
    <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
      <StatDisplay label="Host to Device" value={transfers.H2D} unit="ops" />
      <StatDisplay label="Device to Host" value={transfers.D2H} unit="ops" />
      {transfers.InterGPU !== undefined && <StatDisplay label="Inter-GPU" value={transfers.InterGPU} unit="ops" />}
      <StatDisplay label="Bytes Transferred" value={(transfers.bytes_transferred / (1024*1024)).toFixed(2)} unit="MB" className="col-span-2 md:col-span-1"/>
       {transfers.cycles_transferred !== undefined && <StatDisplay label="Transfer Cycles" value={transfers.cycles_transferred.toLocaleString()} />}
    </div>
  </div>
);

// --- EventLog ---
interface EventLogProps { events: SimulatorEvent[]; }
export const EventLog: React.FC<EventLogProps> = ({ events }) => {
  const getEventTypeColor = (type: SimulatorEvent['type']) => {
    switch (type) {
      case 'KERNEL_LAUNCH': return 'text-purple-400';
      case 'BLOCK_START': case 'BLOCK_END': return 'text-blue-400';
      case 'WARP_DIVERGENCE': return 'text-yellow-400';
      case 'MEM_COPY_H2D': case 'MEM_COPY_D2H': case 'MEM_COPY_INTERGPU': return 'text-teal-400';
      case 'SYNC_EVENT': return 'text-pink-400';
      case 'ERROR_EVENT': return 'text-red-400 font-semibold';
      case 'INFO_EVENT':
      default: return 'text-gray-300';
    }
  };

  return (
    <div className="bg-gray-800 p-4 rounded-lg shadow-lg h-96 flex flex-col">
      <h3 className="text-xl font-semibold text-sky-400 mb-3 flex items-center">
        <IconActivity className="w-6 h-6 mr-2 text-sky-500" /> Event Log
      </h3>
      <div className="overflow-y-auto flex-grow pr-2">
        {events.length === 0 && <p className="text-gray-500">No events yet.</p>}
        {events.map(event => (
          <div key={event.id} className="mb-2.5 pb-2.5 border-b border-gray-700 last:border-b-0 last:pb-0 last:mb-0">
            <div className="flex justify-between text-xs text-gray-500 mb-0.5">
              <span>{new Date(event.timestamp).toLocaleTimeString()}</span>
              <span className={getEventTypeColor(event.type)}>{event.type}</span>
            </div>
            <p className="text-sm text-gray-200">{event.message}</p>
            {event.gpuId && <span className="text-xs text-sky-500 mr-2">GPU: {event.gpuId}</span>}
            {event.smId !== undefined && <span className="text-xs text-teal-500 mr-2">SM: {event.smId}</span>}
            {event.blockIdx && <span className="text-xs text-indigo-500">Block: {event.blockIdx}</span>}
            {/* TODO: Display event.details if needed */}
          </div>
        ))}
      </div>
    </div>
  );
};