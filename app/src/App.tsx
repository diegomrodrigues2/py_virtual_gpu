
import React, { useState, useEffect, useCallback } from 'react';
import { GPUState, SimulatorEvent, GpuSummary, BackendData, MemorySlice } from './types/types';
import { fetchBackendData, fetchGpuState, fetchGlobalMemorySlice, fetchConstantMemorySlice } from './services/gpuSimulatorService';
import { IconChip, IconMemory, IconActivity, IconInfo, IconChevronDown, IconChevronUp, Tooltip, MemoryUsageDisplay, SmCard, GpuOverviewCard, TransfersDisplay, EventLog, IconGpu, IconLink, StatDisplay } from './components/components';
import { MemoryViewer } from './components/MemoryViewer';
import { KernelLogView } from './components/KernelLogView';


interface DashboardLayoutProps {
  gpuSummaries: GpuSummary[];
  selectedGpu: GPUState | undefined;
  allGpuStates: GPUState[];
  events: SimulatorEvent[];
  onSelectGpu: (id: string) => void;
  isLoading: boolean;
  currentView: 'cluster' | 'detail';
  onSetView: (view: 'cluster' | 'detail') => void; // Changed from onToggleView
}

export const DashboardLayout: React.FC<DashboardLayoutProps> = ({
  gpuSummaries,
  selectedGpu,
  events,
  onSelectGpu,
  isLoading,
  currentView,
  onSetView,
  allGpuStates,
}) => {
  const selectedGpuId = selectedGpu?.id;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap justify-between items-center gap-4 mb-6"> {/* Added gap and mb for better spacing */}
        {/* Always show view toggle when GPUs are available */}
        {allGpuStates.length > 0 && (
             <div className="flex items-center space-x-2 bg-gray-800 p-2 rounded-lg">
                <span className="text-sm font-medium text-gray-300">View:</span>
                <button
                    onClick={() => onSetView('cluster')} // Changed from onToggleView
                    disabled={currentView === 'cluster'} 
                    className={`px-4 py-2 rounded-md text-sm font-semibold transition-colors ${currentView === 'cluster' ? 'bg-sky-600 text-white cursor-default' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
                >
                    Cluster Overview
                </button>
                 <button
                    onClick={() => onSetView('detail')} // Changed from onToggleView
                    disabled={currentView === 'detail'} 
                    className={`px-4 py-2 rounded-md text-sm font-semibold transition-colors ${currentView === 'detail' ? 'bg-sky-600 text-white cursor-default' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
                >
                    GPU Detail
                </button>
            </div>
        )}
        {currentView === 'detail' && allGpuStates.length > 1 && (
          <div className="flex items-center space-x-2 bg-gray-800 p-2 rounded-lg">
            <label htmlFor="gpu-select" className="text-sm font-medium text-gray-300">Select GPU:</label>
            <select
              id="gpu-select"
              value={selectedGpuId || ''}
              onChange={(e) => onSelectGpu(e.target.value)}
              className="bg-gray-700 border border-gray-600 text-white text-sm rounded-lg focus:ring-sky-500 focus:border-sky-500 p-2.5"
              disabled={isLoading}
            >
              {allGpuStates.map(gpu => (
                <option key={gpu.id} value={gpu.id}>{gpu.name} ({gpu.id})</option>
              ))}
            </select>
          </div>
        )}
      </div>

      {currentView === 'cluster' && (
        <GpuClusterView
          summaries={gpuSummaries}
          onSelectGpu={onSelectGpu}
          selectedGpuId={selectedGpuId}
        />
      )}
      
      {currentView === 'detail' && (
        selectedGpu ? (
          <GpuDetailView gpu={selectedGpu} />
        ) : (
          !isLoading && (
            <div className="text-center py-10 bg-gray-800 rounded-lg">
              <p className="text-xl text-gray-400">Select a GPU to see details or switch to Cluster View.</p>
            </div>
          )
        )
      )}

      {/* Render EventLog if there's data, or selectedGpu for detail view context */}
      {(events.length > 0 || selectedGpu) && <EventLog events={events} /> }
      
    </div>
  );
};


const GpuClusterView: React.FC<{summaries: GpuSummary[], onSelectGpu: (id: string) => void, selectedGpuId: string | null | undefined}> = ({ summaries, onSelectGpu, selectedGpuId }) => (
  <div className="bg-gray-800 p-6 rounded-lg shadow-xl">
    <h2 className="text-2xl font-semibold text-sky-400 mb-6 flex items-center">
        <IconGpu className="w-7 h-7 mr-3 text-sky-500" /> GPU Cluster Overview
    </h2>
    {summaries.length === 0 && <p className="text-gray-400">No GPU summaries available. GPUs might be offline or initializing.</p>}
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {summaries.map(summary => (
        <GpuOverviewCard key={summary.id} summary={summary} onSelectGpu={onSelectGpu} isSelected={summary.id === selectedGpuId} />
      ))}
    </div>
    {summaries.length > 1 && (
        <div className="mt-8 pt-6 border-t border-gray-700">
            <h3 className="text-xl font-medium text-sky-300 mb-4 flex items-center">
                <IconLink className="w-5 h-5 mr-2 text-sky-400" /> Inter-GPU Communication (Conceptual)
            </h3>
            <div className="flex justify-around items-center h-20 bg-gray-700/50 rounded-lg p-4">
                {summaries.map((gpu, index) => (
                    <React.Fragment key={gpu.id}>
                        <div className="flex flex-col items-center text-center w-20"> {/* Added text-center and w-20 */}
                            <IconGpu className="w-8 h-8 text-sky-400"/>
                            <span className="text-xs mt-1 truncate">{gpu.name}</span> {/* Added truncate */}
                        </div>
                        {index < summaries.length - 1 && (
                             <div className="flex-grow h-0.5 bg-gray-600 relative mx-2"> {/* Added mx-2 */}
                                <IconLink className="w-4 h-4 absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-gray-500" />
                             </div>
                        )}
                    </React.Fragment>
                ))}
            </div>
            <p className="text-xs text-gray-500 mt-2 text-center">Visual representation of potential communication links.</p>
        </div>
    )}
  </div>
);


export const GpuDetailView: React.FC<{ gpu: GPUState }> = ({ gpu }) => {
  const [offset, setOffset] = useState(0);
  const [size, setSize] = useState(64);
  const [memType, setMemType] = useState<'global' | 'constant'>('global');
  const [dtype, setDtype] = useState<'' | 'half' | 'float32' | 'float64'>('');
  const [slice, setSlice] = useState<MemorySlice | null>(null);
  const [loading, setLoading] = useState(false);
  const [showKernelLog, setShowKernelLog] = useState(false);

  const fetchSlice = async () => {
    setLoading(true);
    try {
      const s =
        memType === 'global'
          ? await fetchGlobalMemorySlice(gpu.id, offset, size, dtype || undefined)
          : await fetchConstantMemorySlice(gpu.id, offset, size, dtype || undefined);
      setSlice(s);
    } catch (err) {
      console.error('Failed to fetch memory slice', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800 p-6 rounded-lg shadow-xl">
          <div className="flex flex-col md:flex-row justify-between md:items-center mb-6">
              <h2 className="text-3xl font-bold text-sky-400 flex items-center">
                  <IconGpu className="w-8 h-8 mr-3 text-sky-500" /> {gpu.name} ({gpu.id}) - Details
              </h2>
              <div className="flex space-x-4 mt-3 md:mt-0">
                  <StatDisplay label="Overall Load" value={`${gpu.overall_load}%`} />
                  {gpu.temperature !== undefined && <StatDisplay label="Temp" value={`${gpu.temperature}°C`} />}
                  {gpu.power_draw_watts !== undefined && <StatDisplay label="Power" value={`${gpu.power_draw_watts}W`} />}
              </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <MemoryUsageDisplay used={gpu.global_memory.used} total={gpu.global_memory.total} label="Global Memory" />
              <TransfersDisplay transfers={gpu.transfers}/>
          </div>

          <div className="mt-4">
            <h3 className="text-lg font-semibold text-sky-400 mb-2 flex items-center">
              <IconMemory className="w-5 h-5 mr-2" /> Memory Inspector
            </h3>
            <div className="flex items-center space-x-2 mb-2">
              <select
                value={memType}
                onChange={(e) => setMemType(e.target.value as 'global' | 'constant')}
                className="bg-gray-700 p-1 rounded text-sm"
              >
                <option value="global">Global</option>
                <option value="constant">Constant</option>
              </select>
              <input
                type="number"
                value={offset}
                onChange={(e) => setOffset(Number(e.target.value))}
                className="bg-gray-700 p-1 rounded w-24 text-sm"
                placeholder="Offset"
              />
              <input
                type="number"
                value={size}
                onChange={(e) => setSize(Number(e.target.value))}
                className="bg-gray-700 p-1 rounded w-20 text-sm"
                placeholder="Size"
              />
              <select
                value={dtype}
                onChange={(e) => setDtype(e.target.value as '' | 'half' | 'float32' | 'float64')}
                className="bg-gray-700 p-1 rounded text-sm"
              >
                <option value="">raw</option>
                <option value="half">half</option>
                <option value="float32">float32</option>
                <option value="float64">float64</option>
              </select>
              <button onClick={fetchSlice} className="px-2 py-1 bg-sky-600 rounded text-xs mr-2">
                Fetch
              </button>
              {slice && (
                <button
                  onClick={() => setSlice(null)}
                  className="px-2 py-1 bg-gray-700 rounded text-xs"
                >
                  Clear
                </button>
              )}
            </div>
            {loading && <p className="text-xs text-gray-400">Loading...</p>}
            {slice && <MemoryViewer slice={slice} />}
            <button
              onClick={() => setShowKernelLog((v) => !v)}
              className="mt-4 px-2 py-1 bg-gray-700 rounded text-xs"
            >
              {showKernelLog ? 'Hide Kernel Log' : 'Show Kernel Log'}
            </button>
            {showKernelLog && <KernelLogView gpuId={gpu.id} />}
          </div>
      </div>

      <div>
        <h3 className="text-2xl font-semibold text-sky-400 mb-4 ml-1 flex items-center">
          <IconChip className="w-7 h-7 mr-2 text-sky-500" /> Streaming Multiprocessors ({gpu.config.num_sms})
        </h3>
        {gpu.sms.length === 0 && <p className="text-gray-400 bg-gray-800 p-4 rounded-lg">No SMs configured or active for this GPU.</p>}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {gpu.sms.map(sm => (
            <SmCard key={sm.id} sm={sm} gpuId={gpu.id} />
          ))}
        </div>
      </div>
    </div>
  );
};


const App: React.FC = () => {
  const [backendData, setBackendData] = useState<BackendData | null>(null);
  const [selectedGpuId, setSelectedGpuId] = useState<string | null>(null);
  const [selectedGpuState, setSelectedGpuState] = useState<GPUState | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [currentView, setCurrentView] = useState<'cluster' | 'detail'>('cluster');


  const loadData = useCallback(async () => {
    if (!backendData && !isLoading) setIsLoading(true); 
    try {
      const data = await fetchBackendData();
      setBackendData(data);

      if (data.gpuStates.length > 0) {
        if (!selectedGpuId || !data.gpuStates.find(g => g.id === selectedGpuId)) {
          setSelectedGpuId(data.gpuStates[0].id);
        }
        if (data.gpuStates.length === 1 && currentView === 'cluster' && selectedGpuId === null) {
             setCurrentView('detail');
        }
      } else { 
        setSelectedGpuId(null);
        if (currentView === 'detail') {
            setCurrentView('cluster');
        }
      }
      setError(null);
    } catch (err) {
      console.error("Failed to load simulator data:", err);
      setError("Failed to load simulator data. Please check connection or try again.");
    } finally {
      setIsLoading(false);
    }
  }, [backendData, selectedGpuId, currentView, isLoading]);

  useEffect(() => {
    loadData(); 
    const intervalId = setInterval(loadData, 3000); 
    return () => clearInterval(intervalId);
  }, [loadData]);

  const handleSelectGpu = async (id: string) => {
    setSelectedGpuId(id);
    try {
      const state = await fetchGpuState(id);
      setSelectedGpuState(state);
    } catch (err) {
      console.error('Failed to fetch GPU state', err);
    }
    handleSetView('detail');
  };

  const handleSetView = (newView: 'cluster' | 'detail') => {
    if (newView === 'detail') {
        const gpus = backendData?.gpuStates;
        if (gpus && gpus.length > 0) {
            if (gpus.length === 1) {
                setSelectedGpuId(gpus[0].id);
            } else if (!selectedGpuId) { 
                setSelectedGpuId(gpus[0].id);
            }
        }
    }
    setCurrentView(newView);
  };
  
  const selectedGpu = selectedGpuState ?? backendData?.gpuStates.find(gpu => gpu.id === selectedGpuId);

  useEffect(() => {
    if (backendData) {
        if (backendData.gpuStates.length === 1) {
            setCurrentView('detail');
            if(backendData.gpuStates[0] && (!selectedGpuId || selectedGpuId !== backendData.gpuStates[0].id)) {
              setSelectedGpuId(backendData.gpuStates[0].id);
            }
        } else if (backendData.gpuStates.length > 1) {
            if (!selectedGpuId && backendData.gpuStates[0]) {
                 setSelectedGpuId(backendData.gpuStates[0].id);
            }
        } else if (backendData.gpuStates.length === 0) {
             setCurrentView('cluster'); 
             setSelectedGpuId(null);
        }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backendData?.gpuStates.length]);

  if (isLoading && !backendData) {
    return (
      <div className="min-h-screen bg-gray-900 text-gray-100 p-4 flex flex-col justify-center items-center">
        <div className="animate-spin rounded-full h-24 w-24 border-t-4 border-b-4 border-sky-500 mb-6"></div>
        <h1 className="text-3xl font-bold text-sky-400">GPU Simulator Dashboard</h1>
        <p className="text-xl text-gray-300 mt-2">Initializing Simulator Data...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-4 md:p-8 font-sans">
      <header className="mb-8 text-center">
        <h1 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-sky-400 to-teal-400">
          GPU Simulator Dashboard
        </h1>
      </header>

      {error && (
        <div className="bg-red-800 border border-red-700 text-white p-4 rounded-lg shadow-lg text-center mb-6 max-w-2xl mx-auto">
          <p className="font-bold text-lg flex items-center justify-center"><IconInfo className="w-6 h-6 mr-2"/>Error</p>
          <p>{error}</p>
          <button onClick={loadData} className="mt-3 px-4 py-2 bg-red-600 hover:bg-red-500 rounded-md text-sm font-medium">Try Reload</button>
        </div>
      )}

      {!error && !isLoading && backendData && backendData.gpuStates.length === 0 && (
         <div className="text-center py-12 bg-gray-800 rounded-xl shadow-xl max-w-2xl mx-auto">
           <IconChip className="w-24 h-24 text-gray-600 mx-auto mb-6"/>
           <p className="text-3xl font-semibold text-gray-400">No GPU Data Available</p>
           <p className="text-gray-500 mt-2">The simulator might not be running or connected, or no GPUs are currently active.</p>
           <button onClick={loadData} className="mt-6 px-6 py-2.5 bg-sky-600 hover:bg-sky-500 rounded-lg text-sm font-medium transition-colors">
            Attempt Reconnect
           </button>
         </div>
      )}

      {backendData && backendData.gpuStates.length > 0 && (
        <DashboardLayout
          gpuSummaries={backendData.gpuSummaries}
          selectedGpu={selectedGpu}
          allGpuStates={backendData.gpuStates}
          events={backendData.events}
          onSelectGpu={handleSelectGpu}
          isLoading={isLoading && !!backendData} 
          currentView={currentView}
          onSetView={handleSetView} // Changed from onToggleView
        />
      )}
      <footer className="text-center mt-12 py-4 border-t border-gray-700">
        <p className="text-xs text-gray-500">GPU Simulator Dashboard &copy; {new Date().getFullYear()}. For educational purposes.</p>
      </footer>
    </div>
  );
};

export default App;
