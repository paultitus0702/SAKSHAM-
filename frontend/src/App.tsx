import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import type { NavItem } from './components/Sidebar';
import { MetricCard } from './components/MetricCard';
import { PipelinePanel } from './components/PipelinePanel';
import { PipelineModuleView } from './components/PipelineModuleView';
import { fetchHealth, fetchSystemInfo } from './api/client';
import type {
  PipelineResultResponse,
  SystemHealthResponse,
  SystemInfoResponse,
} from './api/client';
import { BenchmarkSuite } from './components/BenchmarkSuite';
import {
  AlertTriangle
} from 'lucide-react';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavItem>('upload');
  const [health, setHealth] = useState<SystemHealthResponse | null>(null);
  const [systemInfo, setSystemInfo] = useState<SystemInfoResponse | null>(null);
  const [pipelineResult, setPipelineResult] = useState<PipelineResultResponse | null>(null);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [hData, infoData] = await Promise.all([fetchHealth(), fetchSystemInfo()]);
      setHealth(hData);
      setSystemInfo(infoData);
    } catch (err: any) {
      setError(err?.message || 'Failed to connect to SAKSHAM backend.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div className="min-h-screen bg-industrial-950 text-slate-100 flex flex-col font-sans">
      <Header health={health} loading={loading} error={error} onRefresh={loadData} />

      <div className="flex-1 flex overflow-hidden">
        <Sidebar currentTab={currentTab} onSelectTab={setCurrentTab} />

        <main className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Top Banner: Verification & System Status */}
          <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-5 shadow-sm">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 font-semibold">
                    PHASE 8 ACTIVE
                  </span>
                  <span className="text-xs font-mono text-slate-400">
                    END-TO-END SOLVE PIPELINE AVAILABLE
                  </span>
                </div>
                <h2 className="text-xl font-bold text-slate-100 mt-2">
                  Structure-Aware Optimization Cockpit
                </h2>
                <p className="text-sm text-slate-400 mt-1 max-w-3xl leading-relaxed">
                  SAKSHAM parses LP/MPS models, analyzes sparse structure and numerical health, recommends a solver approach, then checks the returned primal solution and records a certificate.
                </p>
              </div>

              <div className="flex items-center space-x-3">
                <div className="px-4 py-2 bg-industrial-850 border border-industrial-800 rounded-lg text-right">
                  <div className="text-[10px] font-mono text-slate-400 uppercase">Live Backend Status</div>
                  <div className={`text-sm font-mono font-bold ${error ? 'text-industrial-danger' : 'text-industrial-success'}`}>
                    {error ? 'OFFLINE' : 'ONLINE (200 OK)'}
                  </div>
                </div>
              </div>
            </div>

            {/* Error Banner if Backend is Unreachable */}
            {error && (
              <div className="mt-4 p-3 bg-red-950/50 border border-red-800/80 rounded-lg text-xs text-red-300 flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0" />
                <span>
                  Backend connection error: <strong>{error}</strong>. Ensure FastAPI server is running on port 8000.
                </span>
              </div>
            )}
          </div>

          {/* Core Metric Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <MetricCard
              label="Engine Status"
              value={health?.engine?.status || 'OPERATIONAL'}
              subtext={`saksham-engine v${health?.engine?.version || '0.1.0'}`}
              statusBadge="IMPLEMENTED"
              badgeType="success"
            />
            <MetricCard
              label="HiGHS Solver"
              value={`v${health?.solvers?.highs?.version || '1.15.1'}`}
              subtext="C++ highspy integration verified"
              statusBadge="IMPLEMENTED"
              badgeType="success"
            />
            <MetricCard
              label="Canonical Representation"
              value="CSR / CSC"
              subtext="SciPy zero-copy matrix layout"
              statusBadge="IMPLEMENTED"
              badgeType="success"
            />
            <MetricCard
              label="Active Pipeline"
              value="Phase 8: End-to-End"
              subtext={systemInfo?.app_name ? `${systemInfo.app_name} LP/MPS workflow` : 'Analysis through verification'}
              statusBadge="AVAILABLE"
              badgeType="success"
            />
          </div>

          {currentTab === 'upload' && (
            <PipelinePanel
              result={pipelineResult}
              onResultChange={setPipelineResult}
              onRunningChange={setPipelineRunning}
            />
          )}
          {currentTab === 'benchmarks' && <BenchmarkSuite />}
          {currentTab !== 'upload' && currentTab !== 'benchmarks' && (
            <PipelineModuleView
              tab={currentTab}
              result={pipelineResult}
              running={pipelineRunning}
              onNavigate={setCurrentTab}
            />
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
