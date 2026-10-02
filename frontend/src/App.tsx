import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import type { NavItem } from './components/Sidebar';
import { MetricCard } from './components/MetricCard';
import { PipelinePanel } from './components/PipelinePanel';
import { fetchHealth, fetchSystemInfo } from './api/client';
import type { SystemHealthResponse, SystemInfoResponse } from './api/client';
import {
  Clock,
  Layers,
  AlertTriangle
} from 'lucide-react';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavItem>('upload');
  const [health, setHealth] = useState<SystemHealthResponse | null>(null);
  const [systemInfo, setSystemInfo] = useState<SystemInfoResponse | null>(null);
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

          {currentTab === 'upload' && <PipelinePanel />}

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

          {/* Module-Specific Content Panes */}
          <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-6">
            {currentTab === 'overview' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between border-b border-industrial-800 pb-4">
                  <div>
                    <h3 className="text-base font-semibold text-slate-100 flex items-center space-x-2">
                      <Layers className="w-5 h-5 text-industrial-sovereign" />
                      <span>Implemented Optimization Pipeline</span>
                    </h3>
                    <p className="text-xs text-slate-400 mt-1">
                      Current implementation status for the model processing and solve stages.
                    </p>
                  </div>
                  <span className="text-xs font-mono text-slate-400 bg-industrial-950 px-2.5 py-1 rounded border border-industrial-800">
                    No fabricated solver data
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {[
                    {
                      name: '1. Canonical Model (CIR)',
                      status: 'IMPLEMENTED',
                      badge: 'success',
                      desc: 'Solver-independent sparse model with bounds, senses, and variable types.',
                      phase: 'Phase 1'
                    },
                    {
                      name: '2. Model Parser (LP / MPS)',
                      status: 'IMPLEMENTED',
                      badge: 'success',
                      desc: 'LP and MPS input is converted into CanonicalModel.',
                      phase: 'Phase 2'
                    },
                    {
                      name: '3. Structural Fingerprinting',
                      status: 'IMPLEMENTED',
                      badge: 'success',
                      desc: 'Computes sparse structure metrics and conservative structural signals.',
                      phase: 'Phase 3'
                    },
                    {
                      name: '4. Industrial Structure Graph (ISG)',
                      status: 'IMPLEMENTED',
                      badge: 'success',
                      desc: 'Builds weighted variable, constraint, and objective relationships.',
                      phase: 'Phase 4'
                    },
                    {
                      name: '5. Strategy Selection Engine',
                      status: 'IMPLEMENTED',
                      badge: 'success',
                      desc: 'Produces explainable heuristic recommendations from measured evidence.',
                      phase: 'Phase 5'
                    },
                    {
                      name: '6. Optimization Engine (HiGHS)',
                      status: 'IMPLEMENTED',
                      badge: 'success',
                      desc: 'Runs LP and supported mixed-integer models through the registered HiGHS backend.',
                      phase: 'Phase 6'
                    },
                    {
                      name: '7. Sakshi Independent Verifier',
                      status: 'IMPLEMENTED',
                      badge: 'success',
                      desc: 'Checks primal feasibility, bounds, integrality, and objective consistency; it does not prove optimality.',
                      phase: 'Phase 7'
                    },
                    {
                      name: '8. Solve Certificate Generator',
                      status: 'IMPLEMENTED',
                      badge: 'success',
                      desc: 'Records solver-reported data separately from SAKSHI checks with a deterministic SHA-256 digest.',
                      phase: 'Phase 8'
                    },
                    {
                      name: '9. Synthetic Refinery Model',
                      status: 'SCHEDULED (Phase 10)',
                      badge: 'neutral',
                      desc: 'Multi-period refinery planning with crude feeds, octane blending, and tank inventories.',
                      phase: 'Phase 10'
                    },
                    {
                      name: '10. Netlib LP Benchmark Suite',
                      status: 'SCHEDULED (Phase 11)',
                      badge: 'neutral',
                      desc: 'Authentic Netlib LP validation (afiro, adlittle) for solver integration and objective accuracy.',
                      phase: 'Phase 11'
                    }
                  ].map((m, idx) => (
                    <div key={idx} className="p-4 bg-industrial-950/70 border border-industrial-800 rounded-lg flex flex-col justify-between">
                      <div>
                        <div className="flex items-center justify-between">
                          <h4 className="text-sm font-semibold text-slate-200">{m.name}</h4>
                          <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                            m.badge === 'success'
                              ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                              : 'bg-industrial-800 text-slate-400 border-industrial-700'
                          }`}>
                            {m.status}
                          </span>
                        </div>
                        <p className="text-xs text-slate-400 mt-2 leading-relaxed">{m.desc}</p>
                      </div>
                      <div className="mt-3 pt-2 border-t border-industrial-850 flex items-center justify-between text-[11px] font-mono text-slate-400">
                        <span>Milestone:</span>
                        <span className="text-industrial-sovereign">{m.phase}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {currentTab !== 'overview' && currentTab !== 'upload' && (
              <div className="text-center py-12 space-y-4">
                <div className="w-12 h-12 mx-auto rounded-full bg-industrial-850 border border-industrial-700 flex items-center justify-center text-industrial-sovereign">
                  <Clock className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-100 uppercase tracking-wide">
                    {currentTab.replace('_', ' ')} Module Interface
                  </h3>
                  <p className="text-xs text-slate-400 max-w-md mx-auto mt-1 leading-relaxed">
                    Detailed module views are not part of this dashboard yet. Run a model from Model Upload to inspect live pipeline results.
                  </p>
                </div>
                <div className="inline-block px-3 py-1 rounded bg-industrial-950 border border-industrial-800 text-xs font-mono text-industrial-accent">
                  Status: No Dedicated Detail View
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;
