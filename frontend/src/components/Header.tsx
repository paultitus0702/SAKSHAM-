import React from 'react';
import { Shield, Cpu, Activity, CheckCircle2, AlertCircle } from 'lucide-react';
import type { SystemHealthResponse } from '../api/client';

interface HeaderProps {
  health: SystemHealthResponse | null;
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
}

export const Header: React.FC<HeaderProps> = ({ health, loading, error, onRefresh }) => {
  return (
    <header className="border-b border-industrial-800 bg-industrial-900 px-6 py-4 flex items-center justify-between shadow-md">
      <div className="flex items-center space-x-3">
        <div className="p-2 bg-industrial-850 rounded-lg border border-industrial-700 flex items-center justify-center">
          <Shield className="w-6 h-6 text-industrial-sovereign" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-xl font-bold tracking-wider text-slate-100 uppercase">
              SAKSHAM <span className="text-xs px-2 py-0.5 rounded font-mono bg-industrial-800 text-industrial-sovereign border border-industrial-700">सक्षम</span>
            </h1>
            <span className="text-xs font-mono text-slate-400 border-l border-industrial-700 pl-2">
              v0.1.0-alpha
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Sovereign Structure-Aware Sparse Optimization Engine for India's Industrial Systems
          </p>
        </div>
      </div>

      <div className="flex items-center space-x-4">
        {/* Backend Connectivity Status */}
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-md bg-industrial-850 border border-industrial-800 text-xs">
          <Activity className="w-3.5 h-3.5 text-slate-400" />
          <span className="text-slate-400 font-mono">BACKEND:</span>
          {loading ? (
            <span className="text-industrial-accent animate-pulse font-mono">CONNECTING...</span>
          ) : error ? (
            <span className="flex items-center text-industrial-danger font-mono font-medium">
              <AlertCircle className="w-3.5 h-3.5 mr-1" /> OFFLINE
            </span>
          ) : (
            <span className="flex items-center text-industrial-success font-mono font-medium">
              <CheckCircle2 className="w-3.5 h-3.5 mr-1" /> OPERATIONAL
            </span>
          )}
        </div>

        {/* Solver Backend Status */}
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-md bg-industrial-850 border border-industrial-800 text-xs">
          <Cpu className="w-3.5 h-3.5 text-slate-400" />
          <span className="text-slate-400 font-mono">SOLVER:</span>
          <span className="text-industrial-sovereign font-mono font-semibold">
            HiGHS {health?.solvers?.highs?.version || 'Ready'}
          </span>
        </div>

        <button
          onClick={onRefresh}
          className="text-xs font-mono px-3 py-1.5 rounded bg-industrial-800 hover:bg-industrial-700 text-slate-200 border border-industrial-700 transition"
        >
          Check Link
        </button>
      </div>
    </header>
  );
};
