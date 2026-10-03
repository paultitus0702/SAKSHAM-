import React from 'react';
import {
  UploadCloud,
  FileText,
  Fingerprint,
  Network,
  Layers,
  Compass,
  PlayCircle,
  BarChart3,
  HeartPulse,
  Award,
  ShieldCheck,
  FlaskConical
} from 'lucide-react';

export type NavItem =
  | 'upload'
  | 'overview'
  | 'fingerprint'
  | 'isg'
  | 'structures'
  | 'strategy'
  | 'progress'
  | 'results'
  | 'health'
  | 'certificate'
  | 'sakshi'
  | 'benchmarks';

interface SidebarProps {
  currentTab: NavItem;
  onSelectTab: (tab: NavItem) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ currentTab, onSelectTab }) => {
  const navItems = [
    { id: 'upload', label: 'Model Upload', icon: UploadCloud, phase: 'Phase 2' },
    { id: 'overview', label: 'Model Overview', icon: FileText, phase: 'Phase 2' },
    { id: 'fingerprint', label: 'Structural Fingerprint', icon: Fingerprint, phase: 'Phase 3' },
    { id: 'isg', label: 'Structure Graph (ISG)', icon: Network, phase: 'Phase 4' },
    { id: 'structures', label: 'Detected Structures', icon: Layers, phase: 'Phase 3' },
    { id: 'strategy', label: 'Recommended Strategy', icon: Compass, phase: 'Phase 5' },
    { id: 'progress', label: 'Solve Progress', icon: PlayCircle, phase: 'Phase 6' },
    { id: 'results', label: 'Optimization Results', icon: BarChart3, phase: 'Phase 6' },
    { id: 'health', label: 'Numerical Health', icon: HeartPulse, phase: 'Phase 3/6' },
    { id: 'certificate', label: 'Solve Certificate', icon: Award, phase: 'Phase 8' },
    { id: 'sakshi', label: 'Sakshi Verification', icon: ShieldCheck, phase: 'Phase 7' },
    { id: 'benchmarks', label: 'Benchmark Suite', icon: FlaskConical, phase: 'Phase 11' },
  ] as const;

  return (
    <aside className="w-64 border-r border-industrial-800 bg-industrial-900/60 p-4 flex flex-col justify-between">
      <div className="space-y-1">
        <div className="px-3 py-2 text-xs font-mono font-semibold uppercase tracking-wider text-slate-400">
          Cockpit Modules
        </div>
        <nav className="space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectTab(item.id as NavItem)}
                className={`w-full flex items-center justify-between px-3 py-2 text-xs rounded-md transition font-medium ${
                  isActive
                    ? 'bg-industrial-800 text-industrial-sovereign border border-industrial-700 shadow-sm'
                    : 'text-slate-300 hover:bg-industrial-850 hover:text-slate-100'
                }`}
              >
                <div className="flex items-center space-x-2.5">
                  <Icon className={`w-4 h-4 ${isActive ? 'text-industrial-sovereign' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
                </div>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-industrial-950 text-slate-400 border border-industrial-800">
                  {item.phase}
                </span>
              </button>
            );
          })}
        </nav>
      </div>

      <div className="p-3 bg-industrial-950 border border-industrial-800 rounded-lg text-xs space-y-1.5">
        <div className="flex items-center justify-between">
          <span className="text-slate-400 font-mono text-[11px]">ACTIVE PHASE</span>
          <span className="text-industrial-accent font-mono text-[11px] font-bold">PHASE 10</span>
        </div>
        <p className="text-[11px] text-slate-400 leading-relaxed">
          Synthetic industrial benchmark generation and verified optimization workflow available.
        </p>
      </div>
    </aside>
  );
};
