import React from 'react';

interface MetricCardProps {
  label: string;
  value: string | number;
  subtext?: string;
  statusBadge?: string;
  badgeType?: 'success' | 'warning' | 'info' | 'neutral';
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subtext,
  statusBadge,
  badgeType = 'neutral',
}) => {
  const badgeColors = {
    success: 'bg-emerald-950 text-emerald-400 border-emerald-800',
    warning: 'bg-amber-950 text-amber-400 border-amber-800',
    info: 'bg-cyan-950 text-cyan-400 border-cyan-800',
    neutral: 'bg-industrial-800 text-slate-300 border-industrial-700',
  };

  return (
    <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-4 shadow-sm hover:border-industrial-700 transition">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-mono uppercase tracking-wider text-slate-400">{label}</span>
        {statusBadge && (
          <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${badgeColors[badgeType]}`}>
            {statusBadge}
          </span>
        )}
      </div>
      <div className="text-xl font-bold font-mono text-slate-100">{value}</div>
      {subtext && <div className="text-xs text-slate-400 mt-1">{subtext}</div>}
    </div>
  );
};
