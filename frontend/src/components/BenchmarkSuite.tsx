import React, { useEffect, useState } from 'react';
import {
  FlaskConical,
  Play,
  CheckCircle2,
  XCircle,
  ShieldCheck,
  Award,
  Activity,
  Clock,
  Database,
  Loader2,
} from 'lucide-react';

import {
  fetchNetlibBenchmarks,
  runNetlibBenchmark,
} from '../api/client';

import type {
  NetlibBenchmarkMetadata,
  NetlibBenchmarkResult,
} from '../api/client';

export const BenchmarkSuite: React.FC = () => {
  const [benchmarks, setBenchmarks] = useState<NetlibBenchmarkMetadata[]>([]);
  const [selected, setSelected] = useState<NetlibBenchmarkResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadBenchmarks = async () => {
      try {
        setLoading(true);
        setError(null);

        const data = await fetchNetlibBenchmarks();
        setBenchmarks(data.benchmarks);
      } catch (err: any) {
        setError(err?.message || 'Unable to load benchmark suite.');
      } finally {
        setLoading(false);
      }
    };

    loadBenchmarks();
  }, []);

  const executeBenchmark = async (name: string) => {
    try {
      setRunning(name);
      setError(null);

      const result = await runNetlibBenchmark(name);
      setSelected(result);
    } catch (err: any) {
      setError(err?.message || `Failed to run ${name}.`);
    } finally {
      setRunning(null);
    }
  };

  return (
    <div className="space-y-6">

      {/* Header */}
      <div className="bg-industrial-950 border border-industrial-800 rounded-xl p-5">
        <div className="flex items-start justify-between gap-4">

          <div className="flex items-start gap-3">
            <div className="w-10 h-10 rounded-lg bg-industrial-900 border border-industrial-700 flex items-center justify-center">
              <FlaskConical className="w-5 h-5 text-industrial-sovereign" />
            </div>

            <div>
              <h2 className="text-lg font-bold text-slate-100">
                Netlib Benchmark Suite
              </h2>

              <p className="text-xs text-slate-400 mt-1">
                Authentic Netlib LP models executed through the complete
                SAKSHAM optimization pipeline.
              </p>
            </div>
          </div>

          <span className="text-[10px] font-mono px-2.5 py-1 rounded border bg-emerald-950 text-emerald-400 border-emerald-800">
            PHASE 11
          </span>

        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="p-3 rounded-lg bg-red-950/50 border border-red-800 text-red-300 text-xs">
          {error}
        </div>
      )}

      {/* Benchmark list */}
      <div className="bg-industrial-900 border border-industrial-800 rounded-xl overflow-hidden">

        <div className="px-5 py-4 border-b border-industrial-800">
          <h3 className="text-sm font-semibold text-slate-100">
            Authentic Benchmark Inputs
          </h3>

          <p className="text-xs text-slate-500 mt-1">
            Select a model to execute it through parsing, analysis, strategy,
            HiGHS, Sakshi verification, and certificate generation.
          </p>
        </div>

        {loading ? (
          <div className="p-10 flex justify-center">
            <Loader2 className="w-6 h-6 animate-spin text-industrial-sovereign" />
          </div>
        ) : (
          <div className="divide-y divide-industrial-800">

            {benchmarks.map((benchmark) => (
              <div
                key={benchmark.name}
                className="px-5 py-4 flex items-center justify-between gap-4 hover:bg-industrial-850/50 transition"
              >

                <div className="flex items-center gap-3">
                  <Database className="w-4 h-4 text-slate-500" />

                  <div>
                    <div className="text-sm font-semibold text-slate-200 uppercase">
                      {benchmark.name}
                    </div>

                    <div className="text-[11px] font-mono text-slate-500">
                      {benchmark.filename}
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => executeBenchmark(benchmark.name)}
                  disabled={running !== null}
                  className="flex items-center gap-2 px-3 py-2 rounded-md text-xs font-semibold
                             bg-industrial-800 border border-industrial-700
                             text-slate-200 hover:bg-industrial-700
                             disabled:opacity-50"
                >
                  {running === benchmark.name ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      RUNNING
                    </>
                  ) : (
                    <>
                      <Play className="w-3.5 h-3.5" />
                      RUN
                    </>
                  )}
                </button>

              </div>
            ))}

          </div>
        )}
      </div>

      {/* Result */}
      {selected && (
        <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-5 space-y-5">

          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-bold text-slate-100 uppercase">
                {selected.benchmark_name}
              </h3>

              <p className="text-[11px] font-mono text-slate-500 mt-1">
                {selected.source_identifier}
              </p>
            </div>

            <div className="flex items-center gap-2">
              {selected.success ? (
                <>
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  <span className="text-xs font-mono font-bold text-emerald-400">
                    SUCCESS
                  </span>
                </>
              ) : (
                <>
                  <XCircle className="w-5 h-5 text-red-400" />
                  <span className="text-xs font-mono font-bold text-red-400">
                    FAILED
                  </span>
                </>
              )}
            </div>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">

            <Metric
              icon={<Database />}
              label="Variables"
              value={
                selected.model_dimensions?.variables?.toString() || '—'
              }
            />

            <Metric
              icon={<Activity />}
              label="Constraints"
              value={
                selected.model_dimensions?.constraints?.toString() || '—'
              }
            />

            <Metric
              icon={<Database />}
              label="Nonzeros"
              value={selected.nonzeros?.toString() || '—'}
            />

            <Metric
              icon={<Clock />}
              label="Solve Time"
              value={
                selected.solve_time_seconds !== null
                  ? `${selected.solve_time_seconds.toFixed(4)} s`
                  : '—'
              }
            />

          </div>

          {/* Pipeline status */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">

            <StatusBox
              label="Parser"
              value={selected.parser_status}
            />

            <StatusBox
              label="Numerical Health"
              value={selected.numerical_health_status}
            />

            <StatusBox
              label="Solver"
              value={selected.solver_status}
            />

            <StatusBox
              label="Sakshi Verification"
              value={selected.sakshi_verification_status}
            />

            <StatusBox
              label="Certificate"
              value={selected.certificate_status}
            />

            <StatusBox
              label="Strategy"
              value={
                selected.recommended_strategy?.primary_algorithm || '—'
              }
            />

          </div>

          {/* Objective */}
          <div className="p-4 bg-industrial-950 border border-industrial-800 rounded-lg">

            <div className="text-[10px] uppercase font-mono text-slate-500">
              Objective Value
            </div>

            <div className="text-xl font-mono font-bold text-industrial-sovereign mt-1">
              {selected.objective_value !== null
                ? selected.objective_value
                : '—'}
            </div>

          </div>

          {/* Certificate */}
          {selected.certificate_digest && (
            <div className="p-4 bg-industrial-950 border border-industrial-800 rounded-lg">

              <div className="flex items-center gap-2">
                <Award className="w-4 h-4 text-industrial-accent" />

                <span className="text-xs font-semibold text-slate-200">
                  Solve Certificate
                </span>
              </div>

              <div className="mt-2 text-[10px] font-mono text-slate-500 break-all">
                SHA-256: {selected.certificate_digest}
              </div>

            </div>
          )}

          {/* Verification */}
          <div className="flex items-center gap-3 p-4 rounded-lg bg-emerald-950/30 border border-emerald-900">

            <ShieldCheck className="w-5 h-5 text-emerald-400" />

            <div>
              <div className="text-xs font-bold text-emerald-400">
                SAKSHI VERIFICATION
              </div>

              <div className="text-[11px] text-slate-400 mt-1">
                Independent verification status:{' '}
                <span className="font-mono text-emerald-300">
                  {selected.sakshi_verification_status}
                </span>
              </div>
            </div>

          </div>

          {/* Warnings */}
          {selected.warnings.length > 0 && (
            <div className="p-4 bg-amber-950/20 border border-amber-900 rounded-lg">

              <div className="text-xs font-semibold text-amber-400 mb-2">
                Pipeline Warnings
              </div>

              <ul className="space-y-1">
                {selected.warnings.map((warning, index) => (
                  <li
                    key={index}
                    className="text-[11px] text-slate-400"
                  >
                    • {warning}
                  </li>
                ))}
              </ul>

            </div>
          )}

        </div>
      )}

    </div>
  );
};

interface MetricProps {
  icon: React.ReactNode;
  label: string;
  value: string;
}

const Metric: React.FC<MetricProps> = ({ icon, label, value }) => (
  <div className="p-4 bg-industrial-950 border border-industrial-800 rounded-lg">

    <div className="flex items-center gap-2 text-slate-500">
      {React.cloneElement(
  icon as React.ReactElement<{ className?: string }>,
  { className: "w-5 h-5" }
)}

      <span className="text-[10px] uppercase font-mono">
        {label}
      </span>
    </div>

    <div className="text-sm font-mono font-bold text-slate-200 mt-2">
      {value}
    </div>

  </div>
);

interface StatusBoxProps {
  label: string;
  value: string;
}

const StatusBox: React.FC<StatusBoxProps> = ({ label, value }) => {
  const good = [
    'OK',
    'HEALTHY',
    'OPTIMAL',
    'VERIFIED',
    'GENERATED',
    'READY',
  ].includes(value);

  return (
    <div className="p-3 bg-industrial-950 border border-industrial-800 rounded-lg">

      <div className="text-[10px] font-mono uppercase text-slate-500">
        {label}
      </div>

      <div
        className={`text-xs font-mono font-bold mt-1 ${
          good
            ? 'text-emerald-400'
            : 'text-slate-300'
        }`}
      >
        {value}
      </div>

    </div>
  );
};