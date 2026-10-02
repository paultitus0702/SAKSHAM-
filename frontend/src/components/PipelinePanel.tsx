import React, { useState } from 'react';
import { AlertTriangle, FileUp, LoaderCircle, Play, ShieldCheck } from 'lucide-react';
import { submitModel } from '../api/client';
import type { PipelineResultResponse } from '../api/client';

const formatNumber = (value: number | null | undefined): string => {
  if (value === null || value === undefined || !Number.isFinite(value)) return 'Unavailable';
  return new Intl.NumberFormat('en', { maximumSignificantDigits: 7 }).format(value);
};

const metricValue = (summary: Record<string, unknown> | null, key: string): string => {
  const value = summary?.[key];
  return typeof value === 'number' ? formatNumber(value) : 'Unavailable';
};

const statusColor = (status: string): string => {
  if (status === 'VERIFIED' || status === 'HEALTHY' || status === 'COMPLETE') {
    return 'text-emerald-400';
  }
  if (status === 'FAILED' || status === 'INVALID') return 'text-red-400';
  return 'text-amber-300';
};

export const PipelinePanel: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<PipelineResultResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0] ?? null;
    setResult(null);
    setError(null);
    if (nextFile && !/\.(lp|mps)$/i.test(nextFile.name)) {
      setFile(null);
      setError('Select an LP or MPS model file.');
      event.target.value = '';
      return;
    }
    setFile(nextFile);
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file || running) return;
    setRunning(true);
    setResult(null);
    setError(null);
    try {
      setResult(await submitModel(file));
    } catch (submissionError) {
      setError(
        submissionError instanceof Error
          ? submissionError.message
          : 'The model pipeline could not be reached.',
      );
    } finally {
      setRunning(false);
    }
  };

  return (
    <section className="space-y-5 border-b border-industrial-800 pb-6" aria-labelledby="pipeline-title">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 id="pipeline-title" className="text-base font-semibold text-slate-100">
            Run optimization pipeline
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            LP or MPS input through analysis, HiGHS, SAKSHI, and certificate generation.
          </p>
        </div>
        <span className="font-mono text-[11px] text-slate-500">SYNCHRONOUS RUN</span>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <label
          htmlFor="model-file"
          className="flex min-h-11 min-w-0 flex-1 cursor-pointer items-center gap-3 border border-industrial-700 bg-industrial-950 px-3 py-2 text-sm text-slate-300 hover:border-industrial-sovereign"
        >
          <FileUp className="h-4 w-4 shrink-0 text-industrial-sovereign" aria-hidden="true" />
          <span className="truncate">{file?.name ?? 'Choose an LP or MPS file'}</span>
          <input
            id="model-file"
            className="sr-only"
            type="file"
            accept=".lp,.mps"
            onChange={handleFileChange}
            disabled={running}
          />
        </label>
        <button
          type="submit"
          disabled={!file || running}
          className="inline-flex min-h-11 items-center justify-center gap-2 bg-industrial-sovereign px-4 text-sm font-semibold text-industrial-950 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-45"
        >
          {running ? (
            <LoaderCircle className="h-4 w-4 animate-spin" aria-hidden="true" />
          ) : (
            <Play className="h-4 w-4" aria-hidden="true" />
          )}
          {running ? 'Running' : 'Run model'}
        </button>
      </form>

      {running && (
        <p className="text-xs text-industrial-accent" role="status" aria-live="polite">
          Uploading and running parsing, analysis, solve, verification, and certificate stages.
        </p>
      )}
      {error && (
        <p className="flex items-center gap-2 text-sm text-red-300" role="alert">
          <AlertTriangle className="h-4 w-4 shrink-0" aria-hidden="true" />
          {error}
        </p>
      )}

      {result && (
        <div className="space-y-5 border-t border-industrial-800 pt-5" aria-live="polite">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-slate-100">
                {result.input.name ?? 'Model result'}
              </p>
              <p className="mt-1 font-mono text-[11px] text-slate-500">
                {result.input.format?.toUpperCase() ?? 'UNKNOWN'} / {result.stage}
              </p>
            </div>
            <span className={`font-mono text-sm font-semibold ${statusColor(result.status)}`}>
              {result.status}
            </span>
          </div>

          {result.model_summary && (
            <div className="grid gap-4 border-y border-industrial-800 py-4 sm:grid-cols-2 lg:grid-cols-4">
              <div>
                <p className="text-[10px] font-mono uppercase text-slate-500">Model</p>
                <p className="mt-1 text-sm text-slate-200">
                  {result.model_summary.sense} / {result.model_summary.is_mip ? 'MIP' : 'LP'}
                </p>
                <p className="text-xs text-slate-400">
                  {result.model_summary.num_variables} variables / {result.model_summary.num_constraints} constraints
                </p>
              </div>
              <div>
                <p className="text-[10px] font-mono uppercase text-slate-500">Structure</p>
                <p className="mt-1 text-sm text-slate-200">
                  {result.model_summary.num_nonzeros} nonzeros / {formatNumber(result.model_summary.density)} density
                </p>
                <p className="text-xs text-slate-400">
                  {result.fingerprint?.tags.join(', ') || 'No tags'}
                </p>
              </div>
              <div>
                <p className="text-[10px] font-mono uppercase text-slate-500">Numerical health</p>
                <p className={`mt-1 text-sm font-semibold ${statusColor(result.health?.status ?? 'UNAVAILABLE')}`}>
                  {result.health?.status ?? 'UNAVAILABLE'}
                </p>
                <p className="text-xs text-slate-400">
                  Dynamic range {formatNumber(result.health?.dynamic_range)}
                </p>
              </div>
              <div>
                <p className="text-[10px] font-mono uppercase text-slate-500">ISG</p>
                <p className="mt-1 text-sm text-slate-200">
                  {metricValue(result.isg?.summary ?? null, 'variable_constraint_edge_count')} variable-constraint edges
                </p>
                <p className="text-xs text-slate-400">
                  {metricValue(result.isg?.summary ?? null, 'objective_edge_count')} objective edges
                </p>
              </div>
            </div>
          )}

          {result.strategy && (
            <div className="space-y-2">
              <h3 className="text-[10px] font-mono uppercase text-slate-500">Strategy recommendation</h3>
              <p className="text-sm text-slate-200">
                {result.strategy.primary_strategy}
                <span className="ml-2 text-xs text-slate-400">
                  {result.strategy.primary_algorithm} / {result.strategy.confidence} confidence
                </span>
              </p>
              <ul className="list-inside list-disc space-y-1 text-xs text-slate-400">
                {result.strategy.rationale.map((reason, index) => <li key={index}>{reason}</li>)}
              </ul>
            </div>
          )}

          {result.solver_result && (
            <div className="grid gap-4 border-t border-industrial-800 pt-4 sm:grid-cols-3">
              <div>
                <p className="text-[10px] font-mono uppercase text-slate-500">Solver</p>
                <p className="mt-1 text-sm text-slate-200">
                  {result.solver_executed ? result.selected_solver : 'Not executed'}
                </p>
                <p className={`text-xs ${statusColor(result.solver_result.status)}`}>
                  {result.solver_result.status}
                </p>
              </div>
              <div>
                <p className="text-[10px] font-mono uppercase text-slate-500">Solver objective</p>
                <p className="mt-1 font-mono text-sm text-slate-200">
                  {formatNumber(result.solver_result.objective_value)}
                </p>
                <p className="text-xs text-slate-400">
                  {result.solver_result.primal_solution
                    ? `Primal vector: [${result.solver_result.primal_solution.map(formatNumber).join(', ')}]`
                    : 'Primal vector unavailable'}
                </p>
              </div>
              <div>
                <p className="text-[10px] font-mono uppercase text-slate-500">SAKSHI</p>
                <p className={`mt-1 text-sm font-semibold ${statusColor(result.verification?.status ?? 'UNAVAILABLE')}`}>
                  {result.verification?.status ?? 'UNAVAILABLE'}
                </p>
                <p className="text-xs text-slate-400">
                  Max constraint violation {formatNumber(result.verification?.max_constraint_violation)}
                </p>
              </div>
            </div>
          )}

          {result.certificate && (
            <div className="flex gap-3 border-t border-industrial-800 pt-4">
              <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-industrial-sovereign" aria-hidden="true" />
              <div className="min-w-0 space-y-1">
                <p className="text-xs text-slate-300">
                  Certificate / verification {result.certificate.verification_status} / optimality {result.certificate.optimality_claim}
                </p>
                <p className="break-all font-mono text-[11px] text-slate-500">
                  SHA-256 {result.certificate.sha256_hash}
                </p>
              </div>
            </div>
          )}

          {result.errors.length > 0 && (
            <ul className="space-y-1 border-t border-red-900/70 pt-3 text-xs text-red-300" role="alert">
              {result.errors.map((item, index) => (
                <li key={`${item.stage}-${index}`}>
                  <span className="font-mono">{item.stage}</span>: {item.message}
                </li>
              ))}
            </ul>
          )}
          {result.warnings.length > 0 && (
            <ul className="space-y-1 border-t border-amber-900/50 pt-3 text-xs text-amber-200/80">
              {result.warnings.map((warning, index) => <li key={index}>{warning}</li>)}
            </ul>
          )}
        </div>
      )}
    </section>
  );
};