export interface SystemHealthResponse {
  status: string;
  app_name: string;
  environment: string;
  engine: {
    name: string;
    version: string;
    status: string;
  };
  solvers: {
    highs: {
      available: boolean;
      version: string;
      status: string;
    };
    scipy: {
      available: boolean;
      version: string;
      status: string;
      role: string;
      solver_backend: boolean;
    };
  };
  scaffolding_phase: string;
}

export interface SystemInfoResponse {
  app_name: string;
  version: string;
  modules: Record<string, string>;
}

export interface PipelineError {
  stage: string;
  message: string;
  code: string | null;
}

export interface PipelineResultResponse {
  success: boolean;
  status: string;
  stage: string;
  input: {
    name: string | null;
    format: string | null;
  };
  parser_status: string;
  model_summary: {
    name: string;
    sense: string;
    num_variables: number;
    num_constraints: number;
    num_nonzeros: number;
    density: number;
    is_mip: boolean;
    variable_types: Record<string, number>;
    constraint_senses: Record<string, number>;
  } | null;
  fingerprint: {
    status: string;
    num_vars: number;
    num_constrs: number;
    num_nonzeros: number;
    density: number;
    tags: string[];
    conservation_rows_detected: number;
    knapsack_rows_detected: number;
    raw_metrics: Record<string, unknown>;
  } | null;
  health: {
    status: string;
    dynamic_range: number | null;
    estimated_condition_number: number | null;
    numerical_warning: boolean;
    coefficient_min_abs: number | null;
    coefficient_max_abs: number | null;
    nonzero_coefficient_count: number;
    stored_coefficient_count: number;
    explicit_zero_count: number;
    zero_row_count: number;
    zero_column_count: number;
    has_non_finite_coefficients: boolean;
    details: string;
  } | null;
  isg: {
    status: string;
    summary: Record<string, unknown>;
  } | null;
  strategy: {
    status: string;
    primary_strategy: string;
    primary_algorithm: string;
    confidence: string;
    rationale: string[];
    warnings: string[];
  } | null;
  selected_solver: string | null;
  solver_executed: boolean;
  solver_result: {
    status: string;
    success: boolean;
    objective_value: number | null;
    primal_solution: number[] | null;
    wall_time_seconds: number | null;
    iteration_count: number | null;
    solver_name: string;
    message: string;
    termination_reason: string | null;
  } | null;
  verification: {
    status: string;
    solver_status: string | null;
    bounds_satisfied: boolean | null;
    constraints_satisfied: boolean | null;
    integrality_satisfied: boolean | null;
    objective_consistent: boolean | null;
    independently_calculated_objective: number | null;
    max_bound_violation: number | null;
    max_constraint_violation: number | null;
    max_integrality_violation: number | null;
    objective_discrepancy: number | null;
    warnings: string[];
    unavailable_checks: string[];
  } | null;
  certificate: {
    status: string;
    verification_status: string;
    sha256_hash: string;
    optimality_claim: string;
    solver_reported_objective_value: number | null;
    independently_calculated_objective_value: number | null;
  } | null;
  warnings: string[];
  errors: PipelineError[];
}

export async function fetchHealth(): Promise<SystemHealthResponse> {
  const response = await fetch('/api/v1/health');
  if (!response.ok) {
    throw new Error(`Health check failed with status: ${response.status}`);
  }
  return response.json();
}

export async function fetchSystemInfo(): Promise<SystemInfoResponse> {
  const response = await fetch('/api/v1/system/info');
  if (!response.ok) {
    throw new Error(`System info failed with status: ${response.status}`);
  }
  return response.json();
}

export async function submitModel(file: File): Promise<PipelineResultResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch('/api/v1/solve', {
    method: 'POST',
    body: formData,
  });
  const result = await response.json() as PipelineResultResponse;
  if (!response.ok && !result.status) {
    throw new Error(`Model submission failed with status: ${response.status}`);
  }
  return result;
}

export interface NetlibBenchmarkMetadata {
  name: string;
  filename: string;
  source_type: string;
  source_identifier: string | null;
  expected_category: string | null;
  notes: string | null;
}

export interface NetlibBenchmarkListResponse {
  source_type: string;
  count: number;
  benchmarks: NetlibBenchmarkMetadata[];
}

export interface NetlibBenchmarkResult {
  benchmark_name: string;
  source_type: string;
  source_identifier: string | null;
  input_path: string | null;
  parser_status: string;
  model_dimensions: {
    variables: number;
    constraints: number;
  } | null;
  nonzeros: number | null;
  density: number | null;
  numerical_health_status: string;
  recommended_strategy: {
    primary_strategy: string;
    primary_algorithm: string;
    status: string;
    confidence: string;
  } | null;
  solver_name: string | null;
  solver_status: string;
  objective_value: number | null;
  solve_time_seconds: number | null;
  sakshi_verification_status: string;
  certificate_status: string;
  certificate_digest: string | null;
  failure_stage: string | null;
  failure_code: string | null;
  failure_message: string | null;
  warnings: string[];
  success: boolean;
}

export async function fetchNetlibBenchmarks(): Promise<NetlibBenchmarkListResponse> {
  const response = await fetch('/api/v1/benchmarks/netlib');

  if (!response.ok) {
    throw new Error(`Benchmark list failed with status: ${response.status}`);
  }

  return response.json();
}

export async function runNetlibBenchmark(
  benchmarkName: string
): Promise<NetlibBenchmarkResult> {
  const response = await fetch(
    `/api/v1/benchmarks/netlib/${encodeURIComponent(benchmarkName)}`
  );

  const result = await response.json() as NetlibBenchmarkResult;

  if (!response.ok) {
    throw new Error(
      result.failure_message ||
      `Benchmark execution failed with status: ${response.status}`
    );
  }

  return result;
}
