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
    tags: string[];
    conservation_rows_detected: number;
    knapsack_rows_detected: number;
    raw_metrics: Record<string, unknown>;
  } | null;
  health: {
    status: string;
    dynamic_range: number | null;
    numerical_warning: boolean;
    zero_row_count: number;
    zero_column_count: number;
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
