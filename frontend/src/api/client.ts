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
    };
  };
  scaffolding_phase: string;
}

export interface SystemInfoResponse {
  app_name: string;
  version: string;
  modules: Record<string, string>;
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
