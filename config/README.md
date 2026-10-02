# SAKSHAM Configuration Directory

This directory stores configuration files for various deployment environments:

- `default.json`: Baseline parameters across engine, backend, and frontend.
- Future configurations (e.g. `production.json`, `docker.json`, `hpc_cluster.json`) will be loaded hierarchically based on the `SAKSHAM_ENV` environment variable.
