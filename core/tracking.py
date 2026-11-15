"""Optional MLflow logging (off by default via ``config.use_mlflow``).

When MLflow is disabled every function is a no-op and the package is never imported.
"""

import config as c


class Tracker:
    """One MLflow run, or nothing when ``enabled`` is false."""

    def __init__(self, enabled, run_name, params=None):
        self.enabled = enabled
        if not enabled:
            return
        import mlflow
        self._mlflow = mlflow
        mlflow.set_tracking_uri(c.mlflow_tracking_uri)
        mlflow.set_experiment(c.mlflow_experiment_name)
        mlflow.start_run(run_name=run_name)
        if params:
            mlflow.log_params({k: str(v) for k, v in params.items()})

    def metric(self, name, value, step=None):
        if self.enabled and value == value:   # skip NaN
            self._mlflow.log_metric(name, float(value), step=step)

    def artifact(self, path):
        if self.enabled:
            self._mlflow.log_artifact(path)

    def close(self):
        if self.enabled:
            self._mlflow.end_run()
