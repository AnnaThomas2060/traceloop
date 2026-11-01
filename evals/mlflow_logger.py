import json
from pathlib import Path
from typing import Any

from app.config import get_settings

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    mlflow = None


class MLflowLogger:
    """Logs evaluation runs, parameters, metrics, and artifacts to MLflow or local files."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
    ):
        settings = get_settings()
        self.tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI
        self.experiment_name = experiment_name or settings.MLFLOW_EXPERIMENT_NAME
        self._mlflow_active = False

        if HAS_MLFLOW and mlflow is not None:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                self._mlflow_active = True
            except Exception:
                self._mlflow_active = False

    def log_evaluation_run(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        artifacts: dict[str, Any] | None = None,
    ) -> str:
        run_id = f"local-{run_name}"

        if self._mlflow_active:
            try:
                with mlflow.start_run(run_name=run_name) as active_run:
                    run_id = active_run.info.run_id
                    for k, v in params.items():
                        mlflow.log_param(k, str(v))
                    for k, v in metrics.items():
                        mlflow.log_metric(k, float(v))
                    if artifacts:
                        tmp_dir = Path("evals/artifacts") / run_id
                        tmp_dir.mkdir(parents=True, exist_ok=True)
                        for fname, data in artifacts.items():
                            fpath = tmp_dir / fname
                            if isinstance(data, (dict, list)):
                                fpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
                            else:
                                fpath.write_text(str(data), encoding="utf-8")
                            mlflow.log_artifact(str(fpath))
                return run_id
            except Exception:
                pass

        # Fallback local artifact logging
        local_runs_dir = Path("evals/runs") / run_name
        local_runs_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "run_id": run_id,
            "params": params,
            "metrics": metrics,
        }
        (local_runs_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if artifacts:
            for fname, data in artifacts.items():
                fpath = local_runs_dir / fname
                if isinstance(data, (dict, list)):
                    fpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
                else:
                    fpath.write_text(str(data), encoding="utf-8")

        return run_id
