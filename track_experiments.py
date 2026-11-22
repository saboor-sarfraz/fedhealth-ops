"""Log the local and federated evaluation results to local MLflow tracking."""

import csv
import os
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


ROOT_DIR = Path(__file__).resolve().parent
TRACKING_DIR = ROOT_DIR / "mlruns"
EXPERIMENT_NAME = "FedHealth-Ops"
METRIC_COLUMNS = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1-score": "f1_score",
    "ROC-AUC": "roc_auc",
}
RUN_CONFIGS = {
    "Local-only (Hospital A)": {
        "run_name": "Local-Only-Autoencoder",
        "model_type": "Local-only convolutional autoencoder trained at Hospital A",
        "checkpoint": ROOT_DIR / "outputs" / "hospital_a_autoencoder.pth",
        "confusion_matrix": (
            ROOT_DIR / "outputs" / "evaluation" / "local_confusion_matrix.png"
        ),
    },
    "Federated global": {
        "run_name": "Federated-Global-Autoencoder",
        "model_type": "Federated convolutional autoencoder aggregated with FedAvg",
        "checkpoint": (
            ROOT_DIR / "outputs" / "federated_global_autoencoder.pth"
        ),
        "confusion_matrix": (
            ROOT_DIR
            / "outputs"
            / "evaluation"
            / "federated_confusion_matrix.png"
        ),
    },
}


def load_evaluation_rows() -> dict[str, dict[str, str]]:
    """Read the two model results from the evaluation metrics CSV."""
    csv_path = ROOT_DIR / "outputs" / "evaluation" / "model_comparison_metrics.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Evaluation metrics CSV not found: {csv_path}. "
            "Run evaluate_models.py first."
        )

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {
            "model",
            "threshold_percentile",
            "calibration_samples",
            "test_samples",
            *METRIC_COLUMNS.values(),
        }
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"{csv_path} is missing columns: "
                f"{', '.join(sorted(missing_columns))}"
            )
        rows = {row["model"]: row for row in reader}

    missing_models = set(RUN_CONFIGS).difference(rows)
    if missing_models:
        raise ValueError(
            f"{csv_path} is missing model results: "
            f"{', '.join(sorted(missing_models))}"
        )
    return rows


def require_artifact(path: Path) -> Path:
    """Return an artifact path or raise a clear error if it is unavailable."""
    if not path.is_file():
        raise FileNotFoundError(f"Required MLflow artifact not found: {path}")
    return path


def verify_run(
    client: MlflowClient,
    run_id: str,
    experiment_id: str,
    expected_artifacts: set[str],
) -> None:
    """Check that the run belongs to the experiment and has its artifacts."""
    run = client.get_run(run_id)
    if run.info.experiment_id != experiment_id:
        raise RuntimeError(f"Run {run_id} was logged to the wrong experiment")

    artifact_paths = set()
    for folder in ("model", "evaluation"):
        artifact_paths.update(
            item.path.replace("\\", "/")
            for item in client.list_artifacts(run_id, folder)
            if not item.is_dir
        )
    missing_artifacts = expected_artifacts.difference(artifact_paths)
    if missing_artifacts:
        raise RuntimeError(
            f"Run {run_id} is missing artifacts: "
            f"{', '.join(sorted(missing_artifacts))}"
        )


def main() -> None:
    evaluation_rows = load_evaluation_rows()
    evaluation_dir = ROOT_DIR / "outputs" / "evaluation"
    report_path = require_artifact(evaluation_dir / "evaluation_report.txt")

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = TRACKING_DIR.as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=tracking_uri)
    logged_runs: list[tuple[str, str, set[str]]] = []

    for model_label, config in RUN_CONFIGS.items():
        row = evaluation_rows[model_label]
        checkpoint = require_artifact(config["checkpoint"])
        confusion_matrix_path = require_artifact(config["confusion_matrix"])
        threshold_percentile = int(row["threshold_percentile"])

        params = {
            "model_type": config["model_type"],
            "threshold_calibration_method": (
                f"{threshold_percentile}th percentile of normal Hospital A "
                "x_val reconstruction errors"
            ),
            "evaluation_dataset": "data/global_test.npz (x, y)",
            "test_samples": int(row["test_samples"]),
            "calibration_samples": int(row["calibration_samples"]),
            "threshold": float(row["threshold"]),
        }
        expected_artifacts = {
            f"model/{checkpoint.name}",
            f"evaluation/{report_path.name}",
            f"evaluation/{confusion_matrix_path.name}",
        }
        metrics = {
            mlflow_name: float(row[csv_name])
            for mlflow_name, csv_name in METRIC_COLUMNS.items()
        }

        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=config["run_name"],
        ) as run:
            mlflow.log_params(params)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            mlflow.log_artifact(str(checkpoint), artifact_path="model")
            mlflow.log_artifact(str(report_path), artifact_path="evaluation")
            mlflow.log_artifact(
                str(confusion_matrix_path),
                artifact_path="evaluation",
            )
            logged_runs.append(
                (config["run_name"], run.info.run_id, expected_artifacts)
            )

    for run_name, run_id, expected_artifacts in logged_runs:
        verify_run(
            client,
            run_id,
            experiment.experiment_id,
            expected_artifacts,
        )
        print(f"Verified run: {run_name} ({run_id})")

    ui_command = (
        '$env:MLFLOW_ALLOW_FILE_STORE = "true"\n'
        f'& "{sys.executable}" -m mlflow ui '
        f'--backend-store-uri "{TRACKING_DIR}" '
        "--host 127.0.0.1 --port 5000"
    )
    print(f"Experiment: {EXPERIMENT_NAME} (ID: {experiment.experiment_id})")
    print(f"Tracking location: {TRACKING_DIR}")
    print(f"Start the MLflow UI in PowerShell with:\n{ui_command}")


if __name__ == "__main__":
    main()
