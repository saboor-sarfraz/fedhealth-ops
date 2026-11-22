"""Log the baseline, local, and federated evaluation results to MLflow."""

import csv
import re
import subprocess
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from hospital_clients.server import NUM_HOSPITALS, NUM_ROUNDS
from hospital_clients.training import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    SEED,
)
from train_baseline import CONTAMINATION, N_ESTIMATORS, RANDOM_STATE


ROOT_DIR = Path(__file__).resolve().parent
TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
ARTIFACT_LOCATION = (ROOT_DIR / "mlruns").as_uri()
EXPERIMENT_NAME = "FedHealth-Ops"
REGISTERED_MODEL_NAME = "fedhealth-ecg-anomaly-detector"
CHAMPION_ALIAS = "champion"
EVALUATION_DIR = ROOT_DIR / "outputs" / "evaluation"
DATA_DIR = ROOT_DIR / "data"
MD5_PATTERN = re.compile(r"^\s*-?\s*md5:\s*(\S+)\s*$", re.MULTILINE)
METRIC_COLUMNS = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1-score": "f1_score",
    "ROC-AUC": "roc_auc",
}
RUN_CONFIGS = {
    "Isolation Forest baseline (Hospital A)": {
        "run_name": "IsolationForest-Baseline",
        "model_type": "Isolation Forest baseline trained at Hospital A",
        "score_name": "anomaly scores",
        "checkpoint": (
            ROOT_DIR / "outputs" / "hospital_a_isolation_forest.joblib"
        ),
        "confusion_matrix": EVALUATION_DIR / "baseline_confusion_matrix.png",
        "training_hospitals": ["A"],
        "hyperparameters": {
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
            "contamination": CONTAMINATION,
        },
    },
    "Local-only (Hospital A)": {
        "run_name": "Local-Only-Autoencoder",
        "model_type": "Local-only convolutional autoencoder trained at Hospital A",
        "score_name": "reconstruction errors",
        "checkpoint": ROOT_DIR / "outputs" / "hospital_a_autoencoder.pth",
        "confusion_matrix": EVALUATION_DIR / "local_confusion_matrix.png",
        "training_hospitals": ["A"],
        "hyperparameters": {
            "batch_size": DEFAULT_BATCH_SIZE,
            "epochs": DEFAULT_EPOCHS,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "seed": SEED,
        },
    },
    "Federated global": {
        "run_name": "Federated-Global-Autoencoder",
        "model_type": "Federated convolutional autoencoder aggregated with FedAvg",
        "score_name": "reconstruction errors",
        "checkpoint": (
            ROOT_DIR / "outputs" / "federated_global_autoencoder.pth"
        ),
        "confusion_matrix": EVALUATION_DIR / "federated_confusion_matrix.png",
        "training_hospitals": ["A", "B", "C"],
        "hyperparameters": {
            "batch_size": DEFAULT_BATCH_SIZE,
            "epochs": DEFAULT_EPOCHS,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "seed": SEED,
            "num_rounds": NUM_ROUNDS,
            "num_hospitals": NUM_HOSPITALS,
            "aggregation": "FedAvg",
        },
    },
}


def load_evaluation_rows() -> dict[str, dict[str, str]]:
    """Read the three model results from the evaluation metrics CSV."""
    csv_path = EVALUATION_DIR / "model_comparison_metrics.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Evaluation metrics CSV not found: {csv_path}. "
            "Run evaluate_models.py first."
        )

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {
            "model",
            "threshold",
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


def read_dataset_version(dataset_name: str) -> str:
    """Return the md5 recorded in data/<dataset_name>.npz.dvc."""
    pointer_path = DATA_DIR / f"{dataset_name}.npz.dvc"
    if not pointer_path.is_file():
        raise FileNotFoundError(f"DVC pointer file not found: {pointer_path}")
    match = MD5_PATTERN.search(pointer_path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"No md5 hash found in DVC pointer file: {pointer_path}")
    return match.group(1)


def read_git_commit() -> str:
    """Return the current git commit hash, or "unknown" if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def get_or_create_experiment_id(client: MlflowClient) -> str:
    """Return the experiment ID, creating it with a local artifact root."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is not None:
        return experiment.experiment_id
    return client.create_experiment(
        EXPERIMENT_NAME,
        artifact_location=ARTIFACT_LOCATION,
    )


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
    report_path = require_artifact(EVALUATION_DIR / "evaluation_report.txt")
    test_version = read_dataset_version("global_test")
    calibration_version = read_dataset_version("hospital_A")
    git_commit = read_git_commit()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    experiment_id = get_or_create_experiment_id(client)
    logged_runs: list[tuple[str, str, float, set[str]]] = []

    for model_label, config in RUN_CONFIGS.items():
        row = evaluation_rows[model_label]
        checkpoint = require_artifact(config["checkpoint"])
        confusion_matrix_path = require_artifact(config["confusion_matrix"])
        threshold_percentile = int(row["threshold_percentile"])
        training_hospitals = config["training_hospitals"]

        params = {
            "model_type": config["model_type"],
            **config["hyperparameters"],
            "training_data": ",".join(
                f"hospital_{hospital}" for hospital in training_hospitals
            ),
            "threshold_calibration_method": (
                f"{threshold_percentile}th percentile of normal Hospital A "
                f"x_val {config['score_name']}"
            ),
            "evaluation_dataset": "data/global_test.npz (x, y)",
            "test_samples": int(row["test_samples"]),
            "calibration_samples": int(row["calibration_samples"]),
            "threshold": float(row["threshold"]),
            "dataset_version_test": test_version,
            "dataset_version_calibration": calibration_version,
        }
        for hospital in training_hospitals:
            params[f"dataset_version_hospital_{hospital}"] = read_dataset_version(
                f"hospital_{hospital}"
            )
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
            experiment_id=experiment_id,
            run_name=config["run_name"],
        ) as run:
            mlflow.set_tag("git_commit", git_commit)
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
                (
                    config["run_name"],
                    run.info.run_id,
                    metrics["ROC-AUC"],
                    expected_artifacts,
                )
            )

    for run_name, run_id, _, expected_artifacts in logged_runs:
        verify_run(client, run_id, experiment_id, expected_artifacts)
        print(f"Verified run: {run_name} ({run_id})")

    best_name, best_run_id, best_auc, _ = max(logged_runs, key=lambda run: run[2])
    if not client.search_registered_models(
        filter_string=f"name = '{REGISTERED_MODEL_NAME}'"
    ):
        client.create_registered_model(REGISTERED_MODEL_NAME)
    # The checkpoint is a plain file artifact, not an MLflow model flavor, so
    # mlflow.register_model (which requires a logged model) cannot be used.
    model_version = client.create_model_version(
        REGISTERED_MODEL_NAME,
        f"runs:/{best_run_id}/model",
        run_id=best_run_id,
    )
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        CHAMPION_ALIAS,
        model_version.version,
    )
    print(
        f"Registered model: {REGISTERED_MODEL_NAME} "
        f"version {model_version.version} "
        f"(alias '{CHAMPION_ALIAS}') from run {best_name} "
        f"({best_run_id}, ROC-AUC={best_auc:.4f})"
    )

    ui_command = (
        f"mlflow ui --backend-store-uri {TRACKING_URI} "
        "--host 127.0.0.1 --port 5000"
    )
    print(f"Experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
    print(f"Tracking location: {TRACKING_URI}")
    print(f"Start the MLflow UI with:\n{ui_command}")


if __name__ == "__main__":
    main()
