# MLflow Experiment Tracking Guide

This project logs evaluation results to a local MLflow tracking database
(`mlflow.db`) with artifacts stored under `mlruns/`. It does not need a cloud
account, tracking server, or additional Docker service.

## Install and log the evaluation runs

From the project root, install the project requirements and run the tracker:

```bash
pip install -r requirements.txt
python track_experiments.py
```

The tracker reads the results from `outputs/evaluation/`, then creates three
runs in the `FedHealth-Ops` experiment, baseline first:

- `IsolationForest-Baseline`
- `Local-Only-Autoencoder`
- `Federated-Global-Autoencoder`

Each run logs:

- **Metrics:** accuracy, precision, recall, F1-score, and ROC-AUC.
- **Hyperparameters:**
  - Baseline: `n_estimators`, `random_state`, `contamination`.
  - Local autoencoder: `batch_size`, `epochs`, `learning_rate`.
  - Federated autoencoder: the same three plus `num_rounds`, `num_hospitals`,
    and `aggregation` (`FedAvg`).
- **Training data:** `training_data` (`hospital_A` for the baseline and local
  runs, `hospital_A,hospital_B,hospital_C` for the federated run), plus the
  threshold calibration parameters.
- **Dataset versions:** the DVC md5 hash read from the `data/*.dvc` pointer
  files: `dataset_version_test` (`global_test`), `dataset_version_calibration`
  (`hospital_A`), and one `dataset_version_hospital_<x>` per training hospital.
- **Git commit:** the tag `git_commit` (`unknown` if git is unavailable).
- **Artifacts:** the model checkpoint (`model/`), the evaluation report, and the
  confusion matrix image (`evaluation/`).

Run the tracker again after regenerating the evaluation CSV to record another
set of three runs.

## Model registry

After the runs are logged and their artifacts verified, the tracker picks the
run with the highest ROC-AUC and registers its `model` artifact as
`fedhealth-ecg-anomaly-detector`. Every execution creates a new registered
model version and moves the `champion` alias to it. The tracker prints the model
name, version, and source run. To look up the current champion:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
print(client.get_model_version_by_alias("fedhealth-ecg-anomaly-detector", "champion"))
```

## View the runs

After tracking completes, start the MLflow UI from the project root with the
command printed by the script:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000), select the
`FedHealth-Ops` experiment, and open any run to view its metrics, parameters,
tags, and artifacts. Registered versions and the `champion` alias are listed on
the **Models** tab. Stop the local UI with `Ctrl+C`.
