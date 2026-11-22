# MLflow Experiment Tracking Guide

This project logs evaluation results to a local MLflow tracking directory. It
does not need a cloud account, tracking server, or additional Docker service.

## Install and log the evaluation runs

From the project root, install the project requirements into the existing
Conda environment and run the tracker:

```powershell
.\venv\python.exe -m pip install -r requirements.txt
.\venv\python.exe track_experiments.py
```

The tracker reads the results from `outputs/evaluation/`, then creates two runs
in the `FedHealth-Ops` experiment:

- `Local-Only-Autoencoder`
- `Federated-Global-Autoencoder`

Each run logs accuracy, precision, recall, F1-score, and ROC-AUC; model and
calibration parameters; its model checkpoint; the evaluation report; and its
confusion matrix image. The local tracking data is stored in `mlruns/` in the
project root. Run the tracker again after regenerating the evaluation CSV to
record another pair of runs.

## View the runs

After tracking completes, use the MLflow UI commands printed by the script. In
PowerShell, from the project root, they have this form:

```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
.\venv\python.exe -m mlflow ui --backend-store-uri ".\mlruns" --host 127.0.0.1 --port 5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000), select the
`FedHealth-Ops` experiment, and open either run to view its metrics, parameters,
and artifacts. Stop the local UI with `Ctrl+C`.
