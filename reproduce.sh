#!/usr/bin/env bash
# One-command reproduction: pull the versioned data and checkpoints, validate
# the data, fit the baseline, evaluate every model, and log/register the
# results in MLflow. By default the autoencoders are NOT retrained: the
# DVC-tracked checkpoints are evaluated, which reproduces the registered
# metrics exactly.
#
#   bash reproduce.sh                  # reproduce registered metrics (~1 min)
#   RETRAIN=1 bash reproduce.sh        # also retrain both autoencoders (seeded, ~50 min)
#   SKIP_DVC_PULL=1 bash reproduce.sh  # data/ and outputs/ already present, skip dvc pull
#   PYTHON=.venv/bin/python bash reproduce.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# Load DagsHub credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) from an
# untracked .env file when present; see DAGSHUB_SETUP.md.
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

PYTHON="${PYTHON:-python}"

echo "=== [1/7] Pull DVC-tracked data ==="
if [[ "${SKIP_DVC_PULL:-0}" == "1" ]]; then
    echo "SKIP_DVC_PULL=1: skipping dvc pull, using the existing data/ directory."
else
    if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
        echo "ERROR: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must both be set" >&2
        echo "to your DagsHub token before 'dvc pull'. See DAGSHUB_SETUP.md (step 4)." >&2
        echo "If data/ is already present, rerun with SKIP_DVC_PULL=1." >&2
        exit 1
    fi
    if ! "$PYTHON" -m dvc pull; then
        echo "ERROR: dvc pull failed. Check your DagsHub token and see DAGSHUB_SETUP.md." >&2
        exit 1
    fi
fi

echo "=== [2/7] Validate data (schema, quality, patient leakage) ==="
"$PYTHON" data_validation.py

echo "=== [3/7] Train Isolation Forest baseline (Hospital A) ==="
"$PYTHON" train_baseline.py

if [[ "${RETRAIN:-0}" == "1" ]]; then
    echo "=== [4/7] Train local-only autoencoder (Hospital A) ==="
    "$PYTHON" train_local.py

    echo "=== [5/7] Train federated autoencoder (Flower, 3 hospitals) ==="
    "$PYTHON" run_federated.py
else
    echo "=== [4-5/7] Using DVC-tracked autoencoder checkpoints (set RETRAIN=1 to retrain) ==="
fi

echo "=== [6/7] Evaluate all models on the global test set ==="
"$PYTHON" evaluate_models.py

echo "=== [7/7] Log runs and register the top model in MLflow ==="
"$PYTHON" track_experiments.py

echo "=== Done. Metrics: outputs/evaluation/model_comparison_metrics.csv; MLflow: mlflow.db + mlruns/ ==="
