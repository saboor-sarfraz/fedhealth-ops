"""Train the classical Isolation Forest baseline on Hospital A normal ECG data."""

from pathlib import Path

import joblib
from sklearn.ensemble import IsolationForest

from hospital_clients.training import load_hospital_data


ROOT_DIR = Path(__file__).resolve().parent
BASELINE_MODEL_PATH = ROOT_DIR / "outputs" / "hospital_a_isolation_forest.joblib"
N_ESTIMATORS = 200
RANDOM_STATE = 42
CONTAMINATION = "auto"


def main() -> None:
    x_train, _ = load_hospital_data("A")

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train)

    BASELINE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, BASELINE_MODEL_PATH)

    print(f"Training samples: {len(x_train)}")
    print(f"Baseline model saved to: {BASELINE_MODEL_PATH}")


if __name__ == "__main__":
    main()
