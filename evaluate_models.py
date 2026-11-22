"""Compare the Isolation Forest baseline with the local and federated ECG
autoencoders on the held-out test set."""

import csv
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from data_validation import validate_test_file
from hospital_clients.training import (
    DEFAULT_BATCH_SIZE,
    get_reconstruction_errors,
    load_hospital_data,
)
from models.autoencoder import ECGAutoencoder


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs" / "evaluation"
THRESHOLD_PERCENTILE = 95
BASELINE_LABEL = "Isolation Forest baseline (Hospital A)"
LOCAL_LABEL = "Local-only (Hospital A)"
FEDERATED_LABEL = "Federated global"
BASELINE_MODEL_PATH = ROOT_DIR / "outputs" / "hospital_a_isolation_forest.joblib"
LOCAL_CHECKPOINT_PATH = ROOT_DIR / "outputs" / "hospital_a_autoencoder.pth"
FEDERATED_CHECKPOINT_PATH = ROOT_DIR / "outputs" / "federated_global_autoencoder.pth"
CONFUSION_MATRIX_FILENAMES = {
    BASELINE_LABEL: "baseline_confusion_matrix.png",
    LOCAL_LABEL: "local_confusion_matrix.png",
    FEDERATED_LABEL: "federated_confusion_matrix.png",
}

# Maps samples of shape (N, 200) to N anomaly scores; higher means more anomalous.
ScoreFunction = Callable[[np.ndarray], np.ndarray]


def load_model(checkpoint_path: Path, device: torch.device) -> ECGAutoencoder:
    """Load a saved autoencoder state dict onto the evaluation device."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    model = ECGAutoencoder().to(device)
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_baseline(model_path: Path) -> IsolationForest:
    """Load the fitted Isolation Forest baseline."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Baseline model not found: {model_path}")
    return joblib.load(model_path)


def build_score_functions(device: torch.device) -> dict[str, ScoreFunction]:
    """Load every model and return its scoring function in reporting order."""
    baseline = load_baseline(BASELINE_MODEL_PATH)
    local_model = load_model(LOCAL_CHECKPOINT_PATH, device)
    federated_model = load_model(FEDERATED_CHECKPOINT_PATH, device)

    def autoencoder_scores(model: ECGAutoencoder) -> ScoreFunction:
        def score(samples: np.ndarray) -> np.ndarray:
            return get_reconstruction_errors(
                samples,
                model,
                batch_size=DEFAULT_BATCH_SIZE,
                device=device,
            )

        return score

    def baseline_scores(samples: np.ndarray) -> np.ndarray:
        return -baseline.score_samples(samples)

    return {
        BASELINE_LABEL: baseline_scores,
        LOCAL_LABEL: autoencoder_scores(local_model),
        FEDERATED_LABEL: autoencoder_scores(federated_model),
    }


def plot_confusion_matrix(
    matrix: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Save an annotated normal/anomaly confusion matrix."""
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=(0, 1),
        yticks=(0, 1),
        xticklabels=("Normal", "Anomalous"),
        yticklabels=("Normal", "Anomalous"),
        xlabel="Predicted label",
        ylabel="True label",
        title=f"{model_name} confusion matrix",
    )
    axis.xaxis.tick_bottom()
    midpoint = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > midpoint else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    # All models use the same normal Hospital A validation split for calibration.
    _, x_val = load_hospital_data("A")
    x_test, y_test = validate_test_file(ROOT_DIR / "data" / "global_test.npz")
    results: list[dict[str, float | int | str]] = []
    roc_curves: list[tuple[str, np.ndarray, np.ndarray, float]] = []

    score_functions = build_score_functions(device)

    print(f"Using device: {device}")
    print(
        f"Calibration: {len(x_val)} normal Hospital A validation samples, "
        f"{THRESHOLD_PERCENTILE}th percentile"
    )
    print(f"Held-out test set: {len(x_test)} samples")

    for model_name, score_function in score_functions.items():
        validation_scores = score_function(x_val)
        threshold = float(np.percentile(validation_scores, THRESHOLD_PERCENTILE))

        # Test labels are used only for evaluation, never threshold calibration.
        test_scores = score_function(x_test)
        predictions = (test_scores > threshold).astype(np.int8)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        auc = float(roc_auc_score(y_test, test_scores))
        fpr, tpr, _ = roc_curve(y_test, test_scores)

        tn, fp, fn, tp = (int(value) for value in matrix.ravel())
        results.append(
            {
                "model": model_name,
                "threshold": threshold,
                "threshold_percentile": THRESHOLD_PERCENTILE,
                "calibration_samples": len(x_val),
                "test_samples": len(x_test),
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision": float(
                    precision_score(y_test, predictions, zero_division=0)
                ),
                "recall": float(recall_score(y_test, predictions, zero_division=0)),
                "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
                "roc_auc": auc,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
            }
        )
        roc_curves.append((model_name, fpr, tpr, auc))

        plot_confusion_matrix(
            matrix,
            model_name,
            OUTPUT_DIR / CONFUSION_MATRIX_FILENAMES[model_name],
        )
        print(
            f"{model_name}: threshold={threshold:.8f}, "
            f"F1={results[-1]['f1_score']:.4f}, ROC-AUC={auc:.4f}"
        )

    metrics_path = OUTPUT_DIR / "model_comparison_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    figure, axis = plt.subplots(figsize=(7, 6))
    for model_name, fpr, tpr, auc in roc_curves:
        axis.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})")
    axis.plot((0, 1), (0, 1), linestyle="--", color="gray", label="Chance")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC curve comparison",
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "roc_curve_comparison.png", dpi=160)
    plt.close(figure)

    report_path = OUTPUT_DIR / "evaluation_report.txt"
    report_lines = [
        "ECG Anomaly Detection Model Evaluation",
        "======================================",
        f"Test samples evaluated: {len(x_test)}",
        f"Normal calibration samples: {len(x_val)} (Hospital A x_val)",
        (
            f"Threshold method: {THRESHOLD_PERCENTILE}th percentile of each "
            "model's anomaly scores on the same normal validation data."
        ),
        (
            "Higher anomaly score indicates an anomaly (autoencoders: "
            "reconstruction error; Isolation Forest: -score_samples)."
        ),
        "Test labels were used only to calculate evaluation metrics.",
        "",
    ]
    for result in results:
        report_lines.extend(
            [
                str(result["model"]),
                "-" * len(str(result["model"])),
                f"Threshold: {result['threshold']:.8f}",
                f"Accuracy:  {result['accuracy']:.4f}",
                f"Precision: {result['precision']:.4f}",
                f"Recall:    {result['recall']:.4f}",
                f"F1-score:  {result['f1_score']:.4f}",
                f"ROC-AUC:   {result['roc_auc']:.4f}",
                (
                    "Confusion matrix [[TN, FP], [FN, TP]]: "
                    f"[[{result['true_negatives']}, {result['false_positives']}], "
                    f"[{result['false_negatives']}, {result['true_positives']}]]"
                ),
                "",
            ]
        )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Evaluation outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
