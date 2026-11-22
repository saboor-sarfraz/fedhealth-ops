"""Compare local and federated ECG autoencoders on the held-out test set."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from hospital_clients.training import (
    DEFAULT_BATCH_SIZE,
    get_reconstruction_errors,
    load_hospital_data,
)
from models.autoencoder import ECGAutoencoder


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs" / "evaluation"
THRESHOLD_PERCENTILE = 95
MODEL_CHECKPOINTS = {
    "Local-only (Hospital A)": ROOT_DIR
    / "outputs"
    / "hospital_a_autoencoder.pth",
    "Federated global": ROOT_DIR
    / "outputs"
    / "federated_global_autoencoder.pth",
}


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


def load_test_data() -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the held-out ECG samples and binary labels."""
    test_path = ROOT_DIR / "data" / "global_test.npz"
    if not test_path.is_file():
        raise FileNotFoundError(f"Global test dataset not found: {test_path}")

    with np.load(test_path, allow_pickle=False) as dataset:
        missing = {"x", "y"}.difference(dataset.files)
        if missing:
            raise ValueError(
                f"{test_path} is missing required arrays: "
                f"{', '.join(sorted(missing))}"
            )
        x_test = dataset["x"]
        y_test = dataset["y"]

    if x_test.ndim != 2 or x_test.shape[1] != 200:
        raise ValueError(
            f"Test inputs must have shape (N, 200); received {x_test.shape}"
        )
    if y_test.ndim != 1 or len(y_test) != len(x_test):
        raise ValueError(
            "Test labels must be a one-dimensional array with one label "
            "per input sample"
        )
    if not np.isin(y_test, [0, 1]).all():
        raise ValueError("Test labels must contain only 0 (normal) and 1 (anomaly)")
    if np.unique(y_test).size != 2:
        raise ValueError("ROC-AUC evaluation requires both test label classes")

    return x_test, y_test.astype(np.int8, copy=False)


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

    # Both models use the same normal Hospital A validation split for calibration.
    _, x_val = load_hospital_data("A")
    x_test, y_test = load_test_data()
    results: list[dict[str, float | int | str]] = []
    roc_curves: list[tuple[str, np.ndarray, np.ndarray, float]] = []

    print(f"Using device: {device}")
    print(
        f"Calibration: {len(x_val)} normal Hospital A validation samples, "
        f"{THRESHOLD_PERCENTILE}th percentile"
    )
    print(f"Held-out test set: {len(x_test)} samples")

    for model_name, checkpoint_path in MODEL_CHECKPOINTS.items():
        model = load_model(checkpoint_path, device)
        validation_errors = get_reconstruction_errors(
            x_val,
            model,
            batch_size=DEFAULT_BATCH_SIZE,
            device=device,
        )
        threshold = float(np.percentile(validation_errors, THRESHOLD_PERCENTILE))

        # Test labels are used only for evaluation, never threshold calibration.
        test_errors = get_reconstruction_errors(
            x_test,
            model,
            batch_size=DEFAULT_BATCH_SIZE,
            device=device,
        )
        predictions = (test_errors > threshold).astype(np.int8)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        auc = float(roc_auc_score(y_test, test_errors))
        fpr, tpr, _ = roc_curve(y_test, test_errors)

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

        plot_name = (
            "local_confusion_matrix.png"
            if model_name.startswith("Local-only")
            else "federated_confusion_matrix.png"
        )
        plot_confusion_matrix(
            matrix,
            model_name,
            OUTPUT_DIR / plot_name,
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
        "ECG Autoencoder Evaluation",
        "=========================",
        f"Test samples evaluated: {len(x_test)}",
        f"Normal calibration samples: {len(x_val)} (Hospital A x_val)",
        (
            f"Threshold method: {THRESHOLD_PERCENTILE}th percentile of each "
            "model's reconstruction errors on the same normal validation data."
        ),
        "Higher reconstruction error indicates an anomaly.",
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
