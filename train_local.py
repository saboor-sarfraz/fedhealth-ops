
import json
import numpy as np
import torch

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

from models.autoencoder import ECGAutoencoder
from hospital_clients.training import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    get_reconstruction_errors,
    load_hospital_data,
    train_local_model,
)
import os
os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = DEFAULT_BATCH_SIZE
EPOCHS = DEFAULT_EPOCHS
LEARNING_RATE = DEFAULT_LEARNING_RATE

THRESHOLD_PERCENTILE = 95

print("Using device:", DEVICE)


# --------------------------------------------------
# 2. Load hospital A dataset
# --------------------------------------------------

X_train, X_val = load_hospital_data("A")

print("\nHospital A")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))


# --------------------------------------------------
# 3. Initialize model
# --------------------------------------------------

model = ECGAutoencoder().to(DEVICE)


# --------------------------------------------------
# 4. Training and validation
# --------------------------------------------------

model, training_metrics = train_local_model(
    model,
    X_train,
    X_val,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    device=DEVICE,
    best_model_path="outputs/hospital_a_autoencoder.pth",
    verbose=True,
)
model.eval()
best_val_loss = training_metrics.best_val_loss

print("\nBest validation loss:", best_val_loss)


# --------------------------------------------------
# 5. Calculate reconstruction errors
# --------------------------------------------------

# Normal validation errors
val_errors = get_reconstruction_errors(X_val, model)


# --------------------------------------------------
# 6. Select anomaly threshold
# --------------------------------------------------

threshold = float(
    np.percentile(val_errors, THRESHOLD_PERCENTILE)
)

print("\nAnomaly threshold:", threshold)
print(
    f"Threshold based on the "
    f"{THRESHOLD_PERCENTILE}th percentile of normal validation errors."
)


# --------------------------------------------------
# 7. Evaluate on global test dataset
# --------------------------------------------------

test_data = np.load("data/global_test.npz")

X_test = test_data["x"]
y_test = test_data["y"]

test_errors = get_reconstruction_errors(X_test, model)

# Higher reconstruction error means more likely anomalous
y_pred = (test_errors > threshold).astype(np.int8)

print("\nGLOBAL TEST RESULTS")
print("-------------------")

print("Test samples:", len(X_test))
print("Actual anomalous samples:", int(y_test.sum()))
print("Predicted anomalous samples:", int(y_pred.sum()))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Anomalous"],
        zero_division=0
    )
)

print("ROC-AUC:", roc_auc_score(y_test, test_errors))
print("Average Precision:", average_precision_score(y_test, test_errors))


# --------------------------------------------------
# 8. Save threshold and configuration
# --------------------------------------------------

with open("outputs/hospital_a_threshold.json", "w") as f:

    json.dump(
        {
            "threshold": threshold,
            "percentile": THRESHOLD_PERCENTILE,
            "best_validation_loss": best_val_loss
        },
        f,
        indent=4
    )

print("\nModel and threshold saved in outputs/")
