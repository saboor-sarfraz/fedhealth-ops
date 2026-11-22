"""Shared data loading, model training, and validation helpers."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.autoencoder import ECGAutoencoder


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 0.001


@dataclass
class TrainingMetrics:
    """Metrics collected during one local training run."""

    best_val_loss: float
    final_train_loss: float
    epochs_trained: int


def load_hospital_data(hospital_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load only the normal training and validation samples for one hospital."""
    hospital_id = hospital_id.upper()
    if hospital_id not in {"A", "B", "C"}:
        raise ValueError("hospital_id must be one of 'A', 'B', or 'C'")

    dataset_path = ROOT_DIR / "data" / f"hospital_{hospital_id}.npz"
    with np.load(dataset_path, allow_pickle=False) as dataset:
        required_keys = {"x_train", "x_val"}
        missing_keys = required_keys.difference(dataset.files)
        if missing_keys:
            raise ValueError(
                f"{dataset_path} is missing required arrays: "
                f"{', '.join(sorted(missing_keys))}"
            )
        x_train = dataset["x_train"]
        x_val = dataset["x_val"]

    for name, samples in (("x_train", x_train), ("x_val", x_val)):
        if samples.ndim != 2 or samples.shape[1] != 200:
            raise ValueError(
                f"{dataset_path} array '{name}' must have shape (N, 200); "
                f"received {samples.shape}"
            )
        if len(samples) == 0:
            raise ValueError(f"{dataset_path} array '{name}' is empty")

    return x_train, x_val


def evaluate_validation_loss(
    model: nn.Module,
    x_val: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device | None = None,
) -> float:
    """Calculate mean MSE on a hospital's normal validation samples."""
    if device is None:
        device = next(model.parameters()).device

    val_tensor = torch.as_tensor(x_val, dtype=torch.float32).unsqueeze(1)
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=batch_size,
        shuffle=False,
    )
    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for (batch,) in val_loader:
            batch = batch.to(device)
            loss = criterion(model(batch), batch)
            total_loss += loss.item() * batch.size(0)

    return total_loss / len(val_loader.dataset)


def train_local_model(
    model: ECGAutoencoder,
    x_train: np.ndarray,
    x_val: np.ndarray,
    *,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    device: torch.device | None = None,
    best_model_path: str | Path | None = None,
    verbose: bool = False,
) -> tuple[ECGAutoencoder, TrainingMetrics]:
    """Train a model locally and restore the weights with lowest validation MSE."""
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tensor = torch.as_tensor(x_train, dtype=torch.float32).unsqueeze(1)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    final_train_loss = float("inf")

    checkpoint_path = Path(best_model_path) if best_model_path is not None else None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch.size(0)

        final_train_loss = total_train_loss / len(train_loader.dataset)
        val_loss = evaluate_validation_loss(model, x_val, batch_size, device)

        if verbose:
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {final_train_loss:.6f} "
                f"Validation Loss: {val_loss:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)

    if best_state is None:
        raise RuntimeError("Training completed without producing model weights")

    model.load_state_dict(best_state)
    metrics = TrainingMetrics(
        best_val_loss=best_val_loss,
        final_train_loss=final_train_loss,
        epochs_trained=epochs,
    )
    return model, metrics


def get_reconstruction_errors(
    data: np.ndarray,
    model: nn.Module,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return one mean-squared reconstruction error per ECG sample."""
    if device is None:
        device = next(model.parameters()).device

    tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=False,
    )
    errors: list[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            batch_errors = torch.mean(
                (reconstructed - batch) ** 2,
                dim=(1, 2),
            )
            errors.append(batch_errors.cpu().numpy())

    return np.concatenate(errors)
