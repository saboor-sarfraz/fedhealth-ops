"""Flower client process for one hospital."""

import argparse
import os

import numpy as np
import torch
from flwr.client import NumPyClient, start_client

from models.autoencoder import ECGAutoencoder
from hospital_clients.training import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    evaluate_validation_loss,
    load_hospital_data,
    train_local_model,
)


class HospitalClient(NumPyClient):
    """A Flower client that reads and trains on one hospital's local data."""

    def __init__(self, hospital_id: str) -> None:
        self.hospital_id = hospital_id.upper()
        self.x_train, self.x_val = load_hospital_data(self.hospital_id)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        model = ECGAutoencoder()
        return [
            value.detach().cpu().numpy()
            for value in model.state_dict().values()
        ]

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict,
    ) -> tuple[list[np.ndarray], int, dict]:
        model = ECGAutoencoder()
        model.load_state_dict(
            {
                name: torch.as_tensor(value)
                for name, value in zip(model.state_dict(), parameters)
            }
        )
        model, metrics = train_local_model(
            model,
            self.x_train,
            self.x_val,
            epochs=int(config.get("local_epochs", DEFAULT_EPOCHS)),
            batch_size=DEFAULT_BATCH_SIZE,
            learning_rate=float(
                config.get("learning_rate", DEFAULT_LEARNING_RATE)
            ),
            device=self.device,
        )

        print(
            f"[Hospital {self.hospital_id}] Trained on "
            f"{len(self.x_train)} samples; local validation loss "
            f"{metrics.best_val_loss:.6f}",
            flush=True,
        )
        updated_parameters = [
            value.detach().cpu().numpy()
            for value in model.state_dict().values()
        ]
        return (
            updated_parameters,
            len(self.x_train),
            {
                "hospital_id": self.hospital_id,
                "val_loss": metrics.best_val_loss,
            },
        )

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict,
    ) -> tuple[float, int, dict]:
        model = ECGAutoencoder()
        model.load_state_dict(
            {
                name: torch.as_tensor(value)
                for name, value in zip(model.state_dict(), parameters)
            }
        )
        model.to(self.device)
        val_loss = evaluate_validation_loss(
            model,
            self.x_val,
            batch_size=DEFAULT_BATCH_SIZE,
            device=self.device,
        )
        return (
            val_loss,
            len(self.x_val),
            {"hospital_id": self.hospital_id, "val_loss": val_loss},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a hospital Flower client")
    hospital_default = os.environ.get("HOSPITAL_ID")
    if hospital_default is not None:
        hospital_default = hospital_default.strip().upper()
        if hospital_default not in {"A", "B", "C"}:
            parser.error("HOSPITAL_ID must be one of 'A', 'B', or 'C'")

    parser.add_argument(
        "--hospital",
        choices=("A", "B", "C"),
        default=hospital_default,
        required=hospital_default is None,
    )
    parser.add_argument(
        "--server-address",
        default=os.environ.get("SERVER_ADDRESS", "127.0.0.1:8080"),
    )
    args = parser.parse_args()

    client = HospitalClient(args.hospital)
    start_client(
        server_address=args.server_address,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
