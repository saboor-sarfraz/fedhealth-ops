"""Flower FedAvg server process for the three hospital clients."""

import os
from pathlib import Path

import numpy as np
import torch
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters
from flwr.common import parameters_to_ndarrays
from flwr.server import ServerConfig, start_server
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from models.autoencoder import ECGAutoencoder
from hospital_clients.training import (
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    ROOT_DIR,
)


NUM_HOSPITALS = 3
NUM_ROUNDS = 5
HOSPITAL_IDS = {"A", "B", "C"}
GLOBAL_MODEL_PATH = ROOT_DIR / "outputs" / "federated_global_autoencoder.pth"
SERVER_BIND_ADDRESS = os.environ.get(
    "SERVER_BIND_ADDRESS",
    "127.0.0.1:8080",
)


def aggregate_fit_metrics(
    results: list[tuple[int, dict[str, Scalar]]],
) -> dict[str, Scalar]:
    """Report each client's local validation loss and its training-set size."""
    if not results:
        return {}

    total_examples = sum(num_examples for num_examples, _ in results)
    weighted_val_loss = sum(
        num_examples * float(metrics["val_loss"])
        for num_examples, metrics in results
    ) / total_examples
    return {"weighted_val_loss": weighted_val_loss}


def aggregate_evaluate_metrics(
    results: list[tuple[int, dict[str, Scalar]]],
) -> dict[str, Scalar]:
    """Report validation loss on the updated global model for each hospital."""
    if not results:
        return {}

    total_examples = sum(num_examples for num_examples, _ in results)
    weighted_val_loss = sum(
        num_examples * float(metrics["val_loss"])
        for num_examples, metrics in results
    ) / total_examples
    return {"weighted_val_loss": weighted_val_loss}


class ReportingFedAvg(FedAvg):
    """FedAvg strategy that records successful rounds for final checkpointing."""

    final_parameters: list[np.ndarray]
    successful_rounds: int

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[BaseException | tuple[ClientProxy, FitRes]],
    ) -> tuple[Parameters | None, dict[str, Scalar] | None]:
        participating = {
            str(fit_res.metrics.get("hospital_id", "unknown"))
            for _, fit_res in results
        }
        print(
            f"\n[Round {server_round}/{NUM_ROUNDS}] "
            f"Received updates from {len(participating)}/{NUM_HOSPITALS} "
            f"hospitals: {', '.join(sorted(participating)) or 'none'}",
            flush=True,
        )
        for _, fit_res in results:
            hospital_id = str(fit_res.metrics.get("hospital_id", "unknown"))
            val_loss = fit_res.metrics.get("val_loss", float("nan"))
            print(
                f"  Hospital {hospital_id}: "
                f"train_samples={fit_res.num_examples}, "
                f"local_val_loss={float(val_loss):.6f}",
                flush=True,
            )
        if participating != HOSPITAL_IDS:
            raise RuntimeError(
                f"Round {server_round} expected hospitals "
                f"{sorted(HOSPITAL_IDS)}, received {sorted(participating)}"
            )

        aggregated, metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )
        if aggregated is not None:
            self.final_parameters = parameters_to_ndarrays(aggregated)
            self.successful_rounds += 1
        return aggregated, metrics


def main() -> None:
    model = ECGAutoencoder()
    initial_parameters = ndarrays_to_parameters(
        [value.detach().cpu().numpy() for value in model.state_dict().values()]
    )
    strategy = ReportingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_HOSPITALS,
        min_evaluate_clients=NUM_HOSPITALS,
        min_available_clients=NUM_HOSPITALS,
        accept_failures=False,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda server_round: {
            "local_epochs": DEFAULT_EPOCHS,
            "learning_rate": DEFAULT_LEARNING_RATE,
        },
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )
    strategy.final_parameters = []
    strategy.successful_rounds = 0

    print(
        f"Starting Flower FedAvg: {NUM_HOSPITALS} hospitals, "
        f"{NUM_ROUNDS} rounds, {DEFAULT_EPOCHS} local epochs per round.",
        flush=True,
    )
    start_server(
        server_address=SERVER_BIND_ADDRESS,
        config=ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    if strategy.successful_rounds != NUM_ROUNDS:
        raise RuntimeError(
            f"Only {strategy.successful_rounds}/{NUM_ROUNDS} rounds "
            "successfully aggregated; refusing to save a partial model."
        )

    final_model = ECGAutoencoder()
    final_model.load_state_dict(
        {
            name: torch.as_tensor(value)
            for name, value in zip(
                final_model.state_dict(),
                strategy.final_parameters,
            )
        }
    )
    GLOBAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_model.state_dict(), GLOBAL_MODEL_PATH)
    print(f"Final global model saved to: {GLOBAL_MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
