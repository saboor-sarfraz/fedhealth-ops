# Docker Guide

## Architecture

Docker Compose builds one shared image and runs four services on the private
`fedhealth-net` bridge network:

- `flower-server` initializes the model, waits for all three clients, runs
  five FedAvg rounds, and saves the final state dict.
- `hospital-a`, `hospital-b`, and `hospital-c` each load only their own normal
  training and validation data, train locally, and send model parameters and
  training metrics to Flower. Raw ECG samples are not sent to the server.

The server has a TCP healthcheck. Compose waits for it to pass before starting
the clients. The Flower client also keeps retrying its connection if the server
is not ready yet.

## Dataset and output mounts

Each client receives exactly one read-only file:

| Service | Host file | Container file |
| --- | --- | --- |
| `hospital-a` | `data/hospital_A.npz` | `/app/data/hospital_A.npz` |
| `hospital-b` | `data/hospital_B.npz` | `/app/data/hospital_B.npz` |
| `hospital-c` | `data/hospital_C.npz` | `/app/data/hospital_C.npz` |

The datasets are not copied into the image. `global_test.npz` is neither mounted
nor used by federated training. The host `outputs/` directory is mounted at
`/app/outputs` in the server container.

## Build and run

From the project root:

```powershell
docker compose build
docker compose up
```

To run detached and follow the logs separately:

```powershell
docker compose up -d
docker compose logs -f
```

Check service status with:

```powershell
docker compose ps
```

During training, the server logs each round and reports the participating
hospitals and their training sample counts. Every round should show
`3/3 hospitals: A, B, C`. The server writes the final model to
`outputs/federated_global_autoencoder.pth` on the host after all five rounds
successfully aggregate.

## Stop and clean up

Stop and remove the containers and Compose network with:

```powershell
docker compose down
```

This does not delete the host datasets or files in `outputs/`.

## Common issues

- **Docker daemon unavailable:** Start Docker Desktop and wait for the engine
  to become ready, then retry `docker compose build`.
- **Dataset bind mount error:** Run Compose from the project root and check the
  exact case-sensitive names under `data/` (`hospital_A.npz`, etc.). Each file
  must exist on the host.
- **Client cannot connect:** Check `docker compose ps` and
  `docker compose logs flower-server`. The server must become healthy on port
  8080; clients use the Compose DNS name `flower-server`, not `localhost`.
- **No global model on the host:** Check the server logs for five successful
  rounds and verify that the host `outputs/` directory is writable.
- **Slow first build:** PyTorch and scientific Python packages take time to
  download. The image installs the CPU-only PyTorch wheel to avoid CUDA
  libraries.
