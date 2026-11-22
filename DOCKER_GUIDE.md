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

## Reproduce the full pipeline

One command pulls the DVC data and model checkpoints, validates the data, fits
the Isolation Forest baseline, evaluates all three models, and logs and
registers the results in MLflow. The steps are in `reproduce.sh`. By default
it evaluates the DVC-tracked autoencoder checkpoints, so the metrics match the
registered ones exactly (about 1 minute). Set `RETRAIN=1` to also retrain the
local-only and federated autoencoders (seeded, about 50 minutes on CPU).

Credentials: put your DagsHub token in an untracked `.env` file in the project
root (`reproduce.sh` loads it), or export it as both variables (see
`DAGSHUB_SETUP.md`):

```bash
export AWS_ACCESS_KEY_ID="<your-dagshub-token>"
export AWS_SECRET_ACCESS_KEY="<your-dagshub-token>"
docker compose --profile pipeline run --rm pipeline
```

Without Docker (Python environment from `requirements.txt` active):

```bash
bash reproduce.sh
```

If `data/` and the checkpoints are already present, skip the download with
`SKIP_DVC_PULL=1`. For Docker, export `SKIP_DVC_PULL` or `RETRAIN` on the host
before `docker compose ... run`. Without credentials and without
`SKIP_DVC_PULL=1`, the script stops at `dvc pull` with a pointer to
`DAGSHUB_SETUP.md`.

The `pipeline` service bind-mounts the whole project directory to `/app`, so
the results land on the host: `data/`, `outputs/` (models and
`outputs/evaluation/`), `mlflow.db` and `mlruns/`. The service is in the
`pipeline` profile, so plain `docker compose up` does not start it. On Linux,
files written by the container are owned by root.

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
