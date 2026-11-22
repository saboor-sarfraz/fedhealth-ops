# ECG Inference API Guide

The API serves the existing federated global autoencoder on CPU. It loads the
checkpoint from `outputs/federated_global_autoencoder.pth` and reads the
federated model's threshold from the `Federated global` row in
`outputs/evaluation/model_comparison_metrics.csv`. It does not retrain the
model.

The prediction input should follow the same 200-sample preprocessing used for
training. The API converts samples to `float32` and adds the model's channel
dimension; it does not rescale or normalize the beat.

## Start locally

From the project root, install the requirements in the existing Conda
environment and start Uvicorn:

```powershell
.\venv\python.exe -m pip install -r requirements.txt
.\venv\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Check model and threshold loading at `http://127.0.0.1:8000/health`. The
interactive Swagger UI is at `http://127.0.0.1:8000/docs`.

## Start with Docker Compose

Build the shared image and start only the API service:

```powershell
docker compose up --build -d api
```

The service is available at `http://localhost:8000`. Compose mounts the
federated checkpoint and evaluation metrics read-only. View logs and stop the
API service with:

```powershell
docker compose logs -f api
docker compose stop api
```

## Send a prediction request

In Swagger UI, open `POST /predict`, select **Try it out**, and provide an
object with exactly 200 numbers:

```json
{
  "samples": [0.0, 0.0, 0.0]
}
```

The abbreviated array above is illustrative; the actual request must contain
exactly 200 numeric values. In PowerShell, this sends a valid zero-valued test
beat:

```powershell
$samples = @(1..200 | ForEach-Object { 0.0 })
$body = @{ samples = $samples } | ConvertTo-Json -Compress
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $body
```

The response contains a `normal` or `anomalous` prediction, the reconstruction
error, and the federated threshold. Inputs with other than 200 samples return a
validation error (HTTP 422).
