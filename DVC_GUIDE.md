# DVC Guide

DVC tracks the large ECG datasets and model checkpoints without putting their
binary contents directly in Git. Git stores the small `.dvc` descriptor files
and ignore rules; DVC stores the file contents in its cache and in the DagsHub
remote at <https://dagshub.com/MHassanAbaid24/fedhealth-ops>.

First-time DagsHub setup (repository, collaborators, tokens) is in
`DAGSHUB_SETUP.md`.

The tracked files are:

- `data/hospital_A.npz`, `data/hospital_B.npz`, and `data/hospital_C.npz`
- `data/global_test.npz`
- `outputs/hospital_a_autoencoder.pth`
- `outputs/federated_global_autoencoder.pth`
- `outputs/hospital_a_isolation_forest.joblib`

Run DVC from the project's Conda environment. In PowerShell, its executable is
`.\venv\Scripts\dvc.exe`.

Install the requirements into that environment with:

```powershell
.\venv\python.exe -m pip install -r requirements.txt
```

## Add and version updated data or models

After updating a tracked file, record its new checksum in DVC:

```powershell
.\venv\Scripts\dvc.exe add data/hospital_A.npz
.\venv\Scripts\dvc.exe add outputs/federated_global_autoencoder.pth
```

Stage and commit the corresponding `.dvc` descriptor(s) and generated
`.gitignore` changes in Git. Do not stage the large `.npz` or `.pth` binaries.
Each Git commit then identifies the corresponding DVC versions.

## Remote credentials

The `origin` remote uses DagsHub's S3-compatible storage. DVC reads the
credentials from environment variables, so no secret is stored in the
repository. Create a token under DagsHub **User Settings > Tokens** and set it
as both variables:

```powershell
$env:AWS_ACCESS_KEY_ID = "<dagshub-token>"
$env:AWS_SECRET_ACCESS_KEY = "<dagshub-token>"
```

Pulling a public repository needs a token from any DagsHub account; pushing
needs a token from an account with write access to the DagsHub repository.

## Push and pull

Upload new DVC objects to DagsHub with:

```powershell
.\venv\Scripts\dvc.exe push
```

On a fresh clone, or a checkout where the DVC objects are not present locally,
retrieve them with:

```powershell
.\venv\Scripts\dvc.exe pull
```

## Restore a previous version

Check out the Git revision containing the desired `.dvc` descriptors, then
restore the corresponding data files:

```powershell
git checkout <commit> -- data/hospital_A.npz.dvc outputs/federated_global_autoencoder.pth.dvc
.\venv\Scripts\dvc.exe checkout
```

For data not present in the local DVC cache, run
`.\venv\Scripts\dvc.exe pull` after checking out the desired revision.
