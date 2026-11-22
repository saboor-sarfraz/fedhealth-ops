# DVC Guide

DVC tracks the large ECG datasets and model checkpoints without putting their
binary contents directly in Git. Git stores the small `.dvc` descriptor files
and ignore rules; DVC stores the file contents in its cache and in the local
remote at `dvc_storage/`.

The tracked files are:

- `data/hospital_A.npz`, `data/hospital_B.npz`, and `data/hospital_C.npz`
- `data/global_test.npz`
- `outputs/hospital_a_autoencoder.pth`
- `outputs/federated_global_autoencoder.pth`

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

## Push and pull

The default DVC remote is the local `dvc_storage/` directory. Save new DVC
objects there with:

```powershell
.\venv\Scripts\dvc.exe push
```

On a checkout where the DVC objects are not present locally, retrieve them with:

```powershell
.\venv\Scripts\dvc.exe pull
```

The remote is local to this project folder; it does not upload data to a cloud
service.

## Restore a previous version

Check out the Git revision containing the desired `.dvc` descriptors, then
restore the corresponding data files:

```powershell
git checkout <commit> -- data/hospital_A.npz.dvc outputs/federated_global_autoencoder.pth.dvc
.\venv\Scripts\dvc.exe checkout
```

For data not present in the local DVC cache, run
`.\venv\Scripts\dvc.exe pull` after checking out the desired revision.
