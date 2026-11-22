# DagsHub Setup Guide

This project stores its DVC-tracked data and model checkpoints on DagsHub. Git
(GitHub) keeps the small `.dvc` pointer files; DagsHub keeps the actual `.npz`
and `.pth` files.

The DVC remote is already configured in `.dvc/config`:

```ini
[core]
    remote = origin
['remote "origin"']
    url = s3://dvc
    endpointurl = https://dagshub.com/MHassanAbaid24/fedhealth-ops.s3
```

This only works after the DagsHub repository
`https://dagshub.com/MHassanAbaid24/fedhealth-ops` exists. Follow the steps in
order.

## Who does what

| Step | Who |
| --- | --- |
| 1. Create the DagsHub repository | Hassan (`MHassanAbaid24`) |
| 2. Add Saboor as a collaborator | Hassan |
| 3. Create an access token | Hassan and Saboor, each their own |
| 4. Set the environment variables | Everyone who runs DVC |
| 5. Push the data | Done (all tracked files are on DagsHub) |
| 6. Verify with a fresh pull | Hassan |

## 1. Create the DagsHub repository (Hassan)

The GitHub repository belongs to `saboor-sarfraz`. A GitHub collaborator
cannot install the DagsHub GitHub app on someone else's account, so the
repository does not appear in the "Connect GitHub" list. Connect it by URL
instead. This works because the GitHub repository is public.

1. Sign in at <https://dagshub.com> as `MHassanAbaid24`.
2. Click **Create** (top right), then **New Repository**.
3. Open the **Connect a repository** tab and choose **Other**.
4. Fill in:
   - **Repository URL**: `https://github.com/saboor-sarfraz/fedhealth-ops.git`
   - **Repository name**: `fedhealth-ops` (exactly this; it must match
     `.dvc/config`)
   - **Authentication**: none (public repository)
5. Click **Connect Repository**.
6. Check that the repository now opens at
   <https://dagshub.com/MHassanAbaid24/fedhealth-ops> and appears next to your
   other repositories.

DagsHub now mirrors the GitHub code. It refreshes the mirror on its own; DVC
data is uploaded separately in step 5.

If the **Other** option fails, create an empty repository instead: **Create >
New Repository**, name it `fedhealth-ops`, and click **Create**. DVC storage
works the same way; only the code view on DagsHub stays empty.

## 2. Add Saboor as a collaborator (Hassan)

Saboor needs write access to upload the data.

1. Open <https://dagshub.com/MHassanAbaid24/fedhealth-ops>.
2. Go to **Settings > Collaborators**.
3. Add Saboor's DagsHub username with the **Write** role.
4. Saboor accepts the invitation from his DagsHub notifications or email.

## 3. Create an access token (each person)

1. Sign in to DagsHub with your own account.
2. Open <https://dagshub.com/user/settings/tokens>.
3. Copy the default token, or generate a new one.

Treat the token like a password. Never commit it, paste it in a `.dvc` file, or
share it in chat.

## 4. Set the environment variables (everyone who runs DVC)

DVC talks to DagsHub through its S3-compatible API. Set your DagsHub token as
**both** variables:

**Linux / macOS (bash):**

```bash
export AWS_ACCESS_KEY_ID="<your-dagshub-token>"
export AWS_SECRET_ACCESS_KEY="<your-dagshub-token>"
```

To keep them across terminals, add the two lines to `~/.bashrc`, then run
`source ~/.bashrc`.

**Windows (PowerShell), current window only:**

```powershell
$env:AWS_ACCESS_KEY_ID = "<your-dagshub-token>"
$env:AWS_SECRET_ACCESS_KEY = "<your-dagshub-token>"
```

**Windows, permanent (open a new terminal afterwards):**

```powershell
setx AWS_ACCESS_KEY_ID "<your-dagshub-token>"
setx AWS_SECRET_ACCESS_KEY "<your-dagshub-token>"
```

Make sure DVC has S3 support installed:

```bash
pip install -r requirements.txt   # includes dvc[s3]
```

Check that DVC can reach DagsHub:

```bash
dvc remote list     # expect: origin  s3://dvc  (default)
dvc status -c       # must NOT print "Unable to locate credentials"
```

## 5. Push the data (done)

All DVC-tracked files are already on DagsHub. Repeat this only after changing
a tracked file (`dvc add <file>`, then commit the updated `.dvc` pointer):

```bash
dvc push
```

The remote holds seven files:

- `data/hospital_A.npz`, `data/hospital_B.npz`, `data/hospital_C.npz`
- `data/global_test.npz`
- `outputs/hospital_a_autoencoder.pth`
- `outputs/federated_global_autoencoder.pth`
- `outputs/hospital_a_isolation_forest.joblib`

On DagsHub, the repository's **Files** view then shows these files with a DVC
badge.

## 6. Verify with a fresh pull (Hassan)

Test on a clean clone, the same way a grader would:

```bash
git clone https://github.com/saboor-sarfraz/fedhealth-ops fedhealth-check
cd fedhealth-check
pip install -r requirements.txt
# set the two environment variables from step 4
dvc pull
dvc status            # expect: "Data and pipelines are up to date."
```

The `data/` and `outputs/` folders should now contain the real files.

## Troubleshooting

| Message | Cause | Fix |
| --- | --- | --- |
| `Unable to locate credentials` | Environment variables not set in this terminal | Repeat step 4, then retry in the same terminal |
| `403 Forbidden` / `AccessDenied` on push | Token's account lacks write access | Do step 2, accept the invite, use your own token |
| `404` / `NoSuchBucket` | DagsHub repository missing or misnamed | Repository must be exactly `MHassanAbaid24/fedhealth-ops` (step 1) |
| `missing cache files` on push | This machine never had the data | Push from Saboor's machine (step 5) |
| `No module named 's3fs'` / S3 error | DVC installed without S3 support | `pip install "dvc[s3]"` |

## Notes for graders

DagsHub requires a token even to pull. Graders can use a token from any free
DagsHub account; set it as shown in step 4, then run `dvc pull`.
