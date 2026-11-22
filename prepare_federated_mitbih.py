"""
Build a federated setup for ECG anomaly detection from a LOCAL MIT-BIH ZIP.

- Clients (hospitals A/B/C): NORMAL beats only, split by RECORD (no patient overlap),
  balanced by number of normal beats. Each client also gets a 10% normal validation set.
- Global test set: de Chazal DS2 records, all beats, label 0 = normal, 1 = anomalous.

Usage:
    pip install wfdb numpy scipy
    python prepare_federated_mitbih.py                      # auto-finds a MIT-BIH zip next to the script
    python prepare_federated_mitbih.py --zip "C:\\path\\to\\file.zip"

Output: ./fed_data/hospital_{A,B,C}.npz  (x_train, x_val, records)
        ./fed_data/global_test.npz       (x, y)
"""
import argparse
import glob
import os
import shutil
import zipfile

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

HERE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(HERE, "mitdb")
OUT_DIR = os.path.join(HERE, "fed_data")
PRE, POST = 100, 100          # samples before/after R-peak at 360 Hz -> 200-sample beats
SEED = 42

# paced records (102, 104, 107, 217) are excluded
DS1 = ["101","106","108","109","112","114","115","116","118","119","122","124",
       "201","203","205","207","208","209","215","220","223","230"]   # train pool
DS2 = ["100","103","105","111","113","117","121","123","200","202","210","212",
       "213","214","219","221","222","228","231","232","233","234"]  # test pool

NORMAL = {"N"}
ANOMALOUS = {"L","R","A","a","J","S","V","F","e","j","E"}


def db_is_complete():
    return all(
        os.path.exists(os.path.join(DB_DIR, f"{r}.{ext}"))
        for r in DS1 + DS2 for ext in ("hea", "dat", "atr")
    )


def find_zip(user_zip):
    if user_zip:
        return user_zip
    candidates = []
    for pattern in ("*.zip",):
        candidates += glob.glob(os.path.join(HERE, pattern))
        candidates += glob.glob(os.path.join(os.getcwd(), pattern))
    candidates = [c for c in candidates
                  if any(k in os.path.basename(c).lower() for k in ("mit", "bih", "arrhythmia"))]
    if not candidates:
        raise SystemExit("No MIT-BIH zip found next to the script. Pass it with --zip <path>.")
    return candidates[0]


def prepare_db(user_zip):
    if db_is_complete():
        print(f"Using existing database in {DB_DIR}")
        return
    if os.path.isdir(DB_DIR):
        shutil.rmtree(DB_DIR)            # remove partial/failed download
    zpath = find_zip(user_zip)
    print(f"Extracting {zpath} ...")
    tmp = os.path.join(HERE, "_mitdb_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(tmp)
    # the zip usually contains a single subfolder; locate the folder holding 100.hea
    src = None
    for root, _, files in os.walk(tmp):
        if "100.hea" in files:
            src = root
            break
    if src is None:
        raise SystemExit("Could not find 100.hea inside the zip - is this the MIT-BIH Arrhythmia Database?")
    shutil.move(src, DB_DIR)
    shutil.rmtree(tmp, ignore_errors=True)
    if not db_is_complete():
        raise SystemExit("Extracted database is missing some records (.hea/.dat/.atr).")
    print(f"Database ready in {DB_DIR}")


def bandpass(sig, fs, lo=0.5, hi=40.0):
    b, a = butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)


def extract_beats(rec_id):
    path = os.path.join(DB_DIR, rec_id)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    sig = bandpass(rec.p_signal[:, 0], rec.fs)    # lead MLII in nearly all records
    X, y = [], []
    for s, sym in zip(ann.sample, ann.symbol):
        if sym not in NORMAL and sym not in ANOMALOUS:
            continue
        if s - PRE < 0 or s + POST > len(sig):
            continue
        beat = sig[s - PRE: s + POST]
        std = beat.std()
        if std < 1e-6:
            continue
        X.append((beat - beat.mean()) / std)      # per-beat z-score
        y.append(0 if sym in NORMAL else 1)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None, help="path to the MIT-BIH zip file")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    prepare_db(args.zip)

    # ---- training pool: normal beats per record
    per_record = {}
    for r in DS1:
        X, y = extract_beats(r)
        per_record[r] = X[y == 0]
        print(f"record {r}: {len(per_record[r])} normal beats")

    # ---- greedy balanced split of records across 3 hospitals
    hospitals = {"A": [], "B": [], "C": []}
    counts = {"A": 0, "B": 0, "C": 0}
    for r in sorted(per_record, key=lambda k: -len(per_record[k])):
        h = min(counts, key=counts.get)
        hospitals[h].append(r)
        counts[h] += len(per_record[r])

    rng = np.random.default_rng(SEED)
    for h, recs in hospitals.items():
        X = np.concatenate([per_record[r] for r in recs])
        rng.shuffle(X)
        n_val = int(0.1 * len(X))
        np.savez_compressed(os.path.join(OUT_DIR, f"hospital_{h}.npz"),
                            x_train=X[n_val:], x_val=X[:n_val], records=np.array(recs))
        print(f"Hospital {h}: records={recs} train={len(X)-n_val} val={n_val}")

    # ---- global test set from unseen patients
    Xs, ys = [], []
    for r in DS2:
        X, y = extract_beats(r)
        Xs.append(X); ys.append(y)
    Xt, yt = np.concatenate(Xs), np.concatenate(ys)
    np.savez_compressed(os.path.join(OUT_DIR, "global_test.npz"), x=Xt, y=yt)
    print(f"Global test: {len(Xt)} beats, {yt.mean():.1%} anomalous")


if __name__ == "__main__":
    main()