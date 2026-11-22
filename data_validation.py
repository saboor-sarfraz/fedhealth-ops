"""Schema and quality checks for the federated ECG datasets.

Run `python data_validation.py` before training to catch corrupted, missing, or
leaking inputs. The same validators are used by the training data loader.
"""

import sys
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
HOSPITAL_IDS = ("A", "B", "C")
SAMPLE_WIDTH = 200
MIN_ROW_STD = 1e-6


class DataValidationError(ValueError):
    """Raised when a dataset file is missing, malformed, or corrupted."""


def _load_arrays(path: Path, required_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Load the required arrays from an npz file, failing with a clear message."""
    if not path.is_file():
        raise DataValidationError(f"{path} not found")

    try:
        with np.load(path, allow_pickle=False) as dataset:
            missing_keys = set(required_keys).difference(dataset.files)
            if missing_keys:
                raise DataValidationError(
                    f"{path} is missing required arrays: "
                    f"{', '.join(sorted(missing_keys))}"
                )
            return {key: dataset[key] for key in required_keys}
    except (OSError, ValueError, BadZipFile) as error:
        if isinstance(error, DataValidationError):
            raise
        raise DataValidationError(f"{path} could not be read: {error}") from error


def _validate_samples(path: Path, name: str, samples: np.ndarray) -> None:
    """Check one ECG sample matrix for shape, dtype, finiteness, and flat rows."""
    if samples.ndim != 2 or samples.shape[1] != SAMPLE_WIDTH:
        raise DataValidationError(
            f"{path} array '{name}' must have shape (N, {SAMPLE_WIDTH}); "
            f"received {samples.shape}"
        )
    if len(samples) == 0:
        raise DataValidationError(f"{path} array '{name}' is empty")
    if not np.issubdtype(samples.dtype, np.floating):
        raise DataValidationError(
            f"{path} array '{name}' must have a floating dtype; "
            f"received {samples.dtype}"
        )

    non_finite = ~np.isfinite(samples)
    if non_finite.any():
        raise DataValidationError(
            f"{path} array '{name}' contains {int(non_finite.sum())} "
            "NaN/Inf values"
        )

    constant_rows = int((samples.std(axis=1) < MIN_ROW_STD).sum())
    if constant_rows:
        raise DataValidationError(
            f"{path} array '{name}' contains {constant_rows} constant rows"
        )


def validate_hospital_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Validate one hospital npz file and return its `(x_train, x_val)` arrays."""
    arrays = _load_arrays(path, ("x_train", "x_val"))
    for name, samples in arrays.items():
        _validate_samples(path, name, samples)
    return arrays["x_train"], arrays["x_val"]


def validate_test_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Validate the global test npz file and return `(x, y)` with int8 labels."""
    arrays = _load_arrays(path, ("x", "y"))
    x_test, y_test = arrays["x"], arrays["y"]

    _validate_samples(path, "x", x_test)
    if y_test.ndim != 1 or len(y_test) != len(x_test):
        raise DataValidationError(
            f"{path} array 'y' must be one-dimensional with one label per "
            f"sample; received y shape {y_test.shape} for {len(x_test)} samples"
        )
    if not np.isin(y_test, [0, 1]).all():
        raise DataValidationError(
            f"{path} array 'y' must contain only 0 (normal) and 1 (anomaly)"
        )
    if np.unique(y_test).size != 2:
        raise DataValidationError(
            f"{path} array 'y' must contain both classes 0 and 1"
        )

    return x_test, y_test.astype(np.int8, copy=False)


def check_record_overlap(paths: dict[str, Path]) -> None:
    """Fail if any ECG record id appears in more than one hospital file."""
    owners: dict[str, str] = {}
    for hospital_id, path in paths.items():
        if not path.is_file():
            continue
        try:
            with np.load(path, allow_pickle=False) as dataset:
                if "records" not in dataset.files:
                    continue
                records = dataset["records"].ravel().tolist()
        except (OSError, ValueError, BadZipFile) as error:
            raise DataValidationError(
                f"{path} could not be read: {error}"
            ) from error

        for record in records:
            record_id = str(record)
            previous_owner = owners.setdefault(record_id, hospital_id)
            if previous_owner != hospital_id:
                raise DataValidationError(
                    f"record {record_id} appears in both Hospital "
                    f"{previous_owner} and Hospital {hospital_id} "
                    "(patient leakage)"
                )


def main() -> int:
    """Validate every dataset file and return a process exit code."""
    failed = False
    hospital_paths = {
        hospital_id: DATA_DIR / f"hospital_{hospital_id}.npz"
        for hospital_id in HOSPITAL_IDS
    }

    for hospital_id, path in hospital_paths.items():
        label = f"hospital_{hospital_id}.npz"
        try:
            x_train, x_val = validate_hospital_file(path)
        except DataValidationError as error:
            failed = True
            print(f"FAIL {label}: {error}")
        else:
            print(f"PASS {label}: x_train={x_train.shape} x_val={x_val.shape}")

    test_path = DATA_DIR / "global_test.npz"
    try:
        x_test, y_test = validate_test_file(test_path)
    except DataValidationError as error:
        failed = True
        print(f"FAIL global_test.npz: {error}")
    else:
        print(
            f"PASS global_test.npz: x={x_test.shape} "
            f"anomalies={int(y_test.sum())}"
        )

    try:
        check_record_overlap(hospital_paths)
    except DataValidationError as error:
        failed = True
        print(f"FAIL hospital record overlap: {error}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
