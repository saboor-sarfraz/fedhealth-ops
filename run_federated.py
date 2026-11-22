"""Launch one Flower server and three hospital processes via its gRPC API.

Flower's documented gRPC compatibility API is deprecated but retained in 1.43;
it avoids the Ray simulation runtime, whose Windows support is experimental.
"""

import subprocess
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


def main() -> int:
    processes: list[tuple[str, subprocess.Popen]] = []
    try:
        for hospital_id in ("A", "B", "C"):
            command = [
                sys.executable,
                "-m",
                "hospital_clients.client",
                "--hospital",
                hospital_id,
            ]
            print(f"Starting Hospital {hospital_id} client.", flush=True)
            processes.append(
                (
                    f"Hospital {hospital_id}",
                    subprocess.Popen(command, cwd=ROOT_DIR),
                )
            )

        print("Starting Flower server.", flush=True)
        server = subprocess.Popen(
            [sys.executable, "-m", "hospital_clients.server"],
            cwd=ROOT_DIR,
        )
        processes.append(("Flower server", server))

        while server.poll() is None:
            for name, process in processes[:-1]:
                return_code = process.poll()
                if return_code is not None and return_code != 0:
                    raise RuntimeError(
                        f"{name} exited with status {return_code}"
                    )
            time.sleep(1)

        if server.returncode != 0:
            raise RuntimeError(
                f"Flower server exited with status {server.returncode}"
            )

        for name, process in processes[:-1]:
            return_code = process.wait(timeout=60)
            if return_code != 0:
                raise RuntimeError(f"{name} exited with status {return_code}")

        return 0
    except KeyboardInterrupt:
        print("Federated training interrupted.", flush=True)
        return 130
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
