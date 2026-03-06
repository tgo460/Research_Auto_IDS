import hashlib
from pathlib import Path


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model_hash(path: str, expected_hash: str) -> None:
    actual = sha256_file(path)
    if actual.lower() != expected_hash.lower():
        raise ValueError(
            f"model hash mismatch for {Path(path).name}: expected={expected_hash}, actual={actual}"
        )
