"""Fetch the UCI dataset, verify its checksum, and store it unmodified as parquet."""

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from drift_to_deploy.params import Source


class ChecksumError(RuntimeError):
    pass


def fetch(url: str, timeout: float = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def verify(payload: bytes, expected_sha256: str) -> None:
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ChecksumError(f"Downloaded file has sha256 {actual}, expected {expected_sha256}.")


def read_archive(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        (name,) = [n for n in archive.namelist() if n.endswith(".xls")]
        # Row 0 holds generic X1..X23 labels; the real column names are on row 1.
        return pd.read_excel(io.BytesIO(archive.read(name)), header=1)


def download(source: Source, out: Path, payload: bytes | None = None) -> pd.DataFrame:
    payload = payload if payload is not None else fetch(source.url)
    verify(payload, source.sha256)
    frame = read_archive(payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    return frame
