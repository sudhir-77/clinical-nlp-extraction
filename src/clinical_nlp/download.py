"""Checksum-verified downloads. Clinical files inside BioScope are not saved."""

from __future__ import annotations

import hashlib
import tempfile
import zipfile
from pathlib import Path

import certifi
import requests

NCBI_URLS = {
    "train": (
        "https://raw.githubusercontent.com/spyysalo/ncbi-disease/master/"
        "original-data/train/NCBItrainset_corpus.txt"
    ),
    "dev": (
        "https://raw.githubusercontent.com/spyysalo/ncbi-disease/master/"
        "original-data/devel/NCBIdevelopset_corpus.txt"
    ),
    "test": (
        "https://raw.githubusercontent.com/spyysalo/ncbi-disease/master/"
        "original-data/test/NCBItestset_corpus.txt"
    ),
}

# SHA-256 of the official NCBI Disease mention files (US government work).
NCBI_SHA256 = {
    "train": "f311381b28a6d0f641f57f4a963bb83619c2bdec5cd9caabb1381f894d5db817",
    "dev": "02db559974624ae9cbe6cd95d64d953e6926bb63afb6b25b63e53bd8b41a4b55",
    "test": "61c8050bf6979ef7b943ec1158dfc03b651ea3367d4c37356f9de4841e85c9a3",
}

BIOSCOPE_ZIP_URL = "https://rgai.inf.u-szeged.hu/sites/rgai.inf.u-szeged.hu/files/bioscope.zip"
BIOSCOPE_ZIP_SHA256 = "b20b525506f783ab9c120548ac4022ab587bf7cc419beaf759f7253f0cf0b1d1"
ABSTRACTS_MEMBER = "abstracts.xml"
ABSTRACTS_SHA256 = "464bfd645aa4935ba2c6f6230cccf3884fe3819499d0471b7c405b05b998c5b7"

USER_AGENT = "clinical-nlp-extraction/1.0 (research; public corpora only)"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get(url: str, timeout: int = 180) -> bytes:
    response = requests.get(
        url,
        timeout=timeout,
        verify=certifi.where(),
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return response.content


def ensure_file(url: str, dest: Path, expected_sha256: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256_file(dest) == expected_sha256:
        return dest
    payload = _get(url)
    digest = sha256_bytes(payload)
    if digest != expected_sha256:
        raise RuntimeError(f"Checksum mismatch for {url}: {digest} != {expected_sha256}")
    dest.write_bytes(payload)
    return dest


def ensure_ncbi(data_dir: Path) -> dict[str, Path]:
    folder = data_dir / "ncbi"
    paths = {}
    for split, url in NCBI_URLS.items():
        paths[split] = ensure_file(url, folder / f"{split}.txt", NCBI_SHA256[split])
    return paths


def ensure_bioscope_abstracts(data_dir: Path) -> Path:
    """Save only abstracts.xml. The zip's radiology reports are discarded."""
    dest = data_dir / "bioscope" / "abstracts.xml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256_file(dest) == ABSTRACTS_SHA256:
        return dest
    payload = _get(BIOSCOPE_ZIP_URL)
    digest = sha256_bytes(payload)
    if digest != BIOSCOPE_ZIP_SHA256:
        raise RuntimeError(f"BioScope zip checksum mismatch: {digest}")
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "bioscope.zip"
        archive_path.write_bytes(payload)
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            if ABSTRACTS_MEMBER not in names:
                raise RuntimeError(f"{ABSTRACTS_MEMBER} missing from BioScope zip")
            xml = archive.read(ABSTRACTS_MEMBER)
    if sha256_bytes(xml) != ABSTRACTS_SHA256:
        raise RuntimeError("abstracts.xml checksum mismatch after extract")
    dest.write_bytes(xml)
    return dest
