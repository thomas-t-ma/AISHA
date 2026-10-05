from __future__ import annotations

import hashlib

import pytest

from scripts.setup_local_vision import (
    download_model,
    verify_sha256,
)


def test_verify_sha256_accepts_expected_digest(tmp_path):
    path = tmp_path / "model.task"
    payload = b"deterministic-model-bytes"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    verify_sha256(path, expected)


def test_verify_sha256_rejects_mismatch(tmp_path):
    path = tmp_path / "model.task"
    path.write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_sha256(path, "0" * 64)


def test_existing_model_is_verified_before_reuse(tmp_path):
    path = tmp_path / "model.task"
    path.write_bytes(b"corrupt-cache")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download_model(
            path,
            url="https://invalid.example/model.task",
            expected_sha256="0" * 64,
        )
