import asyncio
import hashlib
from pathlib import Path

import pytest

from uqgrid.service import (
    CapabilitySigner,
    CreateCaseUploadRequest,
    DownloadManager,
    LocalArtifactStore,
    LocalCaseRepository,
    TransferAuthorizationError,
    UploadConflictError,
    UploadManager,
    UploadValidationError,
)
from uqgrid.service.schemas import ArtifactKind


SECRET = b"test-secret-that-is-at-least-32-bytes-long"


async def chunks(data):
    yield data[:3]
    yield data[3:]


def test_capability_signer_rejects_tampering_and_expiry():
    signer = CapabilitySigner(SECRET)
    token = signer.issue({"op": "download", "artifact_id": "artifact_1"}, 60, now=100)

    assert signer.verify(token, "download", now=101)["artifact_id"] == "artifact_1"
    with pytest.raises(TransferAuthorizationError):
        signer.verify(token[:-1] + ("A" if token[-1] != "A" else "B"), "download", now=101)
    with pytest.raises(TransferAuthorizationError, match="expired"):
        signer.verify(token, "download", now=161)


def test_upload_stream_verifies_hash_and_completes_case(tmp_path):
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    cases = LocalCaseRepository(tmp_path / "cases", artifacts)
    manager = UploadManager(
        tmp_path / "uploads", cases, CapabilitySigner(SECRET), "http://localhost:8000"
    )
    data = b"function mpc = case1\nmpc.baseMVA = 100;\n"
    request = CreateCaseUploadRequest(
        name="case1",
        files=[{
            "name": "case1.m",
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }],
    )
    session = manager.create("alice", request)
    target = session.targets[0]
    token = str(target.url).split("token=", 1)[1]
    slot_id = str(target.url).split("/")[-1].split("?", 1)[0]

    asyncio.run(manager.receive(session.upload_id, slot_id, token, chunks(data)))
    case = manager.complete("alice", session.upload_id)

    assert case.name == "case1"
    assert case.files[0].name == "case1.m"
    assert cases.resolve_files("alice", case.case_id)[0].read_bytes() == data
    with pytest.raises(UploadConflictError):
        manager.complete("alice", session.upload_id)


def test_upload_rejects_checksum_mismatch(tmp_path):
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    cases = LocalCaseRepository(tmp_path / "cases", artifacts)
    manager = UploadManager(
        tmp_path / "uploads", cases, CapabilitySigner(SECRET), "http://localhost:8000"
    )
    session = manager.create(
        "alice",
        CreateCaseUploadRequest(
            name="case1",
            files=[{"name": "case1.m", "size_bytes": 3, "sha256": "0" * 64}],
        ),
    )
    target = session.targets[0]
    token = str(target.url).split("token=", 1)[1]
    slot_id = str(target.url).split("/")[-1].split("?", 1)[0]

    with pytest.raises(UploadValidationError, match="SHA-256"):
        asyncio.run(manager.receive(session.upload_id, slot_id, token, chunks(b"abc")))


def test_download_capability_is_artifact_specific(tmp_path):
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    artifact = artifacts.put_bytes(
        "alice", ArtifactKind.RESULTS, "application/x-npz", b"result", "results.npz"
    )
    manager = DownloadManager(artifacts, CapabilitySigner(SECRET), "http://localhost:8000")
    target = manager.create("alice", artifact.artifact_id)
    token = str(target.url).split("token=", 1)[1]

    owner, authorized, path = manager.authorize(artifact.artifact_id, token)
    assert owner == "alice"
    assert authorized.sha256 == artifact.sha256
    assert path.read_bytes() == b"result"
    with pytest.raises(TransferAuthorizationError):
        manager.authorize("artifact_other", token)
