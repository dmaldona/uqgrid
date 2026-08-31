import asyncio
import hashlib
import threading
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


def upload_coordinates(target):
    url = str(target.url)
    return url.split("/")[-1].split("?", 1)[0], url.split("token=", 1)[1]


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


def test_parallel_bundle_uploads_merge_slot_updates(tmp_path):
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    cases = LocalCaseRepository(tmp_path / "cases", artifacts)
    manager = UploadManager(
        tmp_path / "uploads", cases, CapabilitySigner(SECRET), "http://localhost:8000"
    )
    payloads = {
        "case.raw": b"0, 100.0 / PSS/E test case\nQ\n",
        "case.dyr": b"1 'GENCLS' 1 1.0 0.0 /\n",
    }
    session = manager.create(
        "alice",
        CreateCaseUploadRequest(
            name="parallel",
            files=[
                {
                    "name": name,
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
                for name, data in payloads.items()
            ],
        ),
    )
    targets = {target.name: target for target in session.targets}

    async def exercise():
        both_started = asyncio.Event()
        started = 0

        async def synchronized_chunks(data):
            nonlocal started
            started += 1
            if started == 2:
                both_started.set()
            await both_started.wait()
            yield data

        calls = []
        for name, data in payloads.items():
            slot_id, token = upload_coordinates(targets[name])
            calls.append(
                manager.receive(
                    session.upload_id,
                    slot_id,
                    token,
                    synchronized_chunks(data),
                )
            )
        await asyncio.gather(*calls)

    asyncio.run(exercise())
    case = manager.complete("alice", session.upload_id)

    assert {item.name for item in case.files} == set(payloads)
    record = manager._read("alice", session.upload_id)
    assert all(slot["status"] == "ready" for slot in record["slots"])


def test_duplicate_slot_upload_conflicts_while_first_is_active(tmp_path):
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    cases = LocalCaseRepository(tmp_path / "cases", artifacts)
    manager = UploadManager(
        tmp_path / "uploads", cases, CapabilitySigner(SECRET), "http://localhost:8000"
    )
    data = b"function mpc = case1\n"
    session = manager.create(
        "alice",
        CreateCaseUploadRequest(
            name="case1",
            files=[{
                "name": "case1.m",
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }],
        ),
    )
    slot_id, token = upload_coordinates(session.targets[0])

    async def exercise():
        started = asyncio.Event()
        release = asyncio.Event()

        async def held_chunks():
            started.set()
            await release.wait()
            yield data

        first = asyncio.create_task(
            manager.receive(session.upload_id, slot_id, token, held_chunks())
        )
        await started.wait()
        with pytest.raises(UploadConflictError, match="already in progress"):
            await manager.receive(session.upload_id, slot_id, token, chunks(data))
        release.set()
        await first

    asyncio.run(exercise())


def test_duplicate_case_completion_conflicts_while_first_is_active(
    monkeypatch, tmp_path
):
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    cases = LocalCaseRepository(tmp_path / "cases", artifacts)
    manager = UploadManager(
        tmp_path / "uploads", cases, CapabilitySigner(SECRET), "http://localhost:8000"
    )
    data = b"function mpc = case1\n"
    session = manager.create(
        "alice",
        CreateCaseUploadRequest(
            name="case1",
            files=[{
                "name": "case1.m",
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }],
        ),
    )
    slot_id, token = upload_coordinates(session.targets[0])
    asyncio.run(manager.receive(session.upload_id, slot_id, token, chunks(data)))

    original_import = cases.import_files
    started = threading.Event()
    release = threading.Event()
    call_lock = threading.Lock()
    call_count = 0
    results = []
    errors = []

    def blocked_import(*args, **kwargs):
        nonlocal call_count
        with call_lock:
            call_count += 1
            first_call = call_count == 1
        if first_call:
            started.set()
            release.wait()
        return original_import(*args, **kwargs)

    def complete_first():
        try:
            results.append(manager.complete("alice", session.upload_id))
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(cases, "import_files", blocked_import)
    thread = threading.Thread(target=complete_first)
    thread.start()
    try:
        assert started.wait(timeout=2)
        with pytest.raises(UploadConflictError, match="already in progress"):
            manager.complete("alice", session.upload_id)
    finally:
        release.set()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert errors == []
    assert len(results) == 1
