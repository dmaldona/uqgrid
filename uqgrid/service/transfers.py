"""Signed artifact upload and download capabilities."""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterable, Dict
from urllib.parse import urlencode
from uuid import uuid4

from .artifacts import LocalArtifactStore
from .cases import LocalCaseRepository
from .schemas import (
    CreateCaseUploadRequest,
    DownloadTarget,
    UploadSession,
    UploadTarget,
)


class TransferAuthorizationError(ValueError):
    pass


class UploadNotFoundError(KeyError):
    pass


class UploadConflictError(ValueError):
    pass


class UploadValidationError(ValueError):
    pass


class CapabilitySigner:
    """Issue short-lived HMAC capabilities for one artifact operation."""

    def __init__(self, secret: bytes, max_ttl_seconds: int = 1800):
        if len(secret) < 32:
            raise ValueError("artifact signing secret must be at least 32 bytes")
        self.secret = secret
        self.max_ttl_seconds = max_ttl_seconds

    def issue(self, claims: dict, ttl_seconds: int, now=None) -> str:
        if ttl_seconds <= 0 or ttl_seconds > self.max_ttl_seconds:
            raise ValueError("capability TTL is outside the configured range")
        issued = int(time.time() if now is None else now)
        payload = {
            "v": 1,
            "aud": "uqgrid-artifact",
            "iat": issued,
            "exp": issued + ttl_seconds,
            "nonce": secrets.token_urlsafe(12),
            **claims,
        }
        encoded = self._encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        signature = self._encode(hmac.new(self.secret, encoded.encode(), hashlib.sha256).digest())
        return f"{encoded}.{signature}"

    def verify(self, token: str, operation: str, now=None) -> dict:
        try:
            encoded, supplied = token.split(".", 1)
            expected = self._encode(
                hmac.new(self.secret, encoded.encode(), hashlib.sha256).digest()
            )
            if not hmac.compare_digest(supplied, expected):
                raise TransferAuthorizationError("invalid capability")
            payload = json.loads(self._decode(encoded))
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TransferAuthorizationError("invalid capability") from exc
        current = int(time.time() if now is None else now)
        if payload.get("v") != 1 or payload.get("aud") != "uqgrid-artifact":
            raise TransferAuthorizationError("invalid capability")
        if payload.get("op") != operation:
            raise TransferAuthorizationError("invalid capability operation")
        if payload.get("exp", 0) <= current or payload.get("iat", current) > current + 60:
            raise TransferAuthorizationError("capability expired")
        if payload["exp"] - payload.get("iat", payload["exp"]) > self.max_ttl_seconds:
            raise TransferAuthorizationError("capability TTL is too long")
        return payload

    @staticmethod
    def _encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    @staticmethod
    def _decode(value: str) -> bytes:
        padding = "=" * (-len(value) % 4)
        return base64.b64decode(value + padding, altchars=b"-_", validate=True)


class UploadManager:
    """Persist upload declarations and verify streamed file contents."""

    def __init__(
        self,
        root,
        cases: LocalCaseRepository,
        signer: CapabilitySigner,
        public_base_url: str,
        max_file_bytes: int = 512 * 1024 * 1024,
    ):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.cases = cases
        self.signer = signer
        self.public_base_url = public_base_url.rstrip("/")
        self.max_file_bytes = max_file_bytes

    def create(self, owner_id: str, request: CreateCaseUploadRequest) -> UploadSession:
        if any(item.size_bytes > self.max_file_bytes for item in request.files):
            raise UploadValidationError("declared file exceeds maximum upload size")
        upload_id = f"upload_{uuid4().hex}"
        now = int(time.time())
        expires_at = datetime.fromtimestamp(now + 1800, tz=timezone.utc)
        slots = []
        targets = []
        directory = self.root / upload_id
        directory.mkdir()
        for item in request.files:
            slot_id = f"slot_{uuid4().hex}"
            claims = {
                "op": "upload",
                "owner_id": owner_id,
                "upload_id": upload_id,
                "slot_id": slot_id,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            token = self.signer.issue(claims, ttl_seconds=600, now=now)
            url = (
                f"{self.public_base_url}/api/v1/artifact-uploads/{upload_id}/{slot_id}?"
                + urlencode({"token": token})
            )
            slots.append({
                "slot_id": slot_id,
                "name": item.name,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
                "status": "pending",
            })
            targets.append(
                UploadTarget(name=item.name, url=url, expires_at=datetime.fromtimestamp(now + 600, tz=timezone.utc))
            )
        record = {
            "upload_id": upload_id,
            "owner_id": owner_id,
            "name": request.name,
            "status": "pending",
            "expires_at": expires_at.isoformat(),
            "slots": slots,
        }
        self._record_path(upload_id).write_text(json.dumps(record, indent=2), encoding="utf-8")
        return UploadSession(upload_id=upload_id, targets=targets, expires_at=expires_at)

    async def receive(self, upload_id: str, slot_id: str, token: str, stream: AsyncIterable[bytes]):
        claims = self.signer.verify(token, "upload")
        if claims.get("upload_id") != upload_id or claims.get("slot_id") != slot_id:
            raise TransferAuthorizationError("capability does not match upload path")
        record = self._read(claims["owner_id"], upload_id)
        slot = next((item for item in record["slots"] if item["slot_id"] == slot_id), None)
        if slot is None:
            raise UploadNotFoundError(slot_id)
        if slot["status"] != "pending":
            raise UploadConflictError("upload slot is already complete")
        temporary = self.root / upload_id / f".{slot_id}.partial"
        ready_directory = self.root / upload_id / slot_id
        ready_directory.mkdir(exist_ok=True)
        ready = ready_directory / slot["name"]
        digest = hashlib.sha256()
        size = 0
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as target:
                async for chunk in stream:
                    size += len(chunk)
                    if size > slot["size_bytes"] or size > self.max_file_bytes:
                        raise UploadValidationError("uploaded file exceeds declared size")
                    digest.update(chunk)
                    target.write(chunk)
            if size != slot["size_bytes"] or digest.hexdigest() != slot["sha256"]:
                raise UploadValidationError("uploaded file size or SHA-256 does not match")
            os.replace(temporary, ready)
        finally:
            temporary.unlink(missing_ok=True)
        slot["status"] = "ready"
        slot["path"] = str(ready)
        self._write(record)
        return {"upload_id": upload_id, "slot_id": slot_id, "size_bytes": size, "sha256": digest.hexdigest()}

    def complete(self, owner_id: str, upload_id: str):
        record = self._read(owner_id, upload_id)
        if record["status"] == "complete":
            raise UploadConflictError("upload session is already complete")
        if any(slot["status"] != "ready" for slot in record["slots"]):
            raise UploadConflictError("all upload slots must be completed first")
        manifest = self.cases.import_files(
            owner_id,
            record["name"],
            [Path(slot["path"]) for slot in record["slots"]],
        )
        record["status"] = "complete"
        record["case_id"] = manifest.case_id
        self._write(record)
        return manifest

    def _read(self, owner_id: str, upload_id: str):
        try:
            record = json.loads(self._record_path(upload_id).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise UploadNotFoundError(upload_id) from exc
        if record["owner_id"] != owner_id:
            raise UploadNotFoundError(upload_id)
        if datetime.fromisoformat(record["expires_at"]) <= datetime.now(timezone.utc):
            raise UploadConflictError("upload session expired")
        return record

    def _write(self, record):
        self._record_path(record["upload_id"]).write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

    def _record_path(self, upload_id):
        if not upload_id.startswith("upload_") or not upload_id[7:].isalnum():
            raise UploadNotFoundError(upload_id)
        return self.root / f"{upload_id}.json"


class DownloadManager:
    def __init__(self, artifacts: LocalArtifactStore, signer: CapabilitySigner, public_base_url: str):
        self.artifacts = artifacts
        self.signer = signer
        self.public_base_url = public_base_url.rstrip("/")

    def create(self, owner_id: str, artifact_id: str) -> DownloadTarget:
        artifact = self.artifacts.get(owner_id, artifact_id)
        now = int(time.time())
        token = self.signer.issue(
            {"op": "download", "owner_id": owner_id, "artifact_id": artifact_id},
            ttl_seconds=300,
            now=now,
        )
        url = f"{self.public_base_url}/api/v1/artifacts/{artifact_id}?{urlencode({'token': token})}"
        return DownloadTarget(
            artifact_id=artifact_id,
            url=url,
            media_type=artifact.media_type,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
            expires_at=datetime.fromtimestamp(now + 300, tz=timezone.utc),
        )

    def authorize(self, artifact_id: str, token: str):
        claims = self.signer.verify(token, "download")
        if claims.get("artifact_id") != artifact_id:
            raise TransferAuthorizationError("capability does not match artifact path")
        artifact = self.artifacts.get(claims["owner_id"], artifact_id)
        return claims["owner_id"], artifact, self.artifacts.path(claims["owner_id"], artifact_id)
