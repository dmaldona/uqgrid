"""Immutable local artifact storage for the headless service."""

import hashlib
import json
import os
from pathlib import Path
from typing import BinaryIO, Optional
from uuid import uuid4

from .schemas import Artifact, ArtifactKind


class ArtifactNotFoundError(KeyError):
    pass


class LocalArtifactStore:
    """Store immutable artifacts in a server-owned local directory."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(
        self,
        owner_id: str,
        kind: ArtifactKind,
        media_type: str,
        data: bytes,
        filename: str,
    ) -> Artifact:
        artifact_id = f"artifact_{uuid4().hex}"
        directory = self.root / artifact_id
        directory.mkdir()
        suffix = Path(filename).suffix.lower()
        data_path = directory / f"data{suffix}"
        metadata_path = directory / "metadata.json"
        digest = hashlib.sha256(data).hexdigest()
        artifact = Artifact(
            artifact_id=artifact_id,
            kind=kind,
            media_type=media_type,
            size_bytes=len(data),
            sha256=digest,
            resource_uri=f"uqgrid://artifacts/{artifact_id}",
        )
        self._write_exclusive(data_path, data)
        metadata = {
            "owner_id": owner_id,
            "filename": filename,
            "artifact": artifact.model_dump(mode="json"),
        }
        self._write_exclusive(
            metadata_path,
            json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"),
        )
        return artifact

    def put_file(
        self,
        owner_id: str,
        kind: ArtifactKind,
        media_type: str,
        source: Path,
    ) -> Artifact:
        source = Path(source)
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"case input must be a regular file: {source.name}")
        return self.put_bytes(owner_id, kind, media_type, source.read_bytes(), source.name)

    def get(self, owner_id: str, artifact_id: str) -> Artifact:
        metadata = self._read_metadata(owner_id, artifact_id)
        return Artifact.model_validate(metadata["artifact"])

    def filename(self, owner_id: str, artifact_id: str) -> str:
        return str(self._read_metadata(owner_id, artifact_id)["filename"])

    def path(self, owner_id: str, artifact_id: str) -> Path:
        metadata = self._read_metadata(owner_id, artifact_id)
        suffix = Path(metadata["filename"]).suffix.lower()
        return self.root / artifact_id / f"data{suffix}"

    def read_bytes(self, owner_id: str, artifact_id: str) -> bytes:
        return self.path(owner_id, artifact_id).read_bytes()

    def _read_metadata(self, owner_id: str, artifact_id: str):
        if not artifact_id.startswith("artifact_") or not artifact_id[9:].isalnum():
            raise ArtifactNotFoundError(artifact_id)
        path = self.root / artifact_id / "metadata.json"
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ArtifactNotFoundError(artifact_id) from exc
        if metadata["owner_id"] != owner_id:
            raise ArtifactNotFoundError(artifact_id)
        return metadata

    @staticmethod
    def _write_exclusive(path: Path, data: bytes):
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
