"""Case ingestion, persistence, and inspection."""

import hashlib
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Tuple
from uuid import uuid4

from uqgrid.io.parse import add_dyr, load_matpower, load_psse

from .artifacts import LocalArtifactStore
from .schemas import (
    ArtifactKind,
    CaseFile,
    CaseFormat,
    CaseInspection,
    CaseManifest,
    CaseStatus,
)


class CaseNotFoundError(KeyError):
    pass


class CaseValidationError(ValueError):
    pass


class LocalCaseRepository:
    """Persist immutable case manifests and their input artifacts."""

    def __init__(self, root, artifacts: LocalArtifactStore):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts = artifacts

    def import_files(self, owner_id: str, name: str, files: Iterable[Path]) -> CaseManifest:
        paths = [Path(item) for item in files]
        case_format = self._validate_bundle(paths)
        case_files = []
        for path in paths:
            media_type = {
                ".raw": "text/plain",
                ".dyr": "text/plain",
                ".m": "text/plain",
            }[path.suffix.lower()]
            artifact = self.artifacts.put_file(
                owner_id, ArtifactKind.CASE_INPUT, media_type, path
            )
            case_files.append(
                CaseFile(
                    artifact_id=artifact.artifact_id,
                    name=path.name,
                    size_bytes=artifact.size_bytes,
                    sha256=artifact.sha256,
                    media_type=artifact.media_type,
                )
            )

        digest = hashlib.sha256()
        for item in sorted(case_files, key=lambda value: value.name):
            digest.update(item.name.encode("utf-8"))
            digest.update(bytes.fromhex(item.sha256))
        manifest = CaseManifest(
            case_id=f"case_{uuid4().hex}",
            owner_id=owner_id,
            name=name,
            format=case_format,
            status=CaseStatus.READY,
            files=case_files,
            sha256=digest.hexdigest(),
            created_at=datetime.now(timezone.utc),
        )
        self._manifest_path(manifest.case_id).write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        return manifest

    def get(self, owner_id: str, case_id: str) -> CaseManifest:
        if not case_id.startswith("case_") or not case_id[5:].isalnum():
            raise CaseNotFoundError(case_id)
        try:
            manifest = CaseManifest.model_validate_json(
                self._manifest_path(case_id).read_text(encoding="utf-8")
            )
        except (FileNotFoundError, ValueError) as exc:
            raise CaseNotFoundError(case_id) from exc
        if manifest.owner_id != owner_id:
            raise CaseNotFoundError(case_id)
        return manifest

    def list(self, owner_id: str):
        manifests = []
        for path in sorted(self.root.glob("case_*.json")):
            try:
                manifest = CaseManifest.model_validate_json(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if manifest.owner_id == owner_id:
                manifests.append(manifest)
        return manifests

    def resolve_files(self, owner_id: str, case_id: str) -> Tuple[Path, Optional[Path]]:
        manifest = self.get(owner_id, case_id)
        by_suffix = {
            Path(item.name).suffix.lower(): self.artifacts.path(owner_id, item.artifact_id)
            for item in manifest.files
        }
        primary = by_suffix[".m"] if manifest.format == CaseFormat.MATPOWER else by_suffix[".raw"]
        return primary, by_suffix.get(".dyr")

    def inspect(self, owner_id: str, case_id: str) -> CaseInspection:
        manifest = self.get(owner_id, case_id)
        primary, dynamics = self.resolve_files(owner_id, case_id)
        captured = []
        try:
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                psys = (
                    load_matpower(str(primary))
                    if manifest.format == CaseFormat.MATPOWER
                    else load_psse(str(primary))
                )
                if dynamics is not None:
                    add_dyr(psys, str(dynamics))
                captured = [str(record.message) for record in records]
        except Exception as exc:
            raise CaseValidationError(f"case {case_id} could not be parsed") from exc
        dynamic_devices = psys.gendyn + psys.exc + psys.gov + psys.mot
        model_counts = Counter(device.__class__.__name__ for device in dynamic_devices)
        return CaseInspection(
            case_id=case_id,
            format=manifest.format,
            base_mva=psys.basemva,
            bus_count=psys.nbuses,
            branch_count=psys.nbranches,
            generator_count=psys.ngens,
            load_count=psys.nloads,
            shunt_count=psys.nshunts,
            dynamic_model_count=len(dynamic_devices),
            dynamic_models=dict(sorted(model_counts.items())),
            warnings=captured,
        )

    def _manifest_path(self, case_id: str) -> Path:
        return self.root / f"{case_id}.json"

    @staticmethod
    def _validate_bundle(paths):
        if not paths:
            raise ValueError("case bundle must not be empty")
        names = [path.name for path in paths]
        if len(names) != len(set(names)):
            raise ValueError("case bundle filenames must be unique")
        if any(not path.is_file() or path.is_symlink() for path in paths):
            raise ValueError("case bundle entries must be regular files")
        suffixes = sorted(path.suffix.lower() for path in paths)
        if suffixes == [".m"]:
            return CaseFormat.MATPOWER
        if suffixes in ([".raw"], [".dyr", ".raw"]):
            return CaseFormat.PSSE
        raise ValueError("case bundle must contain one .m file or one .raw and optional .dyr")
