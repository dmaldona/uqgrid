"""Inventory DYR records against active elements in a PSS/E RAW case."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
from pathlib import Path
from typing import Iterable, Mapping

from uqgrid.io.parse import return_dyr_device
from uqgrid.io.parse_psse import read_raw


DEFAULT_NATIVE_MODELS = frozenset(
    {
        "CIM5BL", "ESAC1A", "ESDC1A", "ESDC2A", "ESST4B", "EXAC1",
        "EXAC2", "GAST", "GENROU", "GENSAL", "HYGOV", "IEEEG1",
        "IEEEST", "IEEET1", "IEESGO", "SEXS", "TGOV1",
    }
)
DEFAULT_REDIRECTS = {
    "ESAC6A": "SEXS",
    "EXPIC1": "SEXS",
    "GGOV1": "TGOV1",
    "SCRX": "SEXS",
}
ACTIVSG_TARGET_NATIVE_MODELS = frozenset(
    {
        "ESAC1A",
        "ESDC1A",
        "ESDC2A",
        "ESST4B",
        "EXAC1",
        "EXAC2",
        "GAST",
        "GENROU",
        "GENSAL",
        "HYGOV",
        "IEEEG1",
        "IEEEST",
        "IEEET1",
        "SEXS",
        "TGOV1",
    }
)
ACTIVSG_TARGET_REDIRECTS = {
    "ESAC6A": "SEXS",
    "EXPIC1": "SEXS",
    "GGOV1": "TGOV1",
    "SCRX": "SEXS",
}
MACHINE_MODELS = frozenset({"GENROU", "GENSAL"})
LOAD_MODELS = frozenset({"CIM5BL"})
GOVERNOR_MODELS = frozenset({"GAST", "GGOV1", "HYGOV", "IEEEG1", "IEESGO", "TGOV1"})
EXCITER_MODELS = frozenset(
    {"ESAC1A", "ESAC6A", "ESDC1A", "ESDC2A", "ESST4B", "EXAC1", "EXAC2", "EXPIC1", "IEEET1", "SCRX", "SEXS"}
)
STABILIZER_MODELS = frozenset({"IEEEST"})


class MachineRecordPolicy(str, Enum):
    """Policy used for active generators without source machine records."""

    SOURCE_DYR = "source_dyr"
    SYNTHETIC = "synthetic"


class DyrCoverageError(RuntimeError):
    """Raised when strict DYR coverage requirements are not met."""


@dataclass(frozen=True)
class DyrRecordCoverage:
    record_index: int
    source_model: str
    effective_model: str | None
    bus: int
    device_id: str
    active: bool | None
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ModelCoverage:
    total: int = 0
    active: int = 0
    inactive: int = 0
    native: int = 0
    redirected: int = 0
    unsupported: int = 0
    unmatched: int = 0
    duplicate: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class DyrCoverageReport:
    raw_path: str
    dyr_path: str
    machine_policy: MachineRecordPolicy
    records: tuple[DyrRecordCoverage, ...]
    by_source_model: Mapping[str, ModelCoverage]
    active_generators_without_machine: tuple[tuple[int, str], ...]

    @property
    def counts(self) -> dict[str, int]:
        statuses = Counter(record.status for record in self.records)
        return {
            "active": sum(record.active is True for record in self.records),
            "inactive": sum(record.active is False for record in self.records),
            "native": statuses["native"],
            "redirected": statuses["redirected"],
            "unsupported": statuses["unsupported"],
            "unmatched": statuses["unmatched"],
            "duplicate": statuses["duplicate"],
        }

    def require_complete(self) -> None:
        counts = self.counts
        failures = {
            name: counts[name]
            for name in ("unsupported", "unmatched", "duplicate")
            if counts[name]
        }
        if failures:
            detail = ", ".join(f"{name}={count}" for name, count in failures.items())
            raise DyrCoverageError(f"Incomplete DYR coverage: {detail}")

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_path": self.raw_path,
            "dyr_path": self.dyr_path,
            "machine_policy": self.machine_policy.value,
            "counts": self.counts,
            "active_generators_without_machine": [
                {"bus": bus, "id": device_id}
                for bus, device_id in self.active_generators_without_machine
            ],
            "by_source_model": {
                name: coverage.to_dict()
                for name, coverage in sorted(self.by_source_model.items())
            },
            "records": [record.to_dict() for record in self.records],
        }

    def summary_dict(self) -> dict[str, object]:
        """Return aggregate coverage suitable for committed validation artifacts."""
        source_machine_count = sum(
            coverage.active
            for model, coverage in self.by_source_model.items()
            if model in MACHINE_MODELS
        )
        synthetic_machine_count = (
            len(self.active_generators_without_machine)
            if self.machine_policy == MachineRecordPolicy.SYNTHETIC
            else 0
        )
        return {
            "raw_path": self.raw_path,
            "dyr_path": self.dyr_path,
            "raw_sha256": _sha256(Path(self.raw_path)),
            "dyr_sha256": _sha256(Path(self.dyr_path)),
            "machine_policy": self.machine_policy.value,
            "counts": self.counts,
            "active_generators_without_machine": len(
                self.active_generators_without_machine
            ),
            "source_machine_count": source_machine_count,
            "synthetic_machine_count": synthetic_machine_count,
            "effective_machine_count": source_machine_count + synthetic_machine_count,
            "by_source_model": {
                name: coverage.to_dict()
                for name, coverage in sorted(self.by_source_model.items())
            },
        }


def _clean(value: object) -> str:
    return str(value).replace("'", "").replace('"', "").strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_dyr_records(path: str | Path) -> list[list[str]]:
    """Read whitespace- or comma-delimited, potentially multiline DYR records."""
    lines = Path(path).read_text(errors="ignore").splitlines(keepends=True)
    records = []
    pointer = 0
    while pointer < len(lines):
        line = lines[pointer]
        record = line.strip("\n").split(",") if "," in line else line.split()
        if not record or not record[0].strip() or record[0].lstrip().startswith("//"):
            pointer += 1
            continue
        pointer, record = return_dyr_device(lines, record, pointer)
        records.append(record)
    return records


def _model_coverage(records: Iterable[DyrRecordCoverage]) -> dict[str, ModelCoverage]:
    counters: dict[str, Counter[str]] = {}
    for record in records:
        counter = counters.setdefault(record.source_model, Counter())
        counter["total"] += 1
        if record.active is True:
            counter["active"] += 1
        elif record.active is False:
            counter["inactive"] += 1
        if record.status != "inactive":
            counter[record.status] += 1
    return {
        model: ModelCoverage(**{field: counter[field] for field in ModelCoverage.__dataclass_fields__})
        for model, counter in sorted(counters.items())
    }


def analyze_dyr_coverage(
    raw_path: str | Path,
    dyr_path: str | Path,
    *,
    native_models: Iterable[str] = DEFAULT_NATIVE_MODELS,
    redirects: Mapping[str, str] | None = None,
    machine_policy: MachineRecordPolicy = MachineRecordPolicy.SOURCE_DYR,
    strict: bool = False,
) -> DyrCoverageReport:
    """Classify DYR records using exact RAW bus/device identity and status."""
    raw_path = Path(raw_path)
    dyr_path = Path(dyr_path)
    native = {name.upper() for name in native_models}
    redirects = {
        source.upper(): effective.upper()
        for source, effective in (DEFAULT_REDIRECTS if redirects is None else redirects).items()
    }

    raw = read_raw(str(raw_path))
    active_generator_keys = {
        (int(gen.busn), _clean(gen.name)) for gen in raw.gens if int(gen.status) != 0
    }
    inactive_generator_keys = {
        (int(gen.busn), _clean(gen.name)) for gen in raw.gens if int(gen.status) == 0
    }
    active_load_keys = {
        (int(load.busn), _clean(load.name)) for load in raw.loads if int(load.status) != 0
    }
    inactive_load_keys = {
        (int(load.busn), _clean(load.name)) for load in raw.loads if int(load.status) == 0
    }
    seen: set[tuple[str, int, str]] = set()
    covered_machine_keys: set[tuple[int, str]] = set()
    classified = []
    for index, values in enumerate(read_dyr_records(dyr_path)):
        if len(values) < 3:
            raise DyrCoverageError(f"Malformed DYR record {index}: expected bus, model, and ID")
        source_model = _clean(values[1]).upper()
        try:
            bus = int(float(_clean(values[0])))
        except ValueError as exc:
            raise DyrCoverageError(f"Malformed DYR record {index}: invalid bus {values[0]!r}") from exc
        device_id = _clean(values[2])
        key = (bus, device_id)
        if source_model in MACHINE_MODELS:
            family = "machine"
        elif source_model in GOVERNOR_MODELS:
            family = "governor"
        elif source_model in EXCITER_MODELS:
            family = "exciter"
        elif source_model in STABILIZER_MODELS:
            family = "stabilizer"
        elif source_model in LOAD_MODELS:
            family = "load"
        else:
            family = source_model
        identity = (family, bus, device_id)
        effective_model = None
        if source_model in LOAD_MODELS:
            active_keys = active_load_keys
            inactive_keys = inactive_load_keys
        elif source_model in native or source_model in redirects or source_model in MACHINE_MODELS:
            active_keys = active_generator_keys
            inactive_keys = inactive_generator_keys
        else:
            active_keys = active_generator_keys | active_load_keys
            inactive_keys = inactive_generator_keys | inactive_load_keys

        is_active = key in active_keys
        if identity in seen:
            status = "duplicate"
        elif key in active_keys:
            if source_model in native:
                status = "native"
                effective_model = source_model
            elif source_model in redirects:
                status = "redirected"
                effective_model = redirects[source_model]
            else:
                status = "unsupported"
        elif key in inactive_keys:
            status = "inactive"
        else:
            status = "unmatched"

        seen.add(identity)
        if source_model in MACHINE_MODELS and key in active_generator_keys:
            covered_machine_keys.add(key)
        classified.append(
            DyrRecordCoverage(
                record_index=index,
                source_model=source_model,
                effective_model=effective_model,
                bus=bus,
                device_id=device_id,
                active=is_active if key in active_keys | inactive_keys else None,
                status=status,
            )
        )

    report = DyrCoverageReport(
        raw_path=str(raw_path),
        dyr_path=str(dyr_path),
        machine_policy=MachineRecordPolicy(machine_policy),
        records=tuple(classified),
        by_source_model=_model_coverage(classified),
        active_generators_without_machine=tuple(
            sorted(active_generator_keys - covered_machine_keys)
        ),
    )
    if strict:
        report.require_complete()
    return report
