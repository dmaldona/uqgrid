"""Persistent result metadata and bounded numerical queries."""

import json
from pathlib import Path
from typing import Dict, Iterable, List
from uuid import uuid4

import numpy as np

from .artifacts import ArtifactNotFoundError, LocalArtifactStore
from .schemas import (
    Artifact,
    ArtifactKind,
    DynamicsResultSummary,
    PowerFlowResultSummary,
    ResultQuery,
    ResultQueryResponse,
    SignalDescriptor,
    SignalValues,
)


class ResultNotFoundError(KeyError):
    pass


class LocalResultRepository:
    """Store result summaries, artifacts, and named signals."""

    def __init__(self, root, artifacts: LocalArtifactStore):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts = artifacts

    def create(
        self,
        owner_id: str,
        summary,
        arrays: Dict[str, np.ndarray],
        signals: Iterable[SignalDescriptor],
        state_metadata: Dict[str, dict],
        manifest: dict,
        log: str = "",
    ):
        result_id = summary.result_id
        artifact_items = []
        artifact_items.append(
            self.artifacts.put_bytes(
                owner_id,
                ArtifactKind.SUMMARY,
                "application/json",
                summary.model_dump_json(indent=2).encode("utf-8"),
                "summary.json",
            )
        )
        artifact_items.append(
            self._put_npz(owner_id, arrays)
        )
        artifact_items.append(
            self.artifacts.put_bytes(
                owner_id,
                ArtifactKind.STATE_METADATA,
                "application/json",
                json.dumps(state_metadata, indent=2, sort_keys=True).encode("utf-8"),
                "state_metadata.json",
            )
        )
        artifact_items.append(
            self.artifacts.put_bytes(
                owner_id,
                ArtifactKind.MANIFEST,
                "application/json",
                json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
                "manifest.json",
            )
        )
        artifact_items.append(
            self.artifacts.put_bytes(
                owner_id,
                ArtifactKind.LOG,
                "text/plain",
                log.encode("utf-8"),
                "run.log",
            )
        )
        summary = summary.model_copy(update={"artifacts": artifact_items})
        record = {
            "owner_id": owner_id,
            "summary_type": type(summary).__name__,
            "summary": summary.model_dump(mode="json"),
            "signals": [signal.model_dump(mode="json") for signal in signals],
            "results_artifact_id": next(
                item.artifact_id for item in artifact_items if item.kind == ArtifactKind.RESULTS
            ),
        }
        self._record_path(result_id).write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary

    def get_summary(self, owner_id: str, result_id: str):
        record = self._read_record(owner_id, result_id)
        model = {
            "PowerFlowResultSummary": PowerFlowResultSummary,
            "DynamicsResultSummary": DynamicsResultSummary,
        }[record["summary_type"]]
        return model.model_validate(record["summary"])

    def list_signals(self, owner_id: str, result_id: str) -> List[SignalDescriptor]:
        record = self._read_record(owner_id, result_id)
        return [SignalDescriptor.model_validate(item) for item in record["signals"]]

    def query(self, owner_id: str, query: ResultQuery) -> ResultQueryResponse:
        record = self._read_record(owner_id, query.result_id)
        available = {item["name"]: item for item in record["signals"]}
        missing = [name for name in query.signals if name not in available]
        if missing:
            raise ValueError(f"unknown signals: {', '.join(missing)}")
        artifact_id = record["results_artifact_id"]
        try:
            with np.load(self.artifacts.path(owner_id, artifact_id), allow_pickle=False) as data:
                time = np.asarray(data["time_s"], dtype=float)
                selected = {name: np.asarray(data[name], dtype=float) for name in query.signals}
        except (ArtifactNotFoundError, KeyError) as exc:
            raise ResultNotFoundError(query.result_id) from exc

        mask = np.ones(time.shape, dtype=bool)
        if query.start_s is not None:
            mask &= time >= query.start_s
        if query.end_s is not None:
            mask &= time <= query.end_s
        time = time[mask]
        selected = {name: values[mask] for name, values in selected.items()}
        if time.size == 0:
            raise ValueError("query time range contains no samples")

        if query.aggregate is not None:
            values = [
                SignalValues(
                    signal=name,
                    unit=available[name]["unit"],
                    values=[self._aggregate(query.aggregate, time, signal)],
                )
                for name, signal in selected.items()
            ]
            return ResultQueryResponse(
                result_id=query.result_id,
                signals=values,
                aggregate=query.aggregate,
            )

        indices = self._sample_indices(time.size, query.max_points)
        values = [
            SignalValues(
                signal=name,
                unit=available[name]["unit"],
                values=selected[name][indices].tolist(),
            )
            for name in query.signals
        ]
        return ResultQueryResponse(
            result_id=query.result_id,
            time_s=time[indices].tolist(),
            signals=values,
            downsampled=len(indices) < time.size,
        )

    def _put_npz(self, owner_id: str, arrays: Dict[str, np.ndarray]) -> Artifact:
        from io import BytesIO

        stream = BytesIO()
        np.savez_compressed(stream, **arrays)
        return self.artifacts.put_bytes(
            owner_id,
            ArtifactKind.RESULTS,
            "application/x-npz",
            stream.getvalue(),
            "results.npz",
        )

    def _read_record(self, owner_id: str, result_id: str):
        if not result_id.startswith("result_") or not result_id[7:].isalnum():
            raise ResultNotFoundError(result_id)
        try:
            record = json.loads(self._record_path(result_id).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ResultNotFoundError(result_id) from exc
        if record["owner_id"] != owner_id:
            raise ResultNotFoundError(result_id)
        return record

    def _record_path(self, result_id: str) -> Path:
        return self.root / f"{result_id}.json"

    @staticmethod
    def new_result_id() -> str:
        return f"result_{uuid4().hex}"

    @staticmethod
    def _sample_indices(size: int, max_points: int):
        if size <= max_points:
            return np.arange(size)
        return np.linspace(0, size - 1, max_points, dtype=int)

    @staticmethod
    def _aggregate(operation: str, time: np.ndarray, values: np.ndarray) -> float:
        if operation == "min":
            return float(np.min(values))
        if operation == "max":
            return float(np.max(values))
        if operation == "mean":
            return float(np.mean(values))
        if operation == "final":
            return float(values[-1])
        if operation == "time_of_min":
            return float(time[int(np.argmin(values))])
        if operation == "time_of_max":
            return float(time[int(np.argmax(values))])
        raise ValueError(f"unsupported aggregate: {operation}")
