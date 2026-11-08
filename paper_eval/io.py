from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .models import ExperimentSpec, Issue


IDENTITY_FIELDS = ("pair", "sample")


def finite_number(row: dict[str, str], field: str) -> float | None:
    """Read a finite CSV number, treating missing or invalid values as absent."""
    try:
        value = float(row.get(field, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def load_manifest(path: Path) -> tuple[Path, list[ExperimentSpec]]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    project_root = Path(raw["project_root"])
    specs = [ExperimentSpec.from_dict(item, project_root) for item in raw["experiments"]]
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("Manifest experiment names must be unique")
    return project_root, specs


def resolve_artifact_path(project_root: Path, raw: str) -> Path:
    """Resolve absolute paths directly and relative paths from project_root."""
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def read_rows(path: Path, experiment: str) -> tuple[list[dict[str, str]], list[Issue]]:
    issues: list[Issue] = []
    if not path.exists():
        return [], [Issue("error", "missing_input", experiment, f"Missing CSV: {path}")]
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = set(IDENTITY_FIELDS) - fields
        if missing:
            return [], [
                Issue("error", "missing_columns", experiment, f"{path} lacks {sorted(missing)}")
            ]
        rows = list(reader)

    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["pair"], row["sample"])
        if key in seen:
            issues.append(
                Issue("error", "duplicate_candidate", experiment, f"Duplicate row {key}", row["pair"])
            )
        seen.add(key)
    return rows, issues


def group_by_pair(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["pair"], []).append(row)
    return grouped


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
