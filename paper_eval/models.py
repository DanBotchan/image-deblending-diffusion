from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    experiment: str
    message: str
    pair: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "experiment": self.experiment,
            "pair": self.pair,
            "message": self.message,
        }


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    family: str
    scenario: str
    method: str
    protocol: Literal["single", "select_then_refine"]
    expected_pairs: int
    project_root: Path
    evaluation_csv: Path | None = None
    stage1_csv: Path | None = None
    stage2_csv: Path | None = None
    expected_stage1_candidates: int = 3
    selector: str = "clip"
    selection_score: str = "avg_clip_score"
    tie_break: Literal["error", "first_csv_row"] = "error"

    @classmethod
    def from_dict(cls, raw: dict[str, Any], project_root: Path) -> "ExperimentSpec":
        def resolve(key: str) -> Path | None:
            value = raw.get(key)
            if value is None:
                return None
            path = Path(value)
            return path if path.is_absolute() else project_root / path

        return cls(
            name=raw["name"],
            family=raw["family"],
            scenario=raw["scenario"],
            method=raw["method"],
            protocol=raw["protocol"],
            expected_pairs=int(raw["expected_pairs"]),
            project_root=project_root,
            evaluation_csv=resolve("evaluation_csv"),
            stage1_csv=resolve("stage1_csv"),
            stage2_csv=resolve("stage2_csv"),
            expected_stage1_candidates=int(raw.get("expected_stage1_candidates", 3)),
            selector=raw.get("selector", "clip"),
            selection_score=raw.get("selection_score", "avg_clip_score"),
            tie_break=raw.get("tie_break", "error"),
        )


@dataclass
class ProtocolResult:
    spec: ExperimentSpec
    canonical_rows: list[dict[str, str]] = field(default_factory=list)
    stage1_rows: list[dict[str, str]] = field(default_factory=list)
    stage2_rows: list[dict[str, str]] = field(default_factory=list)
    selected_samples: dict[str, str] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    input_files: list[Path] = field(default_factory=list)


@dataclass
class AuditResult:
    experiments: list[dict[str, Any]]
    issues: list[Issue]
    provenance: dict[str, Any]

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)
