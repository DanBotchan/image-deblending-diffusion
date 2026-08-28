from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .aggregate import aggregate_metrics
from .io import load_manifest, sha256_file
from .models import AuditResult, Issue
from .protocols import evaluate_protocol
from .timing import summarize_timing
from .verification import verify_stored_metrics


def run_audit(
    manifest: Path,
    check_paths: bool = False,
    verify_images: bool = False,
    verify_lpips: bool = False,
) -> AuditResult:
    project_root, specs = load_manifest(manifest)
    experiments: list[dict[str, Any]] = []
    issues: list[Issue] = []
    fingerprints: dict[str, str] = {}

    for spec in specs:
        protocol = evaluate_protocol(spec, check_paths=check_paths)
        metric_summary, metric_issues = aggregate_metrics(spec.name, protocol.canonical_rows)
        timing_summary, timing_issues = summarize_timing(protocol)
        verification_summary = None
        verification_issues: list[Issue] = []
        if verify_images:
            verification_summary, verification_issues = verify_stored_metrics(
                spec, protocol.canonical_rows, include_lpips=verify_lpips
            )
        all_issues = protocol.issues + metric_issues + timing_issues + verification_issues
        issues.extend(all_issues)
        for path in protocol.input_files:
            if path.exists():
                fingerprints[str(path)] = sha256_file(path)
        experiments.append(
            {
                "name": spec.name,
                "family": spec.family,
                "scenario": spec.scenario,
                "method": spec.method,
                "protocol": spec.protocol,
                "selector": spec.selector if spec.protocol == "select_then_refine" else None,
                "metrics": metric_summary,
                "timing": timing_summary,
                "verification": verification_summary,
                "selected_pair_count": len(protocol.selected_samples),
                "issue_count": len(all_issues),
                "error_count": sum(issue.severity == "error" for issue in all_issues),
            }
        )

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "project_root": str(project_root),
        "input_sha256": fingerprints,
        "image_verification": verify_images,
        "lpips_verification": verify_lpips,
    }
    return AuditResult(experiments=experiments, issues=issues, provenance=provenance)
