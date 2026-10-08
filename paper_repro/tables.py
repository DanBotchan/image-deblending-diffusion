"""Summarize evaluated runs into the paper's tables (``summary.csv`` / ``summary.md``).

Reads only the per-row CSVs written by ``scripts/evaluate_artifact_tree.py``.
Every number is a mean over pairs of the two-source average, after the
minimum-MSE source-order assignment.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Iterable

from .experiments import (AUTO_ARMS, BUDGETS, RECURRENCE_STAGE1, RECURRENCE_STAGE2,
                          TEXT_GUIDANCE_CELLS, TIMESTEP_STAGE1, TIMESTEP_STAGE2)

METRICS = ("psnr", "ssim", "lpips")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def means(rows: Iterable[dict[str, str]]) -> dict[str, float]:
    rows = list(rows)
    result: dict[str, Any] = {"n_pairs": len(rows)}
    for metric in METRICS:
        result[metric] = statistics.fmean(float(row[f"avg_{metric}"]) for row in rows)
    return result


def write(out: Path, rows: list[dict[str, Any]], title: str) -> None:
    if not rows:
        raise RuntimeError(f"No results to summarize in {out}")
    fields = list(rows[0])
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def cell(value: Any) -> str:
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |",
             "|" + "|".join("---" for _ in fields) + "|"]
    lines += ["| " + " | ".join(cell(row[f]) for f in fields) + " |" for row in rows]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main_table(out: Path) -> None:
    final = means(read_rows(out / "metrics" / "stage2"))
    coarse = means(read_rows(out / "metrics" / "stage1_selected"))
    write(out, [{"stage": "Stage 1 (selected, resized to 256)", **coarse},
                {"stage": "Stage 2 (final)", **final}],
          "Main result and two-stage ablation")


def text_guidance_table(out: Path) -> None:
    rows = [{"prompts": mode, "cfg": f"{cfg:g}", **means(read_rows(out / f"{mode}_cfg{cfg:g}" / "metrics"))}
            for mode, cfg in TEXT_GUIDANCE_CELLS]
    write(out, rows, "Text-guidance ablation")


def _grid_table(out: Path, title: str, stage1: dict[str, Any], stage2: dict[str, Any]) -> None:
    rows = []
    for s1_name, s1_value in stage1.items():
        rows.append({"stage1": s1_value, "stage2": "stage1 only",
                     **means(read_rows(out / s1_name / "metrics_stage1"))})
        for s2_name, s2_value in stage2.items():
            rows.append({"stage1": s1_value, "stage2": s2_value,
                         **means(read_rows(out / s1_name / f"metrics_{s2_name}"))})
    write(out, rows, title)


def timestep_table(out: Path) -> None:
    _grid_table(out, "Starting-timestep grid",
                {f"s1_{t}": t for t in TIMESTEP_STAGE1}, {f"s2_{t}": t for t in TIMESTEP_STAGE2})


def recurrence_table(out: Path) -> None:
    _grid_table(out, "Recurrence grid",
                {f"n1_{n}": n for n in RECURRENCE_STAGE1}, {f"n2_{n}": n for n in RECURRENCE_STAGE2})


def _selected(out: Path, manifest: str) -> set[tuple[str, str]]:
    with (out / manifest).open(newline="", encoding="utf-8") as handle:
        return {(row["pair"], row["sample"]) for row in csv.DictReader(handle)}


def selector_table(out: Path) -> None:
    rows = read_rows(out / "metrics")
    picks = {name: _selected(out, f"{name}.csv") for name in ("pickscore", "clip")}
    table = [{"selector": name, **means(r for r in rows if (r["pair"], r["sample"]) in chosen)}
             for name, chosen in picks.items()]
    agreement = len(picks["pickscore"] & picks["clip"]) / max(1, len(picks["pickscore"]))
    for row in table:
        row["selector_agreement"] = agreement
    write(out, table, "PickScore vs. CLIP selection (same candidates, matching Stage-2 refinements)")


def budget_table(out: Path) -> None:
    rows = {(r["pair"], r["sample"]): r for r in read_rows(out / "metrics")}
    by_budget = json.loads((out / "winners.budgets.json").read_text(encoding="utf-8"))
    budgets = sorted({int(b) for choice in by_budget.values() for b in choice})
    table = [{"budget": b, **means(rows[pair, choice[str(b)]] for pair, choice in sorted(by_budget.items()))}
             for b in budgets if b in BUDGETS]
    write(out, table, "Stage-1 candidate budget (PickScore selection, Stage-2 metrics)")


def auto_prompts_table(out: Path) -> None:
    table = []
    for study in ("preliminary", "confirmation"):
        for arm in AUTO_ARMS:
            path = out / study / arm / "metrics"
            if path.with_suffix(".csv").is_file():
                table.append({"study": study, "prompts": arm, **means(read_rows(path))})
    write(out, table, "Automated prompts")


TABLES: dict[str, Callable[[Path], None]] = {
    "main": main_table,
    "text-guidance": text_guidance_table,
    "timestep": timestep_table,
    "recurrence": recurrence_table,
    "selector": selector_table,
    "candidate-budget": budget_table,
    "auto-prompts": auto_prompts_table,
}
