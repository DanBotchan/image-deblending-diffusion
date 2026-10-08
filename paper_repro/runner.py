"""Run reproduction jobs as subprocesses, sharded over GPUs.

Jobs inside a phase are independent and run concurrently, one per GPU slot;
phases run in order. Every job is a plain command line, logged to
``<root>/logs/<job>.log`` and printed with ``--dry-run``, so any step can be
rerun by hand. Sampling scripts run with ``general.skip_finished=true``, so an
interrupted experiment resumes where it stopped.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]


@dataclass
class Job:
    name: str
    argv: list[str]
    done: Optional[Callable[[], bool]] = field(default=None, repr=False)

    def command(self) -> str:
        return " ".join(shlex.quote(part) for part in self.argv)


def python(script: str, *args: object) -> list[str]:
    return [sys.executable, str(REPO / script), *map(str, args)]


def shards(start: int, end: int, count: int) -> list[tuple[int, int]]:
    """Split [start, end) into at most ``count`` contiguous non-empty ranges."""
    count = max(1, min(count, end - start))
    edges = [start + (end - start) * i // count for i in range(count + 1)]
    return [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo]


def smoke_overrides(overrides: Sequence[str]) -> list[str]:
    """Shrink step and recurrence counts ~10x so a pipeline finishes in minutes.

    Every grid cell still runs (swept steps/starts stay distinct; recurrence
    counts are capped at 2), which exercises every code path. The numbers
    produced are meaningless.
    """
    result = []
    for item in overrides:
        key, _, value = item.partition("=")
        if key in ("sampler.timestep_respacing", "sampler.start_idx"):
            value = str(max(1, round(int(value) / 10)))
        elif key == "general.n_recurrence":
            value = str(min(int(value), 2))
        result.append(f"{key}={value}")
    return result


class Runner:
    def __init__(self, root: Path, gpus: Sequence[str], dry_run: bool = False):
        self.root = root
        self.gpus = list(gpus) or ["0"]
        self.dry_run = dry_run
        self.logs = root / "logs"

    @property
    def slots(self) -> int:
        return len(self.gpus)

    def run_phase(self, title: str, jobs: "Sequence[Job] | Callable[[], Sequence[Job]]") -> None:
        if callable(jobs):
            jobs = jobs()
        pending = [job for job in jobs if not (job.done and job.done())]
        skipped = len(jobs) - len(pending)
        print(f"\n== {title}: {len(pending)} job(s)" + (f", {skipped} already done" if skipped else ""),
              flush=True)
        if self.dry_run:
            for job in pending:
                print(f"[{job.name}] {job.command()}")
            return
        if not pending:
            return
        self.logs.mkdir(parents=True, exist_ok=True)
        failed: list[str] = []
        free = list(self.gpus)
        queue = list(pending)
        with ThreadPoolExecutor(max_workers=len(free)) as pool:
            active = {}
            while queue or active:
                while queue and free:
                    job, gpu = queue.pop(0), free.pop(0)
                    active[pool.submit(self._execute, job, gpu)] = (job, gpu)
                finished, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in finished:
                    job, gpu = active.pop(future)
                    free.append(gpu)
                    code, minutes = future.result()
                    status = "ok" if code == 0 else f"FAILED (exit {code})"
                    print(f"[{job.name}] gpu {gpu}: {status} after {minutes:.1f} min", flush=True)
                    if code:
                        failed.append(job.name)
        if failed:
            raise SystemExit(f"{len(failed)} job(s) failed: {failed}. Logs: {self.logs}")

    def _execute(self, job: Job, gpu: str) -> tuple[int, float]:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu, USE_TF="0", USE_FLAX="0", TRANSFORMERS_NO_TF="1",
                   HYDRA_FULL_ERROR="1")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")]))
        started = time.time()
        log = self.logs / f"{job.name}.log"
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n$ CUDA_VISIBLE_DEVICES={gpu} {job.command()}\n")
            handle.flush()
            code = subprocess.run(job.argv, cwd=REPO, env=env, stdout=handle,
                                  stderr=subprocess.STDOUT).returncode
        return code, (time.time() - started) / 60
