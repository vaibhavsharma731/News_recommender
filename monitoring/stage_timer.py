"""
Lightweight stage-timing primitives for NewsRec v2.0 monitoring.

Design:
  - StageTimer is a context manager — zero overhead on the happy path.
  - ExecutionContext accumulates stage results for the whole operation.
  - No DB calls here: callers serialise the context to dict and persist it.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class StageResult:
    name: str
    start_ms: float          # epoch ms at stage start
    end_ms: float            # epoch ms at stage end
    elapsed_ms: float        # wall-clock duration
    articles_processed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class StageTimer:
    """
    Context manager that times one stage of a pipeline.

    Usage::
        with StageTimer("BM25 retrieval") as t:
            scores = bm25.scores(query)
        print(t.elapsed_ms)   # → float
    """

    def __init__(self, name: str, ctx: Optional["ExecutionContext"] = None):
        self.name = name
        self.ctx = ctx
        self.elapsed_ms: float = 0.0
        self._t0: float = 0.0
        self.start_ms: float = 0.0
        self.end_ms: float = 0.0

    def __enter__(self) -> "StageTimer":
        self._t0 = time.perf_counter()
        self.start_ms = time.time() * 1000
        return self

    def __exit__(self, *_):
        self.end_ms = time.time() * 1000
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000
        if self.ctx is not None:
            self.ctx.record_stage(self)
        return False   # do not suppress exceptions

    def to_stage_result(self, articles_processed: int = 0, **extra) -> StageResult:
        return StageResult(
            name=self.name,
            start_ms=self.start_ms,
            end_ms=self.end_ms,
            elapsed_ms=round(self.elapsed_ms, 2),
            articles_processed=articles_processed,
            extra=extra,
        )


@contextmanager
def timed_stage(ctx: "ExecutionContext", name: str, articles_processed: int = 0, **extra):
    """
    Convenience context manager that creates a StageTimer and records it into ctx.

    Usage::
        with timed_stage(ctx, "BM25 retrieval", articles_processed=50):
            ...
    """
    t = StageTimer(name)
    try:
        yield t
    finally:
        t.__exit__(None, None, None)
        result = t.to_stage_result(articles_processed=articles_processed, **extra)
        ctx._stages.append(result)


class ExecutionContext:
    """
    Accumulates StageResult records for one pipeline execution.
    Thread-safe for the single-threaded pipeline; callers should not share across threads.
    """

    def __init__(self, operation: str, job_id: Optional[str] = None):
        self.operation = operation
        self.job_id = job_id
        self._stages: List[StageResult] = []
        self._start_time = time.time()

    def record_stage(self, timer: StageTimer, articles_processed: int = 0, **extra):
        """Record a completed StageTimer."""
        self._stages.append(timer.to_stage_result(articles_processed, **extra))

    def stage(self, name: str) -> StageTimer:
        """Return a StageTimer pre-bound to this context."""
        return StageTimer(name, ctx=self)

    def elapsed_ms(self) -> float:
        return (time.time() - self._start_time) * 1000

    def to_dict(self) -> dict:
        """Serialise to a JSON-able dict for DB storage."""
        stages_list = [s.to_dict() for s in self._stages]
        # Compact flat dict for quick DB column storage
        flat = {s.name: round(s.elapsed_ms, 1) for s in self._stages}
        return {
            "operation": self.operation,
            "job_id": self.job_id,
            "total_elapsed_ms": round(self.elapsed_ms(), 1),
            "stages": stages_list,
            "stage_timings_flat": flat,
        }

    def flat_timings(self) -> Dict[str, float]:
        """Return {stage_name: elapsed_ms} mapping, for compact JSON column."""
        return {s.name: round(s.elapsed_ms, 1) for s in self._stages}

    def get_stage(self, name: str) -> Optional[StageResult]:
        for s in self._stages:
            if s.name == name:
                return s
        return None
