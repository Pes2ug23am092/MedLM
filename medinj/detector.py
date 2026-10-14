from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .backends import Backend, create_backend
from .normalize import normalize_text


@dataclass(frozen=True)
class Result:
    label: str          # "injected" or "clean"
    score: float        # max over windows, higher = more likely injected
    threshold: float    # decision threshold used (score >= threshold -> injected)
    backend: str
    latency_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


class Detector:
    def __init__(
        self,
        backend: str | Backend = "prompt_guard_2",
        threshold: float | None = None,
        shadow_mode: bool = True,
        log_path: str | Path | None = "logs/shadow_decisions.jsonl",
        log_text: bool = False,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.backend: Backend = (
            create_backend(backend, device=device) if isinstance(backend, str) else backend
        )
        self.threshold = (
            float(threshold) if threshold is not None else self.backend.default_threshold
        )
        self.shadow_mode = shadow_mode
        self.log_path = Path(log_path) if log_path else None
        self.log_text = log_text
        self.batch_size = batch_size
        self._lock = threading.Lock()
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def device(self) -> str:
        return str(getattr(self.backend, "device", "cpu"))

    def score_many(
        self, texts: Sequence[str], batch_size: int | None = None
    ) -> list[float]:
        """Batched scoring (used for evaluation). Applies normalization."""
        normed = [normalize_text(t) for t in texts]
        return self.backend.score_texts(normed, batch_size or self.batch_size)

    def detect(self, text: str) -> Result:
        t0 = time.perf_counter()
        score = self.backend.score_texts([normalize_text(text)], 1)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        label = "injected" if score >= self.threshold else "clean"
        result = Result(
            label=label,
            score=float(score),
            threshold=self.threshold,
            backend=self.backend.name,
            latency_ms=latency_ms,
        )
        self._log(result, text)
        return result

    def should_block(self, result: Result) -> bool:
        """The pipeline should call this: never blocks while shadow_mode is on."""
        return (not self.shadow_mode) and result.label == "injected"

    def _log(self, result: Result, raw_text: str) -> None:
        if not self.log_path:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **result.as_dict(),
            "shadow_mode": self.shadow_mode,
            "enforced": not self.shadow_mode,
            "n_chars": len(raw_text),
            "sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        }
        if self.log_text:
            record["text"] = raw_text
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


# --------------------------------------------------------------------------
# Module-level convenience: detect(text) -> Result
# --------------------------------------------------------------------------

_default: Detector | None = None


def configure(**kwargs) -> Detector:
    """configure(backend="deberta_v2", threshold=0.9, shadow_mode=True, ...)"""
    global _default
    _default = Detector(**kwargs)
    return _default


def detect(text: str) -> Result:
    global _default
    if _default is None:
        _default = Detector()
    return _default.detect(text)
