from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def calibrate_threshold(benign_scores: Sequence[float], target_fpr: float = 0.01) -> float:
    """Smallest threshold t with FPR(score >= t) <= target_fpr on benign scores."""
    s = np.asarray(benign_scores, dtype=float)
    if s.size == 0:
        raise ValueError("No benign scores given for calibration.")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be in (0, 1).")
    n = s.size
    sorted_s = np.sort(s)
    uniq = np.unique(s)
    fprs = (n - np.searchsorted(sorted_s, uniq, side="left")) / n
    ok = np.nonzero(fprs <= target_fpr)[0]
    if ok.size:
        return float(uniq[ok[0]])
    # Even the highest benign score gives FPR > target (ties): go just above it.
    return float(np.nextafter(uniq[-1], np.inf))


def save_thresholds(path: str | Path, thresholds: dict[str, float]) -> None:
    path = Path(path)
    existing: dict[str, float] = {}
    if path.exists():
        existing = json.loads(path.read_text())
    existing.update(thresholds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2))


def load_threshold(path: str | Path, backend: str) -> float:
    data = json.loads(Path(path).read_text())
    if backend not in data:
        raise KeyError(f"No threshold for {backend!r} in {path}; have {list(data)}")
    return float(data[backend])
