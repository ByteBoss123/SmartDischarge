"""Population Stability Index for monitoring score and feature drift."""
from __future__ import annotations

import numpy as np


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI of `actual` against `expected`, with quantile bins taken from `expected`.

    Rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 major shift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
