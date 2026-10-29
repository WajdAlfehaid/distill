"""Adaptive audits and reliability bounds.

An audit re-extracts a code-path document with an LLM and compares. Audits are
expensive, so the rate adapts to the evidence behind each parser:

  * a freshly promoted parser is audited heavily, and the rate decays as clean
    audits accumulate, down to a floor (default 2%);
  * targeted audits fire whenever the drift monitor is suspicious, a history
    check warns, or the invoice value is above a threshold.

Every parser carries an exact one-sided 95% upper bound on its error rate
(Clopper-Pearson). With zero errors in n audits this is ~3/n, the "rule of three".
"""
from __future__ import annotations

import hashlib
import math


def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    total, term = 0.0, (1 - p) ** n
    for i in range(0, k + 1):
        if i > 0:
            term *= (n - i + 1) / i * p / (1 - p)
        total += term
    return min(1.0, total)


def error_upper_bound(errors: int, n: int, confidence: float = 0.95) -> float:
    """Exact one-sided upper confidence bound on the true error rate."""
    if n <= 0:
        return 1.0
    if errors >= n:
        return 1.0
    alpha = 1 - confidence
    lo, hi = errors / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _binom_cdf(errors, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def rule_of_three(n: int) -> float:
    return 1.0 if n <= 0 else min(1.0, 3.0 / n)


def audit_rate(clean_audits: int, floor: float = 0.02, warmup: int = 3, half_life: int = 4) -> float:
    """1.0 for the first `warmup` audits, then halves every `half_life` clean audits, never below `floor`."""
    if clean_audits < warmup:
        return 1.0
    return max(floor, 0.5 ** ((clean_audits - warmup) / half_life))


def sampled(doc_id: str, rate: float, salt: str = "audit") -> bool:
    """Deterministic sampling so runs are reproducible and auditable."""
    h = int(hashlib.sha256(f"{salt}:{doc_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < rate


def describe_bound(errors: int, n: int) -> str:
    ub = error_upper_bound(errors, n)
    if n == 0:
        return "no audits yet"
    return f"{errors} errors in {n} audits -> true error rate < {ub:.1%} (95% confidence)"


assert math.isclose(error_upper_bound(0, 300), 0.00994, rel_tol=0.02)  # ~3/300
