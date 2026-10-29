"""Per-vendor drift monitor.

Signals per document (from the code path):
  hard  : required anchor missing, parse error, FAIL check on code output,
          audit disagreement confirmed by the arbiter model
  medium: unseen layout lines (novelty), core layout lines missing
  soft  : WARN history checks (prices, quantities, timing)

States:
  learning -> no active parser yet
  stable   -> parser trusted, base audit rate
  suspect  -> something looks off; every document gets a targeted audit
  drift    -> confirmed; the parser is suspended, documents go to the LLM,
              recent code-path documents are quarantined for re-checking,
              and re-synthesis starts from fresh examples
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class DriftMonitor:
    vendor_id: str
    state: str = "learning"
    window: deque = field(default_factory=lambda: deque(maxlen=6))
    clean_streak: int = 0
    reason: str = ""
    events: list = field(default_factory=list)

    def _enter(self, state: str, reason: str, seq: int) -> None:
        if state != self.state:
            self.events.append({"seq": seq, "from": self.state, "to": state, "reason": reason})
        self.state, self.reason = state, reason

    def observe(self, seq: int, hard: list[str], medium: list[str], soft: list[str]) -> str:
        if self.state in ("learning", "drift"):
            return self.state
        self.window.append({"hard": hard, "medium": medium, "soft": soft})
        n_hard = sum(1 for w in self.window if w["hard"])
        n_soft = sum(1 for w in self.window if w["soft"] or w["medium"])
        if hard:
            # drift itself is only ever confirmed by comparing with a verified extraction (engine)
            self._enter("suspect", f"hard signal: {hard[:3]}", seq)
        elif n_soft >= 2:
            self._enter("suspect", f"repeated warnings: {(medium + soft)[:3] or self._last('soft')}", seq)
        if not (hard or medium or soft):
            self.clean_streak += 1
            if self.state == "suspect" and self.clean_streak >= 6:
                self._enter("stable", "6 clean documents in a row", seq)
        else:
            self.clean_streak = 0
        return self.state

    def confirm_drift(self, seq: int, reason: str) -> None:
        self._enter("drift", reason, seq)

    def recovered(self, seq: int, reason: str) -> None:
        self.window.clear()
        self.clean_streak = 0
        self._enter("stable", reason, seq)

    def _last(self, kind: str) -> list[str]:
        for w in reversed(self.window):
            if w[kind]:
                return w[kind]
        return []
