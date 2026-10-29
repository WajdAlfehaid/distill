"""Risk-, uncertainty-, cost- and latency-aware routing.

Instead of a fixed code -> small -> large cascade, every document gets a route
chosen by a small, transparent policy. Each decision carries the reasons that
produced it, so a reviewer can read *why* a document cost what it cost.

Routes
  code    accept the compiled parser's output (zero LLM tokens)
  audit   run the parser AND a small model; accept on agreement, arbitrate on disagreement
  llm     extract with a model (small first, large for high risk or on failed checks)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .audit import audit_rate, sampled

FX_TO_USD = {"USD": 1.0, "SAR": 0.2667, "AED": 0.2723, "EUR": 1.08, "GBP": 1.27}

PROFILES = {
    # risk threshold for a targeted audit, audit floor, value that counts as "high"
    "economy": {"risk_threshold": 0.75, "audit_floor": 0.01, "high_value_usd": 100_000, "large_first_usd": 250_000},
    "balanced": {"risk_threshold": 0.5, "audit_floor": 0.02, "high_value_usd": 50_000, "large_first_usd": 100_000},
    "assurance": {"risk_threshold": 0.3, "audit_floor": 0.05, "high_value_usd": 20_000, "large_first_usd": 50_000},
}


@dataclass
class RouteInput:
    doc_id: str
    vendor_known: bool
    has_parser: bool
    drift_state: str
    anchors_ok: bool = True
    code_fails: list = field(default_factory=list)
    code_warns: list = field(default_factory=list)
    novelty: list = field(default_factory=list)
    value: float = 0.0
    currency: str = "USD"
    parser_clean_audits: int = 0


@dataclass
class Route:
    path: str  # code | audit | llm
    model: str = "small"  # for llm/audit: small | large
    risk: float = 0.0
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"path": self.path, "model": self.model, "risk": round(self.risk, 3), "reasons": self.reasons}


def decide(x: RouteInput, profile: str = "balanced") -> Route:
    P = PROFILES[profile]
    usd = x.value * FX_TO_USD.get(x.currency, 1.0)

    if not x.vendor_known:
        return Route("llm", "large", 1.0, ["vendor not in master data -> large model + human review"])
    if not x.has_parser:
        return Route("llm", "large" if usd >= P["large_first_usd"] else "small", 0.8,
                     ["no compiled parser for this vendor yet (learning from this document)"])
    if x.drift_state == "drift":
        return Route("llm", "large" if usd >= P["large_first_usd"] else "small", 0.9,
                     ["parser suspended: drift confirmed, re-synthesis in progress"])
    if not x.anchors_ok:
        return Route("llm", "small", 0.9, ["parser could not find required fields (layout changed?)"])
    if x.code_fails:
        return Route("llm", "small", 0.9, [f"parser output failed checks: {', '.join(x.code_fails[:3])}"])
    if x.novelty:
        return Route("llm", "small", 0.7, [f"{len(x.novelty)} unseen layout line(s), e.g. {x.novelty[0]!r}"])

    risk, reasons = 0.0, []
    if usd >= P["high_value_usd"]:
        risk += 0.6
        reasons.append(f"high value ({usd:,.0f} USD)")
    elif usd >= P["high_value_usd"] / 2:
        risk += 0.25
        reasons.append(f"elevated value ({usd:,.0f} USD)")
    if x.code_warns:
        risk += 0.5
        reasons.append(f"history warning: {', '.join(x.code_warns[:2])}")
    if x.drift_state == "suspect":
        risk += 0.6
        reasons.append("drift monitor suspicious")
    risk = min(risk, 1.0)

    if risk >= P["risk_threshold"]:
        return Route("audit", "small", risk, ["targeted audit: " + "; ".join(reasons)])
    rate = audit_rate(x.parser_clean_audits, floor=P["audit_floor"])
    if sampled(x.doc_id, rate):
        return Route("audit", "small", risk, [f"random audit (rate {rate:.0%} for a parser with {x.parser_clean_audits} clean audits)"])
    return Route("code", "", risk, ["compiled parser, all checks passed" + (f"; {', '.join(reasons)}" if reasons else "")])
