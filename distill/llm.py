"""LLM tiers: real Claude via the Messages API, or a clearly-labelled simulation.

The real client uses only the standard library (urllib), so the project has no
SDK dependency. Set ANTHROPIC_API_KEY to use it.

The simulated client exists so the whole pipeline, the benchmark and the live
dashboard run offline and reproducibly. It returns the ground truth with
injected, realistic errors at configurable per-tier rates and bills tokens with
the same price table as the real client. Every number produced in simulated mode
is labelled SIMULATED in reports and on the dashboard.
"""
from __future__ import annotations

import copy
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

from .model import INVOICE, Record, Schema

# USD per million tokens (input, output). Source: platform.claude.com/docs/en/about-claude/pricing (Oct 2026).
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

SMALL_MODEL = os.environ.get("DISTILL_SMALL_MODEL", "claude-haiku-5-5")
LARGE_MODEL = os.environ.get("DISTILL_LARGE_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You extract structured data from supplier invoices (plain text from PDF/OCR).
Return ONLY a JSON object with exactly these keys:
  invoice_number (string), invoice_date (YYYY-MM-DD), due_date (YYYY-MM-DD),
  po_number (string or null), vendor_tax_id (string), currency (ISO 4217 code),
  subtotal, discount, tax, total (numbers; discount is 0 when absent),
  line_items: list of {description, quantity, unit_price, amount} (numbers as JSON numbers).
Rules: interpret dates and number separators using the document's own conventions and the
vendor's country; copy descriptions exactly; never invent values; do not include currency symbols."""


def prompt_for(schema: Schema) -> str:
    """System prompt for a schema. The invoice prompt is fixed; other schemas are described field by field."""
    if schema.name == "invoice":
        return SYSTEM_PROMPT
    lines = [f"You extract fields from {schema.description} (plain text produced by OCR, so expect damaged",
             "characters). Return ONLY a JSON object with exactly these keys:"]
    for f in schema.fields:
        kind = "a number" if f.type == "money" else "a string"
        lines.append(f"  {f.name}: {kind} - {f.label}{' (null if not on the document)' if f.optional else ''}")
    lines.append("Rules: copy printed values as they appear on the document (repair obvious OCR slips such as O for 0),")
    lines.append("never invent values, use null for anything that is not on the document.")
    return "\n".join(lines)


@dataclass
class LLMResult:
    invoice: Optional[Record]  # the extracted record (named for history; any schema)
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    simulated: bool = False
    error: str = ""
    raw: str = ""


def cost_of(model: str, tin: int, tout: int) -> float:
    pin, pout = PRICES.get(model, (3.0, 15.0))
    return tin / 1e6 * pin + tout / 1e6 * pout


def _parse_json_obj(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in model output")
    return json.loads(m.group(0))


# ------------------------------------------------------------------ real client
class AnthropicLLM:
    simulated = False

    def __init__(self, api_key: Optional[str] = None, small: str = SMALL_MODEL, large: str = LARGE_MODEL,
                 timeout: float = 60.0):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.models = {"small": small, "large": large}
        self.timeout = timeout

    def _call(self, model: str, system: str, user: str, max_tokens: int = 2000) -> tuple[str, int, int, float]:
        body = json.dumps({"model": model, "max_tokens": max_tokens, "temperature": 0,
                           "system": system, "messages": [{"role": "user", "content": user}]}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, method="POST", headers={
            "x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        t0 = time.perf_counter()
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 529) and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Anthropic API error {e.code}: {e.read()[:300]!r}") from e
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        u = data.get("usage", {})
        return text, int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0)), (time.perf_counter() - t0) * 1000

    def extract(self, doc_id: str, text: str, tier: str = "small", schema: Schema = INVOICE) -> LLMResult:
        model = self.models[tier]
        try:
            out, tin, tout, ms = self._call(model, prompt_for(schema), f"Document text:\n<<<\n{text}\n>>>")
            inv = Record.from_dict(_parse_json_obj(out), schema)
            return LLMResult(inv, model, tin, tout, cost_of(model, tin, tout), ms, raw=out)
        except Exception as e:  # network/JSON problems become a failed extraction, never a crash
            return LLMResult(None, model, error=f"{type(e).__name__}: {e}")

    def propose_spec(self, examples: list, dsl_doc: str) -> tuple[list[dict], float]:
        """Ask the large model to write a parser spec in the JSON DSL. Returns (specs, cost)."""
        model = self.models["large"]
        shots = "\n\n".join(f"### Document\n{e.text}\n### Correct extraction\n{json.dumps(e.labels.to_dict())}"
                            for e in examples[:3])
        user = (f"{dsl_doc}\n\nWrite ONE parser spec (JSON only) that extracts every field for documents with this "
                f"layout.\n\n{shots}")
        try:
            out, tin, tout, _ = self._call(model, "You write precise regex-based parser specs. Output JSON only.", user, 3000)
            spec = _parse_json_obj(out)
            spec["generator"] = "llm"
            return [spec], cost_of(model, tin, tout)
        except Exception:
            return [], 0.0


# ------------------------------------------------------------------ simulated client
TIER_PROFILE = {
    # base P(document contains an extraction error), extra per line of text, base latency (ms), ms per output token,
    # P(a damaged value is read back correctly)
    "small": {"p_err": 0.03, "per_line": 0.0008, "lat": 700, "per_tok": 4, "recover": 0.6},
    "large": {"p_err": 0.008, "per_line": 0.0002, "lat": 1800, "per_tok": 12, "recover": 0.8},
}


class SimulatedLLM:
    """Ground truth + realistic, injected mistakes. Clearly labelled as simulated everywhere.

    `oracle(doc_id)` returns the true record. `damage(doc_id)` (optional) returns the fields
    whose printed value is damaged in the document; the simulated model reads those back
    correctly only with probability `recover`, as a real model reading OCR text would.
    With `p_err=0` and no damage this is a perfect teacher (used to measure the parser layer
    in isolation on public datasets, where the labels are human).
    """

    simulated = True

    def __init__(self, oracle: Callable[[str], Record], small: str = SMALL_MODEL, large: str = LARGE_MODEL,
                 seed: int = 0, profiles: Optional[dict] = None, damage: Optional[Callable[[str], dict]] = None):
        self.oracle = oracle
        self.damage = damage
        self.models = {"small": small, "large": large}
        self.seed = seed
        self.profiles = profiles or TIER_PROFILE

    @staticmethod
    def tokens_for(text: str, inv: Record) -> tuple[int, int]:
        tin = int(len(text) / 3.6) + len(SYSTEM_PROMPT) // 4 + 20
        tout = 40 + 9 * len(inv.schema.fields) + 32 * len(inv.line_items)
        return tin, tout

    def extract(self, doc_id: str, text: str, tier: str = "small", schema: Schema = INVOICE) -> LLMResult:
        model = self.models[tier]
        prof = self.profiles[tier]
        truth = self.oracle(doc_id)
        inv = copy.deepcopy(truth)
        rng = random.Random(f"{self.seed}:{doc_id}:{tier}")
        notes = []
        n_lines = text.count("\n") + 1
        if rng.random() < prof["p_err"] + prof.get("per_line", 0) * n_lines:
            notes.append(_inject_error(inv, rng))
        if self.damage:
            for fname, misread in (self.damage(doc_id) or {}).items():
                if rng.random() > prof.get("recover", 1.0):
                    _apply_misread(inv, fname, misread)
                    notes.append(f"misread damaged {fname}")
        tin, tout = self.tokens_for(text, truth)
        ms = prof["lat"] + prof["per_tok"] * tout
        return LLMResult(inv, model, tin, tout, cost_of(model, tin, tout), ms, simulated=True, raw="; ".join(notes))

    def propose_spec(self, examples: list, dsl_doc: str) -> tuple[list[dict], float]:
        return [], 0.0


def _apply_misread(inv: Record, fname: str, misread) -> None:
    if fname.startswith("line_items[") and inv.line_items:
        i = int(fname[11:-1])
        if i < len(inv.line_items) and isinstance(misread, dict):
            for k, v in misread.items():
                setattr(inv.line_items[i], k, v)
    elif fname in inv.values:
        inv.values[fname] = misread


def _inject_error(inv: Record, rng: random.Random) -> str:
    """Realistic model slips. About half are catchable by arithmetic, half are not."""
    if inv.schema.name != "invoice":
        names = [f.name for f in inv.schema.fields if inv.values.get(f.name) not in (None, "")]
        if not names:
            return "none"
        f = rng.choice(names)
        v = inv.values[f]
        if isinstance(v, float):
            inv.values[f] = round(v + rng.choice([0.1, 1.0, -1.0, 10.0]), 2)
        else:
            s = str(v)
            i = rng.randrange(len(s))
            inv.values[f] = s[:i] + s[i + 1:] if len(s) > 3 else s + "1"
        return f"slip in {f}"
    kind = rng.choice(["amount_digit", "drop_line", "date_swap", "desc_typo", "invoice_no", "qty_misread"])
    if kind == "amount_digit" and inv.line_items:
        li = rng.choice(inv.line_items)
        li.amount = round(li.amount + rng.choice([9, 90, 0.9, -9]), 2)
    elif kind == "drop_line" and len(inv.line_items) > 1:
        inv.line_items.pop(rng.randrange(len(inv.line_items)))
    elif kind == "date_swap" and inv.invoice_date:
        y, m, d = inv.invoice_date.split("-")
        if int(d) <= 12 and d != m:
            inv.invoice_date = f"{y}-{d}-{m}"
        else:
            inv.invoice_date = f"{y}-{m}-{int(d) - 1:02d}" if int(d) > 1 else inv.invoice_date
    elif kind == "desc_typo" and inv.line_items:
        li = rng.choice(inv.line_items)
        li.description = li.description.replace("e", "c", 1) if "e" in li.description else li.description + "."
    elif kind == "invoice_no" and inv.invoice_number:
        s = inv.invoice_number
        inv.invoice_number = s[:-1] + str((int(s[-1]) + 1) % 10) if s[-1].isdigit() else s + "1"
    elif kind == "qty_misread" and inv.line_items:
        li = rng.choice(inv.line_items)
        li.quantity = li.quantity + 1
    return kind
