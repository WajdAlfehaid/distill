"""Declarative parser specs and their executor.

A compiled parser is *data*, not code: a JSON spec of line patterns. That makes
parsers safe to run (no generated code is executed), easy to diff between
versions, and easy for an LLM to propose or repair.

Spec shape
----------
{
  "num_style": "en", "date_style": "dmy_slash",
  "fields": {
     "<field>": {"mode": "same", "pattern": "<full-line regex with (?P<v>...)>", "type": "money"},
     "<field>": {"mode": "next", "label": "<full-line regex>", "pattern": "<full-line regex>", "type": "date"}
  },
  "required": ["invoice_number", ...],          # fields that must be found, else anchor miss
  "items": {"start": "<regex of the line before the items>", "row": "<full-line regex with named groups>",
            "max_gap": 2}
}
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from .formats import DATE_REGEX, NUM_REGEX, parse_date, parse_num
from .model import HEADER_FIELDS, Invoice, LineItem

FIELD_TYPES = {
    "invoice_number": "text", "po_number": "text", "vendor_tax_id": "text", "currency": "code",
    "invoice_date": "date", "due_date": "date",
    "subtotal": "money", "discount": "money", "tax": "money", "total": "money",
}


def value_pattern(ftype: str, num_style: str, date_style: str) -> str:
    if ftype == "money":
        return f"(?P<v>{NUM_REGEX[num_style]})"
    if ftype == "date":
        return f"(?P<v>{DATE_REGEX[date_style]})"
    if ftype == "code":
        return r"(?P<v>[A-Z]{3})"
    return r"(?P<v>\S+)"


@lru_cache(maxsize=4096)
def _rx(p: str) -> re.Pattern:
    return re.compile(p)


@dataclass
class ExecResult:
    invoice: Invoice
    missing_required: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    items_found: bool = True

    @property
    def anchors_ok(self) -> bool:
        return not self.missing_required and not self.parse_errors and self.items_found


def _convert(ftype: str, raw: str, spec: dict) -> tuple[object, bool]:
    if ftype == "money":
        v = parse_num(raw, spec["num_style"])
        return v, v is not None
    if ftype == "date":
        v = parse_date(raw, spec["date_style"])
        return v, v is not None
    return raw.strip(), True


def execute(spec: dict, text: str) -> ExecResult:
    lines = [ln.rstrip() for ln in text.splitlines()]
    nonempty = [(i, ln) for i, ln in enumerate(lines) if ln.strip()]
    inv = Invoice()
    res = ExecResult(invoice=inv)

    for fname, rule in spec.get("fields", {}).items():
        ftype = rule.get("type", FIELD_TYPES.get(fname, "text"))
        raw = None
        if rule["mode"] == "same":
            rx = _rx(rule["pattern"])
            for _, ln in nonempty:
                m = rx.fullmatch(ln.strip())
                if m:
                    raw = m.group("v")
                    break
        else:  # value on the next non-empty line after a label line
            lab, rx = _rx(rule["label"]), _rx(rule["pattern"])
            for k, (_, ln) in enumerate(nonempty[:-1]):
                if lab.fullmatch(ln.strip()):
                    m = rx.fullmatch(nonempty[k + 1][1].strip())
                    if m:
                        raw = m.group("v")
                        break
        if raw is None:
            if fname in spec.get("required", []):
                res.missing_required.append(fname)
            continue
        val, ok = _convert(ftype, raw, spec)
        if not ok:
            res.parse_errors.append(fname)
            continue
        setattr(inv, fname, val)

    items = spec.get("items")
    if items:
        start_rx, row_rx = _rx(items["start"]), _rx(items["row"])
        max_gap = items.get("max_gap", 2)
        started, gap = None, 0
        for idx, (_, ln) in enumerate(nonempty):
            s = ln.strip()
            if started is None:
                if start_rx.fullmatch(s):
                    started, gap = idx, 0
                continue
            m = row_rx.fullmatch(s)
            if m:
                q = parse_num(m.group("quantity"), spec["num_style"])
                p = parse_num(m.group("unit_price"), spec["num_style"])
                a = parse_num(m.group("amount"), spec["num_style"])
                if q is None or p is None or a is None:
                    res.parse_errors.append("line_items")
                    break
                inv.line_items.append(LineItem(m.group("description").strip(), q, p, a))
                gap = max_gap + 1  # once rows started, the first non-row ends the table
            elif inv.line_items or gap >= max_gap:
                break
            else:
                gap += 1
        if not inv.line_items:
            res.items_found = False
    # money fields that are legitimately absent (discount) default to 0
    if inv.discount is None and "discount" not in spec.get("required", []):
        inv.discount = 0.0
    return res


def describe(spec: dict) -> dict:
    """Small human-readable summary used by the dashboard and the trace."""
    return {
        "fields": sorted(spec.get("fields", {})),
        "required": spec.get("required", []),
        "num_style": spec.get("num_style"),
        "date_style": spec.get("date_style"),
        "items_row": spec.get("items", {}).get("row"),
    }


ALL_FIELDS = HEADER_FIELDS
