"""Declarative parser specs and their executor.

A compiled parser is *data*, not code: a JSON spec of line patterns run by this
fixed interpreter. Parsers are safe to run (nothing generated is executed), easy
to diff between versions, and easy for an LLM to propose or repair.

Spec shape
----------
{
  "schema": "invoice",
  "num_style": "en", "date_style": "dmy_slash",
  "fields": {
     "<f>": {"mode": "same", "pattern": "<full-line regex with (?P<v>...)>", "type": "money",
             "label_text": "Subtotal:", "value": "<value regex>"},
     "<f>": {"mode": "next", "label": "<full-line regex>", "pattern": "<full-line regex>", "type": "date",
             "label_text": "Due Date"},
     "<f>": {"mode": "const", "value": "ACME TRADING SDN BHD", "type": "raw"}
  },
  "required": ["invoice_number", ...],
  "items": {"start": "<regex of the line before the items>", "row": "<regex with named groups>", "max_gap": 2}
}

Robustness to scanning noise is built in, and reported, so the engine can tell
"this copy is smudged" apart from "this parser is wrong":
  * numbers and dates tolerate look-alike characters (O/0, l/1, S/5, B/8);
  * when a label is damaged, a fuzzy label match recovers the value (recorded in `fuzzy`);
  * wrapped item descriptions and tables continued on a second page are followed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Optional

from .formats import DATE_REGEX, NUM_REGEX, parse_date, parse_num
from .model import INVOICE, LineItem, Record, Schema
from .ocr import canon, fuzzy_label, numeric_fix

OCR_DIGIT = r"[\dOoIlSB]"

# kept for older imports: field types of the invoice schema
FIELD_TYPES = INVOICE.types


def _ocr(rx: str) -> str:
    return rx.replace(r"\d", OCR_DIGIT)


NUM_REGEX_OCR = {k: _ocr(v) for k, v in NUM_REGEX.items()}
DATE_REGEX_OCR = {k: _ocr(v) for k, v in DATE_REGEX.items()}


def value_pattern(ftype: str, num_style: str, date_style: str, raw_value: Optional[str] = None) -> str:
    if ftype == "money":
        return f"(?P<v>{NUM_REGEX_OCR[num_style]})"
    if ftype == "date":
        return f"(?P<v>{DATE_REGEX_OCR[date_style]})"
    if ftype == "code":
        return r"(?P<v>[A-Z]{3})"
    if ftype == "raw":
        return f"(?P<v>{raw_value or '.+?'})"
    return r"(?P<v>\S+)"


@lru_cache(maxsize=8192)
def _rx(p: str) -> re.Pattern:
    return re.compile(p)


@dataclass
class ExecResult:
    invoice: Record  # the extracted record (named for history; any schema)
    missing_required: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    fuzzy: list[str] = field(default_factory=list)  # fields found only through a fuzzy (damaged) label
    ocr_fixed: list[str] = field(default_factory=list)  # values that needed look-alike character repair
    items_found: bool = True

    @property
    def anchors_ok(self) -> bool:
        return not self.missing_required and not self.parse_errors and self.items_found

    @property
    def record(self) -> Record:
        return self.invoice


def _convert(ftype: str, raw: str, spec: dict) -> tuple[object, bool, bool]:
    """-> (value, ok, needed_ocr_repair)"""
    if ftype == "money":
        v = parse_num(raw, spec["num_style"])
        if v is None:
            v2 = parse_num(numeric_fix(raw), spec["num_style"])
            return v2, v2 is not None, v2 is not None
        return v, True, False
    if ftype == "date":
        v = parse_date(raw, spec["date_style"])
        if v is None:
            v2 = parse_date(numeric_fix(raw), spec["date_style"])
            return v2, v2 is not None, v2 is not None
        return v, True, False
    return raw.strip(), True, False


def _const_present(value: str, text: str) -> bool:
    """A constant field (vendor name, address) must still be printed on the document, allowing OCR damage."""
    toks = [canon(t) for t in re.findall(r"[A-Za-z0-9]{3,}", value)]
    if not toks:
        return canon(value) in canon(text)
    doc = {canon(t) for t in re.findall(r"[A-Za-z0-9]{3,}", text)}
    hits = 0
    for t in toks:
        if t in doc or any(abs(len(d) - len(t)) <= 2 and SequenceMatcher(None, t, d).ratio() >= 0.8 for d in doc):
            hits += 1
    return hits / len(toks) >= 0.75


def _strict_same(rule: dict, lines: list[str]) -> Optional[str]:
    rx = _rx(rule["pattern"])
    for ln in lines:
        m = rx.fullmatch(ln)
        if m:
            return m.group("v")
    return None


def _fuzzy_same(rule: dict, lines: list[str]) -> Optional[str]:
    vp = rule.get("value")
    lab = rule.get("label_text", "")
    if not vp or not lab or vp in (".+?", r"\S+"):
        return None  # an open-ended value has no shape to anchor a fuzzy search on
    vrx = _rx(vp)
    for ln in lines:
        for m in vrx.finditer(ln):
            prefix = ln[:m.start()].strip()
            if prefix and fuzzy_label(prefix, lab):
                return m.group(0)
    return None


def _next_mode(rule: dict, lines: list[str], fuzzy: bool) -> Optional[str]:
    lab, rx = _rx(rule["label"]), _rx(rule["pattern"])
    for k, ln in enumerate(lines[:-1]):
        ok = lab.fullmatch(ln) if not fuzzy else fuzzy_label(ln, rule.get("label_text", ""))
        if ok:
            m = rx.fullmatch(lines[k + 1])
            if m:
                return m.group("v")
    return None


def execute(spec: dict, text: str, schema: Optional[Schema] = None, words: Optional[list] = None) -> ExecResult:
    from .doctypes import schema_by_name  # local import: doctypes imports parser

    schema = schema or schema_by_name(spec.get("schema", "invoice"))
    all_lines = [ln.rstrip() for ln in text.splitlines()]
    lines = [ln.strip() for ln in all_lines if ln.strip()]
    rec = Record(schema)
    res = ExecResult(invoice=rec)
    required = set(spec.get("required", []))
    aligned = None  # words registered onto the template's reference page, computed on first use

    for fname, rule in spec.get("fields", {}).items():
        if fname not in rec.values:
            continue
        ftype = rule.get("type") or schema.types.get(fname, "text")
        mode = rule["mode"]
        raw, fuzzy_hit = None, False
        if mode == "const":
            if _const_present(rule["value"], text):
                rec.values[fname] = rule["value"]
            elif fname in required:
                res.missing_required.append(fname)
            continue
        if mode == "zone":
            from .zones import execute_zone, register
            if words and aligned is None:
                aligned, ok = register(words, spec.get("registration"))
                if not ok:
                    res.fuzzy.append("_registration")
            raw = execute_zone(rule, aligned) if words else None
        elif mode == "same":
            raw = _strict_same(rule, lines)
            if raw is None:
                raw = _fuzzy_same(rule, lines)
                fuzzy_hit = raw is not None
        else:
            raw = _next_mode(rule, lines, fuzzy=False)
            if raw is None and rule.get("label_text"):
                raw = _next_mode(rule, lines, fuzzy=True)
                fuzzy_hit = raw is not None
        if raw is None:
            if fname in required:
                res.missing_required.append(fname)
            continue
        val, ok, repaired = _convert(ftype, raw, spec)
        if not ok:
            res.parse_errors.append(fname)
            continue
        rec.values[fname] = val
        if fuzzy_hit:
            res.fuzzy.append(fname)
        if repaired:
            res.ocr_fixed.append(fname)

    items = spec.get("items")
    if items and schema.items:
        _execute_items(items, spec, all_lines, rec, res)
        if not rec.line_items:
            res.items_found = False
    if schema.name == "invoice" and rec.discount is None and "discount" not in required:
        rec.discount = 0.0
    return res


def _execute_items(items: dict, spec: dict, all_lines: list[str], rec: Record, res: ExecResult) -> None:
    """Read item rows block by block: a table may continue after a page break under a repeated header,
    and a long description may wrap onto the next (usually indented) line."""
    start_rx, row_rx = _rx(items["start"]), _rx(items["row"])
    max_gap = items.get("max_gap", 2)
    ns = spec["num_style"]
    in_block, gap, block_rows, last_was_row = False, 0, 0, False
    for ln in all_lines:
        s = ln.strip()
        if not in_block:
            if s and start_rx.fullmatch(s):
                in_block, gap, block_rows, last_was_row = True, 0, 0, False
            continue
        m = row_rx.fullmatch(s) if s else None
        if m:
            vals = []
            for g in ("quantity", "unit_price", "amount"):
                raw = m.group(g)
                v = parse_num(raw, ns)
                if v is None:
                    v = parse_num(numeric_fix(raw), ns)
                    if v is not None and "line_items" not in res.ocr_fixed:
                        res.ocr_fixed.append("line_items")
                vals.append(v)
            if None in vals:
                res.parse_errors.append("line_items")
                return
            rec.line_items.append(LineItem(m.group("description").strip(), *vals))
            block_rows += 1
            last_was_row, gap = True, 0
            continue
        indented = ln[:1].isspace()
        if last_was_row and s and len(s) <= 60 and (indented or not re.search(r"\d", s)) \
                and len(re.findall(r"\d[\d.,]*", s)) <= 1:
            li = rec.line_items[-1]  # wrapped description: the rest of the previous item's text
            li.description = f"{li.description} {s}"
            last_was_row = False
            continue
        if block_rows and (not s or not last_was_row):
            in_block = False  # this block is over; a later page may start another one
            last_was_row = False
            continue
        if not s:
            continue
        gap += 1  # header rules and similar lines between the header and the first row
        if gap > max_gap:
            in_block = False


def describe(spec: dict) -> dict:
    """Small human-readable summary used by the dashboard and the trace."""
    return {
        "fields": sorted(spec.get("fields", {})),
        "required": spec.get("required", []),
        "num_style": spec.get("num_style"),
        "date_style": spec.get("date_style"),
        "items_row": spec.get("items", {}).get("row"),
    }
