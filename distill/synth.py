"""Parser synthesis: induce candidate parser specs from labelled examples.

Labels come from the LLM tier (or from human-verified documents). The inducer
locates every labelled value in the text, generalises the surrounding line into
a pattern, and keeps only patterns that agree across *all* examples. Where the
evidence is ambiguous (e.g. 03/04/2026 could be DD/MM or MM/DD, 512.40 could be
"en" or "plain" numbers) it emits several candidates instead of guessing; the
validator decides between them on held-out evidence.

Generators
----------
- induce_strict  : exact label + separator + currency affixes
- induce_relaxed : label words only, tolerant of separators/currency affixes
- (optional) LLM : the model writes a spec in the same JSON DSL (see llm.py)
All candidates are validated identically, so no generator is trusted blindly.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

from .formats import DATE_FORMATS, NUM_REGEX, NUM_STYLES, fmt_date, fmt_num, fmt_qty, parse_num
from .model import HEADER_FIELDS, Invoice, diff
from .parser import FIELD_TYPES, execute, value_pattern


# Business knowledge: these fields are legitimately absent on many invoices.
OPTIONAL_FIELDS = {"po_number", "discount"}


@dataclass
class Example:
    text: str
    labels: Invoice
    received_at: str
    doc_id: str = ""


# ------------------------------------------------------------- regex helpers
def lit(seg: str) -> str:
    out = []
    for p in re.split(r"(\s+)", seg):
        if not p:
            continue
        out.append(r"\s+" if p.isspace() else re.escape(p))
    return "".join(out)


def relaxed_label(prefix: str) -> str:
    """Keep the label words, relax the glue (separators, currency affixes)."""
    words = re.findall(r"[^\W\d_]+|\d+%?|[()%]", prefix)
    words = [w for w in words if not re.fullmatch(r"[A-Z]{3}", w)]  # drop currency codes
    if not words:
        return r"\s*"
    return r"\W*".join(re.escape(w) for w in words) + r"[\s:#.\-)]*(?:[A-Z]{3}|[$€£])?\s*"


def _boundary_find(line: str, s: str) -> list[int]:
    out, start = [], 0
    while True:
        i = line.find(s, start)
        if i < 0:
            return out
        before = line[i - 1] if i > 0 else " "
        after = line[i + len(s)] if i + len(s) < len(line) else " "
        if not (before.isalnum() or before in "/-") and not (after.isalnum() or after in "/-"):
            # reject when the match is the middle of a longer number such as 1,234.56
            if not (before in ".," and i > 1 and line[i - 2].isdigit()) and not (after in ".," and i + len(s) + 1 < len(line) and line[i + len(s) + 1].isdigit()):
                out.append(i)
        start = i + 1


def _renderings(fname: str, value) -> list[tuple[str, str]]:
    """(rendered string, style) pairs for a labelled value."""
    ftype = FIELD_TYPES[fname]
    if value is None:
        return []
    if ftype == "money":
        return [(fmt_num(float(value), st), st) for st in NUM_STYLES]
    if ftype == "date":
        return [(fmt_date(value, st), st) for st in DATE_FORMATS]
    return [(str(value), "")]


# ------------------------------------------------------------- field evidence
def _field_keys(fname: str, ex: Example) -> dict[tuple, set[str]]:
    """Map candidate key -> set of styles consistent with this example."""
    value = getattr(ex.labels, fname)
    if FIELD_TYPES[fname] == "money" and (value is None or abs(value) < 1e-9):
        return {}
    lines = [ln.rstrip() for ln in ex.text.splitlines() if ln.strip()]
    keys: dict[tuple, set[str]] = {}
    for rendered, style in _renderings(fname, value):
        for li, ln in enumerate(lines):
            s = ln.strip()
            for pos in _boundary_find(s, rendered):
                prefix, suffix = s[:pos], s[pos + len(rendered):]
                if prefix.strip():
                    key = ("same", prefix, suffix, None)
                elif li > 0 and not suffix.strip():
                    key = ("next", "", "", lines[li - 1].strip())
                else:
                    continue
                keys.setdefault(key, set()).add(style)
    return keys


def _consensus(fname: str, examples: list[Example]):
    """Keys present in every example that has the field, with the styles they allow."""
    present = [ex for ex in examples if _has(fname, ex.labels)]
    if not present:
        return [], 0
    agg: Optional[dict] = None
    for ex in present:
        k = _field_keys(fname, ex)
        if agg is None:
            agg = k
        else:
            agg = {key: agg[key] & styles for key, styles in k.items() if key in agg and agg[key] & styles}
        if not agg:
            return [], len(present)
    return sorted(agg.items(), key=lambda kv: (kv[0][0] != "same", -len(kv[0][1] or kv[0][3] or ""))), len(present)


def _has(fname: str, inv: Invoice) -> bool:
    v = getattr(inv, fname)
    if FIELD_TYPES[fname] == "money":
        return v is not None and abs(v) > 1e-9
    return v is not None


# ------------------------------------------------------------- items evidence
def _row_template(ex: Example, style: str) -> Optional[tuple]:
    """Return (start_line, row_parts) for this example under a number style, or None."""
    lines = [ln.rstrip() for ln in ex.text.splitlines() if ln.strip()]
    templates, first_idx = set(), None
    numrx = re.compile(NUM_REGEX[style])
    for it in ex.labels.line_items:
        idx = next((i for i, ln in enumerate(lines) if it.description in ln), None)
        if idx is None:
            return None
        first_idx = idx if first_idx is None else min(first_idx, idx)
        s = lines[idx].strip()
        d0 = s.find(it.description)
        d1 = d0 + len(it.description)
        tail = s[d1:]
        toks = [(m.start() + d1, m.end() + d1, m.group()) for m in numrx.finditer(tail)]
        want = {"quantity": it.quantity, "unit_price": it.unit_price, "amount": it.amount}
        assign = {}
        for name, val in want.items():
            hits = [t for t in toks if (pv := parse_num(t[2], style)) is not None and abs(pv - val) < 0.0051
                    and t[2] in (fmt_num(val, style), fmt_qty(val, style))]
            hits = [t for t in hits if t not in assign.values()]
            if len(hits) != 1:
                break
            assign[name] = hits[0]
        if len(assign) != 3:
            continue  # ambiguous line (e.g. qty == price); other lines carry the evidence
        order = sorted(assign.items(), key=lambda kv: kv[1][0])
        norm = lambda seg: re.sub(r"\s+", " ", seg)
        parts = [("lit", norm(s[:d0])), ("grp", "description")]
        cur = d1
        for name, (a, b, _) in order:
            parts.append(("lit", norm(s[cur:a])))
            parts.append(("grp", name))
            cur = b
        parts.append(("lit", norm(s[cur:])))
        templates.add(tuple(parts))
    if len(templates) != 1 or first_idx is None:
        return None
    start = None
    for j in range(first_idx - 1, -1, -1):
        cand = lines[j].strip()
        if re.search(r"[A-Za-z]{3}", cand) and not re.search(r"\d", cand):
            start = re.sub(r"\s+", " ", cand)
            break
    if start is None:
        return None
    return start, next(iter(templates))


def _row_regex(parts: tuple, style: str) -> str:
    out = []
    for kind, v in parts:
        if kind == "lit":
            if not v:
                continue
            out.append(r"\s+" if v.isspace() else (r"\s*" + lit(v.strip()) + r"\s*"))
        elif v == "description":
            out.append(r"(?P<description>.+?)")
        else:
            out.append(f"(?P<{v}>{NUM_REGEX[style]})")
    return "".join(out)


# ------------------------------------------------------------- spec assembly
def _date_plausibility(spec: dict, examples: list[Example]) -> float:
    score = 0.0
    for ex in examples:
        r = execute(spec, ex.text).invoice
        try:
            rec = datetime.fromisoformat(ex.received_at).date()
            if r.invoice_date:
                lag = (rec - datetime.fromisoformat(r.invoice_date).date()).days
                score += 1.0 if -2 <= lag <= 45 else 0.0
            if r.invoice_date and r.due_date:
                score += 1.0 if r.due_date >= r.invoice_date else 0.0
        except ValueError:
            pass
    return score


def induce(examples: list[Example], relaxed: bool = False, max_candidates: int = 6) -> list[dict]:
    if not examples:
        return []
    field_opts: dict[str, list] = {}
    required: list[str] = []
    num_styles = set(NUM_STYLES)
    date_styles = set(DATE_FORMATS)
    for fname in HEADER_FIELDS:
        opts, n_present = _consensus(fname, examples)
        if not opts:
            continue
        field_opts[fname] = opts[:2]
        if n_present == len(examples) and fname not in OPTIONAL_FIELDS:
            required.append(fname)
        ftype = FIELD_TYPES[fname]
        allowed = set().union(*(styles for _, styles in opts[:1]))
        if ftype == "money":
            num_styles &= allowed
        elif ftype == "date":
            date_styles &= allowed

    # items: the row template must agree across every example
    row_by_style = {}
    for st in sorted(num_styles):
        rows = [_row_template(ex, st) for ex in examples]
        if all(rows) and len(set(rows)) == 1:
            row_by_style[st] = rows[0]
    num_choices = [s for s in ["en", "eu", "space", "plain"] if s in row_by_style]
    if not num_choices:
        return []
    date_choices = [s for s in ["iso", "dmy_slash", "mdy_slash", "dmy_dot", "d_mon_y", "mon_d_y"] if s in date_styles] or ["iso"]

    specs = []
    for ns, ds in itertools.product(num_choices, date_choices):
        fields = {}
        for fname, opts in field_opts.items():
            (mode, prefix, suffix, label), _ = opts[0]
            vp = value_pattern(FIELD_TYPES[fname], ns, ds)
            if mode == "same":
                if relaxed:
                    pat = relaxed_label(prefix) + vp + r"(?:\s*[A-Z]{3})?"
                else:
                    pat = lit(prefix.rstrip()) + (r"\s*" if prefix.endswith(" ") or not prefix[-1:].isalnum() else "") + vp + lit(suffix)
                fields[fname] = {"mode": "same", "pattern": pat, "type": FIELD_TYPES[fname]}
            else:
                lab = relaxed_label(label) if relaxed else lit(label)
                fields[fname] = {"mode": "next", "label": lab, "pattern": vp, "type": FIELD_TYPES[fname]}
        start, parts = row_by_style[ns]
        spec = {
            "num_style": ns,
            "date_style": ds,
            "fields": fields,
            "required": required,
            "items": {"start": lit(start), "row": _row_regex(parts, ns), "max_gap": 2},
            "generator": "induce_relaxed" if relaxed else "induce_strict",
        }
        specs.append(spec)
    specs.sort(key=lambda s: -_date_plausibility(s, examples))
    return specs[:max_candidates]


def generate_candidates(examples: list[Example], llm_proposals: Iterable[dict] = ()) -> list[dict]:
    cands = induce(examples) + induce(examples, relaxed=True)
    for p in llm_proposals:
        p = dict(p)
        p.setdefault("generator", "llm")
        cands.append(p)
    # de-duplicate on the parts that change behaviour
    seen, out = set(), []
    for c in cands:
        key = repr((c.get("num_style"), c.get("date_style"), sorted(c.get("fields", {}).items()), c.get("items")))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


# ------------------------------------------------------------- isolated validation
@dataclass
class Validation:
    n: int
    exact: int
    errors: dict

    @property
    def accuracy(self) -> float:
        return self.exact / self.n if self.n else 0.0


def validate(spec: dict, examples: list[Example]) -> Validation:
    exact, errors = 0, {}
    for ex in examples:
        try:
            r = execute(spec, ex.text)
            bad = diff(r.invoice, ex.labels)
            if r.missing_required or r.parse_errors or not r.items_found:
                bad = bad or ["anchors"]
        except Exception as e:  # a malformed (e.g. LLM-written) spec must never crash the pipeline
            bad = [f"error:{type(e).__name__}"]
        if not bad:
            exact += 1
        else:
            errors[ex.doc_id or str(len(errors))] = bad
    return Validation(n=len(examples), exact=exact, errors=errors)


def strictness(spec: dict) -> int:
    return sum(len(r.get("pattern", "")) + len(r.get("label", "")) for r in spec.get("fields", {}).values()) + (
        0 if spec.get("generator") == "induce_relaxed" else 10_000)


def best_candidate(cands: list[dict], examples: list[Example]) -> tuple[Optional[dict], list[tuple[dict, Validation]]]:
    scored = [(c, validate(c, examples)) for c in cands]
    scored.sort(key=lambda cv: (-cv[1].accuracy, -strictness(cv[0])))
    best = scored[0][0] if scored and scored[0][1].accuracy == 1.0 else None
    return best, scored
