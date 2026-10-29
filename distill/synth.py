"""Parser synthesis: induce candidate parser specs from labelled examples.

Labels come from the LLM tier (or from human-verified documents). The inducer
locates every labelled value in the text, generalises the surrounding line into
a pattern, and keeps only patterns that agree across *all* examples. Where the
evidence is ambiguous (e.g. 03/04/2026 could be DD/MM or MM/DD, 512.40 could be
"en" or "plain" numbers) it emits several candidates instead of guessing; the
validator decides between them on the examples.

Works for any schema (see doctypes.py):
  * money/date fields: located through every number/date rendering style;
  * raw/text fields: located whitespace-insensitively, value shape learned from examples;
  * fields whose value never changes for a source (vendor name, address) become
    `const` rules, which are only emitted when that text is still printed on the page;
  * when a value appears on several lines (TOTAL, CASH, ROUNDED TOTAL), lines whose
    label carries the field's hint words win.

Generators: induce_strict (exact label and glue), induce_relaxed (label words, loose
glue), and optionally LLM-written specs. All are validated identically.
"""
from __future__ import annotations

import itertools
import re
from collections import Counter
from difflib import SequenceMatcher
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

from .formats import DATE_FORMATS, NUM_REGEX, NUM_STYLES, fmt_date, fmt_num, fmt_qty, parse_num
from .model import Record, _norm_raw, diff
from .ocr import canon, ocr_equivalent
from .parser import NUM_REGEX_OCR, OCR_DIGIT, execute, value_pattern

# Business knowledge: these invoice fields are legitimately absent on many documents.
OPTIONAL_FIELDS = {"po_number", "discount"}


@dataclass
class Example:
    text: str
    labels: Record
    received_at: str
    doc_id: str = ""
    words: Optional[list] = None  # OCR words with positions, when the source provides them


# ------------------------------------------------------------- regex helpers
def lit(seg: str) -> str:
    """Literal text as a regex; whitespace may grow, shrink or vanish (OCR spacing)."""
    out = []
    for p in re.split(r"(\s+)", seg):
        if not p:
            continue
        out.append(r"\s*" if p.isspace() else re.escape(p))
    return "".join(out)


def masked(seg: str) -> str:
    """Like lit(), but digit runs match any digits: '8:13:39 PM' -> '\\d+:\\d+:\\d+ [A-Z]{2}'."""
    out = []
    for p in re.findall(r"\s+|\d+|[A-Za-z]{1,2}(?![A-Za-z])|[A-Za-z]+|.", seg):
        if p.isspace():
            out.append(r"\s*")
        elif p.isdigit():
            out.append(OCR_DIGIT + "+")
        elif p.isalpha() and len(p) <= 2:
            out.append("[A-Za-z]{1,2}")
        else:
            out.append(re.escape(p))
    return "".join(out)


def relaxed_label(prefix: str) -> str:
    """Keep the label words, relax the glue (separators, currency affixes)."""
    words = re.findall(r"[^\W\d_]+|\d+%?|[()%]", prefix)
    words = [w for w in words if not re.fullmatch(r"[A-Z]{3}", w)]  # drop currency codes
    if not words:
        return r"\s*"
    return r"\W*".join(re.escape(w) for w in words) + r"[\s:#.\-)]*(?:[A-Z]{2,3}|[$€£])?\s*"


def value_shape(v: str) -> str:
    """Regex for the *shape* of a printed value: 'TD01167104' -> '[A-Za-z]+\\d+'."""
    out = []
    for p in re.findall(r"\s+|\d+|[A-Za-z]+|.", v):
        if p.isspace():
            out.append(r"\s*")
        elif p.isdigit():
            out.append(OCR_DIGIT + "+")
        elif p.isalpha():
            out.append("[A-Za-z]+")
        else:
            out.append(re.escape(p))
    return "".join(out)


_LOOKALIKE = {}
for _group in ("0OoD", "1lIi|/", "5Ss", "8B", "6G", ",.", "-~_"):
    for _c in _group:
        _LOOKALIKE[_c] = "[" + re.escape(_group) + "]"


def _flex(v: str) -> re.Pattern:
    """Search for a labelled value, tolerant of spacing and of look-alike characters (OCR)."""
    chars = [_LOOKALIKE.get(c, re.escape(c)) for c in v if not c.isspace()]
    return re.compile(r"\s*".join(chars))


def _boundary_ok(line: str, a: int, b: int) -> bool:
    before = line[a - 1] if a > 0 else " "
    after = line[b] if b < len(line) else " "
    if before.isalnum() or after.isalnum():
        return False
    if before in ".," and a > 1 and line[a - 2].isdigit():
        return False
    if after in ".," and b + 1 < len(line) and line[b + 1].isdigit():
        return False
    return True


def _renderings(ftype: str, value) -> list[tuple[str, str]]:
    """(rendered string, style) pairs for a labelled value."""
    if value is None:
        return []
    if ftype == "money":
        return [(fmt_num(float(value), st), st) for st in NUM_STYLES]
    if ftype == "date":
        return [(fmt_date(value, st), st) for st in DATE_FORMATS]
    return [(str(value), "")]


# ------------------------------------------------------------- field evidence
def _has(ftype: str, v) -> bool:
    if ftype == "money":
        return v is not None and v == v and abs(v) > 1e-9
    return v is not None and str(v).strip() != ""


def _field_keys(fname: str, ftype: str, ex: Example) -> dict[tuple, set[str]]:
    """Map candidate key -> set of styles consistent with this example (cached per example)."""
    cache = ex.__dict__.setdefault("_cache", {})
    ck = ("keys", fname)
    if ck not in cache:
        cache[ck] = _field_keys_uncached(fname, ftype, ex)
    return cache[ck]


def _field_keys_uncached(fname: str, ftype: str, ex: Example) -> dict[tuple, set[str]]:
    value = getattr(ex.labels, fname)
    if not _has(ftype, value):
        return {}
    lines = [ln.strip() for ln in ex.text.splitlines() if ln.strip()]
    keys: dict[tuple, set[str]] = {}
    readable = legible(ex, fname)
    for rendered, style in _renderings(ftype, value):
        rx = _flex(rendered)
        for li, s in enumerate(lines):
            spans = [(m.start(), m.end()) for m in rx.finditer(s)]
            if not spans and not readable:
                spans = _fuzzy_spans(s, rendered)
            for a, b in spans:
                if not _boundary_ok(s, a, b):
                    continue
                prefix, suffix = s[:a], s[b:]
                if prefix.strip():
                    key = ("same", prefix, suffix if ftype in ("money", "date", "code") else masked(suffix), None)
                elif li > 0 and lines[li - 1].strip():
                    key = ("next", "", masked(suffix), lines[li - 1])
                else:
                    continue
                keys.setdefault(key, set()).add(style)
    return keys


def items_legible(ex: Example) -> bool:
    """Are all line items readable (description and numbers printed intact)? Scans damage rows too."""
    cache = ex.__dict__.setdefault("_cache", {})
    if "_items_legible" in cache:
        return cache["_items_legible"]
    flat = canon(ex.text)
    ok = True
    for li in ex.labels.line_items:
        if canon(li.description) not in flat:
            ok = False
            break
        for v, q in ((li.quantity, True), (li.unit_price, False), (li.amount, False)):
            if not any(canon(fmt_qty(v, st) if q else fmt_num(v, st)) in flat for st in NUM_STYLES):
                ok = False
                break
        if not ok:
            break
    cache["_items_legible"] = ok
    return ok


def legible(ex: Example, fname: str) -> bool:
    """Is the labelled value actually readable in this document's text? (Scans can destroy it.)"""
    ftype = ex.labels.schema.types[fname]
    value = getattr(ex.labels, fname)
    if not _has(ftype, value):
        return True  # an absent value is checked as absent
    key = ("_legible", fname)
    cache = ex.__dict__.setdefault("_cache", {})
    if key in cache:
        return cache[key]
    flat = canon(ex.text)
    ok = any(canon(r) in flat for r, _ in _renderings(ftype, value))
    cache[key] = ok
    return ok


def _fuzzy_spans(s: str, rendered: str) -> list[tuple[int, int]]:
    """Spans of `s` that look like `rendered` after OCR damage (same length +-1, >= 80% similar)."""
    if len(rendered) < 5:
        return []
    out = []
    target = canon(rendered)
    for m in re.finditer(r"\S+(?:\s\S+){0,%d}" % rendered.count(" "), s):
        tok = m.group()
        if abs(len(tok) - len(rendered)) <= 1 and SequenceMatcher(None, canon(tok), target).ratio() >= 0.8:
            out.append((m.start(), m.end()))
    return out


def _all_find(s: str, sub: str) -> list[int]:
    out, i = [], s.find(sub)
    while i >= 0:
        out.append(i)
        i = s.find(sub, i + 1)
    return out


def _hint_score(fname: str, key: tuple) -> int:
    from .doctypes import FIELD_HINTS
    label = (key[1] or key[3] or "").casefold()
    return sum(1 for h in FIELD_HINTS.get(fname, []) if h in label)


def _label_of(key: tuple) -> str:
    return key[1] if key[0] == "same" else key[3]


def _label_norm(label: str) -> str:
    """Comparable form of a label: digits masked, punctuation and spacing dropped, OCR look-alikes merged."""
    t = re.sub(r"\d+", "#", label)
    return canon(re.sub(r"[^\w#]+", "", t))


def _similar_label(a: str, b: str) -> bool:
    na, nb = _label_norm(a), _label_norm(b)
    if na == nb:
        return True
    if not na or not nb:
        return False
    return SequenceMatcher(None, na, nb).ratio() >= 0.75


def _consensus(fname: str, ftype: str, examples: list[Example]):
    """Candidate locations present in every example that has the field.

    Two examples agree on a location when they use the same mode and their labels are the
    same after masking digits and allowing OCR damage ('Date :' ~ 'Cate  ;'). The surviving
    key is the most common exact spelling; the executor's fuzzy fallback covers the others.
    """
    present = [ex for ex in examples if _has(ftype, getattr(ex.labels, fname))]
    if not present:
        return [], 0
    per_ex = [_field_keys(fname, ftype, ex) for ex in present]
    # an example whose value the scan destroyed says nothing about where the field lives
    per_ex = [k for k, ex in zip(per_ex, present) if k or legible(ex, fname)]
    if not per_ex or not all(per_ex) or len(per_ex) < max(2, len(present) - 1):
        return [], len(present)
    survivors = []
    for key, styles in per_ex[0].items():
        group, allowed = [key], set(styles)
        for other in per_ex[1:]:
            match = [(k, st) for k, st in other.items() if k[0] == key[0] and _similar_label(_label_of(k), _label_of(key))
                     and (ftype not in ("money", "date", "code") or k[2] == key[2])]
            if not match:
                group = None
                break
            group.append(match[0][0])
            allowed &= set().union(*(st for _, st in match))
        if group is None or not allowed:
            continue
        exact = Counter(group).most_common(1)[0][0]
        survivors.append((exact, allowed, len(set(group)) > 1))
    # prefer: a label that names the field, then a label OCR read identically every time, then same-line
    ranked = sorted(survivors, key=lambda kv: (_hint_score(fname, kv[0]) == 0, kv[2], -_hint_score(fname, kv[0]),
                                               kv[0][0] != "same", -len(_label_of(kv[0]) or "")))
    return [(k, st) for k, st, _ in ranked], len(present)


def _const_value(fname: str, ftype: str, examples: list[Example]) -> Optional[str]:
    """A raw/text field whose value is identical (OCR-insensitively) in every example: vendor name, address.

    Short values with digits (dates, numbers) are never treated as constant, even if the
    examples happen to agree.
    """
    if ftype not in ("raw", "text") or len(examples) < 2:
        return None
    vals = [getattr(ex.labels, fname) for ex in examples]
    if any(v is None or not str(v).strip() for v in vals):
        return None
    if len({_norm_raw(v) for v in vals}) != 1:
        return None
    v = str(vals[-1])
    if re.search(r"\d", v) and len(v) <= 15:
        return None
    return v


# ------------------------------------------------------------- items evidence
def _locate_description(lines: list[str], desc: str) -> Optional[tuple[int, str]]:
    """Index of the line holding the item description, and the part of it on that line (wrapped rows)."""
    for i, ln in enumerate(lines):
        if desc in ln:
            return i, desc
    for k in range(len(desc) - 1, 7, -1):  # wrapped: first part here, the rest on the next line
        head = desc[:k].rstrip()
        if not desc[k:].strip() or desc[k - 1] != " " and desc[k:k + 1] != " ":
            continue
        for i, ln in enumerate(lines[:-1]):
            if head in ln and desc[k:].strip() in lines[i + 1]:
                return i, head
    return None


def _row_template(ex: Example, style: str) -> Optional[tuple]:
    """Return (start_line, row_parts) for this example under a number style, or None."""
    lines = [ln.rstrip() for ln in ex.text.splitlines() if ln.strip()]
    templates, first_idx = set(), None
    numrx = re.compile(NUM_REGEX[style])
    for it in ex.labels.line_items:
        loc = _locate_description(lines, it.description)
        if loc is None:
            continue  # description damaged by the scan; the other rows carry the evidence
        idx, shown = loc
        first_idx = idx if first_idx is None else min(first_idx, idx)
        s = lines[idx].strip()
        d0 = s.find(shown)
        d1 = d0 + len(shown)
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
            out.append(f"(?P<{v}>{NUM_REGEX_OCR[style]})")
    return "".join(out)


# ------------------------------------------------------------- spec assembly
def _date_plausibility(spec: dict, examples: list[Example]) -> float:
    score = 0.0
    for ex in examples:
        r = execute(spec, ex.text, ex.labels.schema, ex.words).record
        if "invoice_date" not in r.values:
            return 0.0
        try:
            rec = datetime.fromisoformat(ex.received_at).date()
            if r.invoice_date:
                lag = (rec - datetime.fromisoformat(r.invoice_date).date()).days
                score += 1.0 if -2 <= lag <= 45 else 0.0
            if r.invoice_date and r.due_date:
                score += 1.0 if r.due_date >= r.invoice_date else 0.0
        except (ValueError, TypeError):
            pass
    return score


def induce(examples: list[Example], relaxed: bool = False, max_candidates: int = 6) -> list[dict]:
    if not examples:
        return []
    schema = examples[0].labels.schema
    field_rules: dict[str, tuple] = {}
    required: list[str] = []
    num_styles = set(NUM_STYLES)
    date_styles = set(DATE_FORMATS)
    for fs in schema.fields:
        fname, ftype = fs.name, fs.type
        const = _const_value(fname, ftype, examples)
        if const is not None:
            field_rules[fname] = ("const", const)
            if fname not in OPTIONAL_FIELDS:
                required.append(fname)
            continue
        opts, n_present = _consensus(fname, ftype, examples)
        if not opts:
            continue
        field_rules[fname] = ("line", opts[0])
        if n_present == len(examples) and fname not in OPTIONAL_FIELDS:
            required.append(fname)
        allowed = set(opts[0][1])
        if ftype == "money":
            num_styles &= allowed
        elif ftype == "date":
            date_styles &= allowed
    if not field_rules:
        return []

    num_choices = [s for s in ["en", "eu", "space", "plain"] if s in num_styles] or ["en"]
    row_by_style = {}
    if schema.items:
        for st in num_choices:
            rows = [_row_template(ex, st) for ex in examples]
            got = [r for r in rows if r]
            # every legible example must agree; damaged ones may be missing
            need = sum(1 for ex in examples if items_legible(ex))
            if got and len(set(got)) == 1 and len(got) >= max(1, need):
                row_by_style[st] = got[0]
        num_choices = [s for s in num_choices if s in row_by_style]
        if not num_choices:
            return []
    date_choices = [s for s in ["iso", "dmy_slash", "mdy_slash", "dmy_dot", "d_mon_y", "mon_d_y"] if s in date_styles] or ["iso"]
    if not any(f.type == "date" for f in schema.fields):
        date_choices = ["iso"]

    raw_shapes = {}
    for fs in schema.fields:
        if fs.type in ("raw",) and field_rules.get(fs.name, ("",))[0] == "line":
            shapes = sorted({value_shape(str(getattr(ex.labels, fs.name))) for ex in examples
                             if _has(fs.type, getattr(ex.labels, fs.name))})
            raw_shapes[fs.name] = "|".join(f"(?:{s})" for s in shapes) if len(shapes) <= 4 else ".+?"

    specs = []
    for ns, ds in itertools.product(num_choices, date_choices):
        fields = {}
        for fname, (kind, payload) in field_rules.items():
            ftype = schema.types[fname]
            if kind == "const":
                fields[fname] = {"mode": "const", "value": payload, "type": ftype}
                continue
            (mode, prefix, suffix, label), _ = payload
            vp = value_pattern(ftype, ns, ds, raw_shapes.get(fname))
            bare = vp[len("(?P<v>"):-1]
            suf_rx = suffix if ftype not in ("money", "date", "code") else lit(suffix)
            if relaxed:
                suf_rx = r"(?:\s.*)?" if ftype in ("raw", "text") else r"(?:\s*[A-Z]{2,3})?.*"
            if mode == "same":
                if relaxed:
                    pat = relaxed_label(prefix) + vp + suf_rx
                else:
                    pat = masked(prefix.rstrip()) + r"\s*" + vp + suf_rx
                fields[fname] = {"mode": "same", "pattern": pat, "type": ftype, "label_text": prefix.strip(),
                                 "value": bare}
            else:
                lab = relaxed_label(label) + ".*" if relaxed else masked(label)
                fields[fname] = {"mode": "next", "label": lab, "pattern": vp + suf_rx, "type": ftype,
                                 "label_text": label}
        spec = {
            "schema": schema.name,
            "num_style": ns,
            "date_style": ds,
            "fields": fields,
            "required": required,
            "generator": "induce_relaxed" if relaxed else "induce_strict",
        }
        if schema.items:
            start, parts = row_by_style[ns]
            spec["items"] = {"start": lit(start), "row": _row_regex(parts, ns), "max_gap": 2}
        specs.append(spec)
    if any(f.name == "invoice_date" for f in schema.fields):
        specs.sort(key=lambda s: -_date_plausibility(s, examples))
    return specs[:max_candidates]


def induce_zones(examples: list[Example]) -> list[dict]:
    """Registered geometric templates (see zones.py), for sources whose OCR carries word positions."""
    from .zones import learn_zones
    if not examples or any(ex.words is None for ex in examples):
        return []
    schema = examples[0].labels.schema
    line_specs = induce(examples)
    line_fields = line_specs[0]["fields"] if line_specs else {}
    fields, required = {}, []
    zone_fields = []
    for fs in schema.fields:
        const = _const_value(fs.name, fs.type, examples)
        if const is not None:
            fields[fs.name] = {"mode": "const", "value": const, "type": fs.type}
            required.append(fs.name)
        elif fs.type in ("raw", "text"):
            zone_fields.append(fs.name)
    registration, zrules = learn_zones([(ex.words, {f: getattr(ex.labels, f) for f in zone_fields}) for ex in examples],
                                       zone_fields)
    for f in zone_fields:
        present = sum(1 for ex in examples if _has(schema.types[f], getattr(ex.labels, f)))
        if f in zrules:
            fields[f] = {**zrules[f], "type": schema.types[f]}
        elif f in line_fields:
            fields[f] = line_fields[f]
        else:
            continue
        if present == len(examples) and f not in OPTIONAL_FIELDS:
            required.append(f)
    if not fields:
        return []
    spec = {"schema": schema.name, "num_style": "en", "date_style": "iso", "fields": fields, "required": required,
            "generator": "induce_zones"}
    if registration:
        spec["registration"] = registration
    return [spec]


def generate_candidates(examples: list[Example], llm_proposals: Iterable[dict] = ()) -> list[dict]:
    cands = induce_zones(examples) + induce(examples) + induce(examples, relaxed=True)
    for p in llm_proposals:
        p = dict(p)
        p.setdefault("generator", "llm")
        cands.append(p)
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
    """Run a candidate on the examples. Only values legible in each example's text are compared:
    a parser cannot be blamed for a total the scanner destroyed."""
    exact, errors = 0, {}
    for ex in examples:
        names = [f.name for f in ex.labels.schema.fields if legible(ex, f.name)]
        try:
            r = execute(spec, ex.text, ex.labels.schema, ex.words)
            bad = diff(r.record, ex.labels, fields=names)
            # where the scan swapped look-alike characters, reading what is printed is correct for a parser
            bad = [f for f in bad if not (ex.labels.schema.types.get(f) in ("text", "raw")
                                          and ocr_equivalent(r.record.values.get(f), getattr(ex.labels, f), allow_drop=False))]
            check_items = ex.labels.schema.items and items_legible(ex)
            if check_items:
                bad += [b for b in diff(r.record, ex.labels) if b.startswith("line_items")]
            blocking = [f for f in r.missing_required + r.parse_errors if f in names or (f == "line_items" and check_items)]
            if blocking or (check_items and not r.items_found):
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
