"""Geometric templates: where on the page a field lives.

On a known layout a field occupies a predictable region: fixed on the page (the
invoice number box), or at a fixed offset from a label word (the amount to the right
of "TOTAL", wherever the table above pushes it). This is how template-based
extraction has always worked ("zonal OCR"); Distill learns the zones itself from a
few labelled examples instead of someone drawing them.

Words are [page, x0, y0, x1, y1, text] with coordinates normalised to 0..1.

Learning a field from K examples:
  1. find the OCR words that spell the labelled value in each example;
  2. look for an anchor: a label-like word present in every example at a consistent
     offset from the value;
  3. the zone is the union of the value boxes (relative to the anchor, or to the page)
     plus a small margin;
  4. keep the zone only if reading it back reproduces every example.

Spec rules (stored inside an ordinary parser spec, so versioning, shadowing and
audits work unchanged):
  {"mode": "zone", "page": 0, "anchor": "total", "anchor_xy": [x, y], "box": [dx0, dy0, dx1, dy1]}
  {"mode": "zone", "page": 0, "anchor": null, "box": [x0, y0, x1, y1]}
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

Word = list  # [page, x0, y0, x1, y1, text]

MARGIN_X, MARGIN_Y = 0.012, 0.004


_NONWORD = re.compile(r"[\W_]+")


@lru_cache(maxsize=1_000_000)
def _norm_str(s: str) -> str:
    return _NONWORD.sub("", s).casefold()


def norm(s) -> str:
    return _norm_str(s if isinstance(s, str) else str(s))


def reading_order(words: list[Word]) -> list[Word]:
    if not words:
        return []
    hs = sorted(w[4] - w[2] for w in words)
    med = hs[len(hs) // 2] or 0.01
    rows, out = [], []
    for w in sorted(words, key=lambda w: (w[0], (w[2] + w[4]) / 2, w[1])):
        cy = (w[2] + w[4]) / 2
        if rows and rows[-1][0] == w[0] and abs(rows[-1][1] - cy) <= med * 0.55:
            rows[-1][2].append(w)
        else:
            rows.append([w[0], cy, [w]])
    for _, _, ws in rows:
        out.extend(sorted(ws, key=lambda w: w[1]))
    return out


_ORDER_CACHE: dict = {}


def _ordered(words: list[Word]) -> list[Word]:
    k = id(words)
    hit = _ORDER_CACHE.get(k)
    if hit is not None and hit[0] is words:
        return hit[1]
    if len(_ORDER_CACHE) > 4096:
        _ORDER_CACHE.clear()
    o = reading_order(words)
    _ORDER_CACHE[k] = (words, o)
    return o


def locate_all(words: list[Word], value: str, max_rows: int = 6, limit: int = 8) -> list[list[Word]]:
    """Every run of OCR words that spells `value` (letters and digits only), other words allowed in between."""
    target = norm(value)
    if not target:
        return []
    ordered = _ordered(words)
    found = []
    for i, w in enumerate(ordered):
        wn = norm(w[5])
        if not wn or not target.startswith(wn):
            continue
        got, rest, rows_used, last_y = [w], target[len(wn):], 0, w[2]
        j = i + 1
        while rest and j < len(ordered):
            nw = ordered[j]
            if nw[0] != w[0]:
                break
            if nw[2] > last_y + 0.002:
                rows_used += 1
                last_y = nw[2]
                if rows_used > max_rows:
                    break
            n = norm(nw[5])
            if n and rest.startswith(n):
                # only accept words that sit near the value (same line, or same column)
                if nw[2] - got[-1][4] < 0.03 or abs(nw[1] - got[0][1]) < 0.15:
                    got.append(nw)
                    rest = rest[len(n):]
            j += 1
        if not rest:
            found.append(got)
            if len(found) >= limit:
                break
    return found


def locate(words: list[Word], value: str, max_rows: int = 6) -> Optional[list[Word]]:
    hits = locate_all(words, value, max_rows, limit=1)
    return hits[0] if hits else None


# ---------------------------------------------------------------- registration
def signature(words: list[Word], page: int = 0) -> dict[str, tuple[float, float]]:
    """Words that occur exactly once on the page (>= 4 letters): stable landmarks for alignment."""
    seen: dict[str, list] = {}
    for w in words:
        if w[0] != page:
            continue
        t = norm(w[5])
        if len(t) >= 4 and re.search(r"[a-z]{4}", t):
            seen.setdefault(t, []).append(((w[1] + w[3]) / 2, (w[2] + w[4]) / 2))
    return {t: v[0] for t, v in seen.items() if len(v) == 1}


def _lsq(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    if sxx < 1e-6:
        return 1.0, my - mx
    a = sum((p[0] - mx) * (p[1] - my) for p in pairs) / sxx
    if not 0.8 <= a <= 1.25:  # page scale barely changes between scans of one layout
        a = 1.0
    return a, my - a * mx


def fit_transform(ref: dict, doc: dict) -> Optional[tuple[float, float, float, float]]:
    """Map document coordinates onto the reference page: x' = ax*x + bx, y' = ay*y + by."""
    common = [t for t in doc if t in ref]
    if len(common) < 4:
        return None
    pts = [(doc[t], ref[t]) for t in common]
    for _ in range(3):
        ax, bx = _lsq([(d[0], r[0]) for d, r in pts])
        ay, by = _lsq([(d[1], r[1]) for d, r in pts])
        res = [abs(ax * d[0] + bx - r[0]) + abs(ay * d[1] + by - r[1]) for d, r in pts]
        keep = [p for p, e in zip(pts, res) if e <= max(0.015, 2.5 * sorted(res)[len(res) // 2])]
        if len(keep) == len(pts) or len(keep) < 4:
            break
        pts = keep
    return ax, bx, ay, by


def apply_transform(words: list[Word], T) -> list[Word]:
    if T is None:
        return words
    ax, bx, ay, by = T
    return [[w[0], ax * w[1] + bx, ay * w[2] + by, ax * w[3] + bx, ay * w[4] + by, w[5]] for w in words]


def register(words: list[Word], registration: Optional[list]) -> tuple[list[Word], bool]:
    """Align a document to the template's reference page. Returns (aligned words, aligned?)."""
    if not registration:
        return words, True
    ref = {t: (x, y) for t, x, y in registration}
    T = fit_transform(ref, signature(words))
    return apply_transform(words, T), T is not None


def bbox(ws: list[Word]) -> list[float]:
    return [min(w[1] for w in ws), min(w[2] for w in ws), max(w[3] for w in ws), max(w[4] for w in ws)]


def read_zone(words: list[Word], page: int, box: list[float]) -> Optional[str]:
    x0, y0, x1, y1 = box
    inside = [w for w in words if w[0] == page and x0 <= (w[1] + w[3]) / 2 <= x1 and y0 <= (w[2] + w[4]) / 2 <= y1]
    if not inside:
        return None
    return " ".join(w[5] for w in reading_order(inside))


def find_anchor(words: list[Word], token: str, near: Optional[list[float]], page: int) -> Optional[Word]:
    cands = [w for w in words if w[0] == page and norm(w[5]) == token]
    if not cands:
        return None
    if near is None:
        return cands[0]
    return min(cands, key=lambda w: (w[1] - near[0]) ** 2 + (w[2] - near[1]) ** 2)


def execute_zone(rule: dict, words: list[Word]) -> Optional[str]:
    page = rule.get("page", 0)
    box = rule["box"]
    if rule.get("anchor"):
        a = find_anchor(words, rule["anchor"], rule.get("anchor_xy"), page)
        if a is None:
            return None
        box = [a[1] + box[0], a[2] + box[1], a[1] + box[2], a[2] + box[3]]
    return read_zone(words, page, box)


# ---------------------------------------------------------------- learning
def _anchor_candidates(words: list[Word], value_ws: list[Word]) -> dict[str, Word]:
    """Label-like words on the value's page, outside the value, keyed by normalised text."""
    vb = bbox(value_ws)
    page = value_ws[0][0]
    ids = {id(w) for w in value_ws}
    out = {}
    for w in words:
        if w[0] != page or id(w) in ids:
            continue
        t = norm(w[5])
        if len(t) < 3 or not re.search(r"[a-z]{3}", t) or re.search(r"\d", t):
            continue
        # only nearby words make useful anchors: left of or above the value, within reach
        dx, dy = vb[0] - w[1], vb[1] - w[2]
        if -0.05 <= dy <= 0.12 and -0.3 <= dx <= 0.6:
            if t not in out or abs(dx) + abs(dy) < abs(vb[0] - out[t][1]) + abs(vb[1] - out[t][2]):
                out[t] = w
    return out


def _center(ws: list[Word]) -> tuple[float, float]:
    b = bbox(ws)
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _consistent_occurrences(per_example: list[list[list[Word]]]) -> Optional[list[Optional[list[Word]]]]:
    """Pick one occurrence per example so they all sit in the same place (one example may be missing)."""
    best, best_spread = None, None
    base_idx = next((k for k, occ in enumerate(per_example) if occ), None)
    if base_idx is None:
        return None
    for cand in per_example[base_idx]:
        cx, cy = _center(cand)
        chosen, spread = [], 0.0
        for occ in per_example:
            if not occ:
                chosen.append(None)
                continue
            o = min(occ, key=lambda ws: abs(_center(ws)[0] - cx) + abs(_center(ws)[1] - cy))
            ox, oy = _center(o)
            spread = max(spread, abs(oy - cy) + 0.3 * abs(ox - cx))
            chosen.append(o)
        if best_spread is None or spread < best_spread:
            best, best_spread = chosen, spread
    return best


def learn_zones(examples: list[tuple[list[Word], dict]], fields: list[str]) -> tuple[Optional[list], dict]:
    """Learn a registered template from (words, {field: value}) examples.

    Returns (registration landmarks, {field: zone rule}). Pages are aligned to the first
    example before positions are compared, so a layout scanned with a shifted page still
    yields one zone per field.
    """
    if len(examples) < 2:
        return None, {}
    ref_sig = signature(examples[0][0])
    registration = [[t, round(x, 4), round(y, 4)] for t, (x, y) in sorted(ref_sig.items())[:120]]
    aligned = []
    for words, values in examples:
        T = fit_transform(ref_sig, signature(words))
        aligned.append((apply_transform(words, T), values, T is not None))
    if sum(1 for _, _, ok in aligned if ok) < len(aligned) - 1:
        registration = None  # pages could not be aligned: fall back to raw coordinates
        aligned = [(w, v, True) for w, v in examples]
    rules = {}
    for f in fields:
        vals = [values.get(f) for _, values, _ in aligned]
        present = [(w, v) for (w, _, _), v in zip(aligned, vals) if v]
        if len(present) < max(2, len(aligned) - 1):
            continue
        occs = [locate_all(w, v) for w, v in present]
        missing = sum(1 for o in occs if not o)
        if missing > (1 if len(present) >= 3 else 0):
            continue
        chosen = _consistent_occurrences(occs)
        if not chosen:
            continue
        pairs = [(w, ws) for (w, _), ws in zip(present, chosen) if ws]
        pages = {ws[0][0] for _, ws in pairs}
        if len(pages) != 1:
            continue
        page = pages.pop()
        boxes = [bbox(ws) for _, ws in pairs]
        cands = []
        if max(b[1] for b in boxes) - min(b[1] for b in boxes) < 0.02:
            cands.append({"mode": "zone", "page": page, "anchor": None,
                          "box": [round(min(b[0] for b in boxes) - MARGIN_X, 4), round(min(b[1] for b in boxes) - MARGIN_Y, 4),
                                  round(max(b[2] for b in boxes) + MARGIN_X, 4), round(max(b[3] for b in boxes) + MARGIN_Y, 4)]})
        cands += _anchored(pairs, page)
        checks = [(w, v) for (w, v), ws in zip(present, chosen) if ws]
        for r in cands:
            if all(norm(execute_zone(r, w) or "") == norm(v) for w, v in checks):
                rules[f] = r
                break
    return registration, rules


def _anchored(pairs: list[tuple[list[Word], list[Word]]], page: int) -> list[dict]:
    per = [_anchor_candidates(words, ws) for words, ws in pairs]
    common = set(per[0]).intersection(*per[1:]) if per else set()
    scored = []
    for t in common:
        offs = []
        for (words, ws), cand in zip(pairs, per):
            a, vb = cand[t], bbox(ws)
            offs.append([vb[0] - a[1], vb[1] - a[2], vb[2] - a[1], vb[3] - a[2]])
        spread = max(o[1] for o in offs) - min(o[1] for o in offs) + (max(o[0] for o in offs) - min(o[0] for o in offs)) * 0.3
        if spread > 0.02:
            continue
        dist = sum(abs(o[0]) + abs(o[1]) for o in offs) / len(offs)
        scored.append((dist + spread * 3, t, offs, [per[k][t] for k in range(len(per))]))
    out = []
    for _, t, offs, anchors in sorted(scored)[:2]:
        box = [min(o[0] for o in offs) - MARGIN_X, min(o[1] for o in offs) - MARGIN_Y,
               max(o[2] for o in offs) + MARGIN_X, max(o[3] for o in offs) + MARGIN_Y]
        axy = [sum(a[1] for a in anchors) / len(anchors), sum(a[2] for a in anchors) / len(anchors)]
        out.append({"mode": "zone", "page": page, "anchor": t, "anchor_xy": [round(axy[0], 4), round(axy[1], 4)],
                    "box": [round(b, 4) for b in box]})
    return out


def learn_zone(examples: list[tuple[list[Word], str]]) -> list[dict]:
    """Candidate zone rules for one field. `examples` are (words, labelled value) pairs."""
    located = []
    for words, value in examples:
        ws = locate(words, value)
        if ws is None:
            return []
        located.append((words, ws))
    pages = {ws[0][0] for _, ws in located}
    if len(pages) != 1:
        return []
    page = pages.pop()
    rules = []
    # anchored: a label word every example shares, at a consistent offset
    per = [_anchor_candidates(words, ws) for words, ws in located]
    common = set(per[0]).intersection(*per[1:]) if per else set()
    scored = []
    for t in common:
        offs = []
        for (words, ws), cand in zip(located, per):
            a, vb = cand[t], bbox(ws)
            offs.append([vb[0] - a[1], vb[1] - a[2], vb[2] - a[1], vb[3] - a[2]])
        spread = max(o[1] for o in offs) - min(o[1] for o in offs) + (max(o[0] for o in offs) - min(o[0] for o in offs)) * 0.3
        if spread > 0.02:
            continue
        dist = sum(abs(o[0]) + abs(o[1]) for o in offs) / len(offs)
        scored.append((dist + spread * 3, t, offs, [per[k][t] for k in range(len(per))]))
    for _, t, offs, anchors in sorted(scored)[:2]:
        box = [min(o[0] for o in offs) - MARGIN_X, min(o[1] for o in offs) - MARGIN_Y,
               max(o[2] for o in offs) + MARGIN_X, max(o[3] for o in offs) + MARGIN_Y]
        axy = [sum(a[1] for a in anchors) / len(anchors), sum(a[2] for a in anchors) / len(anchors)]
        rules.append({"mode": "zone", "page": page, "anchor": t, "anchor_xy": [round(axy[0], 4), round(axy[1], 4)],
                      "box": [round(b, 4) for b in box]})
    # absolute: the value sits in the same place on every page
    boxes = [bbox(ws) for _, ws in located]
    if max(b[1] for b in boxes) - min(b[1] for b in boxes) < 0.02:
        box = [min(b[0] for b in boxes) - MARGIN_X, min(b[1] for b in boxes) - MARGIN_Y,
               max(b[2] for b in boxes) + MARGIN_X, max(b[3] for b in boxes) + MARGIN_Y]
        rules.append({"mode": "zone", "page": page, "anchor": None, "box": [round(b, 4) for b in box]})
    # keep only rules that read every example back exactly
    good = []
    for r in rules:
        if all(norm(execute_zone(r, words) or "") == norm(v) for words, v in examples):
            good.append(r)
    return good
