"""Vendor identification and layout fingerprints.

A layout fingerprint is the set of *line shapes* of a document: whitespace
collapsed and digits masked, with value-dense rows (line items, phone numbers)
left out. Shapes are stable across invoices from the same template and change
when the template changes, which makes them a cheap, parser-independent drift
signal.
"""
from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Optional

_NUM_TOKEN = re.compile(r"\d[\d.,]*")
_MONTH = re.compile(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
                    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b")


def shape(line: str) -> Optional[str]:
    s = " ".join(line.split())
    if not s:
        return None
    if len(_NUM_TOKEN.findall(s)) >= 3:
        return None  # item rows and other value-dense lines vary per document
    s = re.sub(r"([-=_*.|])\1{2,}", r"\1\1\1", s)  # separator rules vary in length
    s = _MONTH.sub("Mon", s)  # dates written with month names
    s = re.sub(r"\d", "9", s)
    s = re.sub(r"9+", "9", s)
    return s


def _near_any(s: str, pool: list[str], threshold: float = 0.85) -> bool:
    for k in pool:
        if abs(len(k) - len(s)) > max(3, len(s) // 5):
            continue
        sm = SequenceMatcher(None, s, k)
        if sm.real_quick_ratio() >= threshold and sm.quick_ratio() >= threshold and sm.ratio() >= threshold:
            return True
    return False


def structural(shape_line: str) -> bool:
    """A line that looks like a label with a value ('Promo discount GBP 9.9', 'Order Ref: 9'),
    as opposed to prose notes or a damaged fragment ('9/9')."""
    return ":" in shape_line or bool(re.search(r"[A-Za-z]{3,}.*9", shape_line))


def doc_shapes(text: str) -> set[str]:
    return {sh for ln in text.splitlines() if (sh := shape(ln))}


def identify_vendor(text: str, vendor_master: list[dict]) -> Optional[str]:
    """Match the vendor by its registered tax ID, as an AP system would.

    Scans damage IDs too, so an exact match is tried first, then an OCR-tolerant one
    (look-alike characters, one dropped character), then the vendor's printed name.
    """
    for v in vendor_master:
        if v["tax_id"] in text:
            return v["vendor_id"]
    from .ocr import canon, ocr_equivalent
    tokens = re.findall(r"[\w\-/]{6,}", text)
    for v in vendor_master:
        tid = v["tax_id"]
        for t in tokens:
            if abs(len(t) - len(tid)) <= 1 and ocr_equivalent(t, tid):
                return v["vendor_id"]
    head = canon(" ".join(text.splitlines()[:3]))
    for v in vendor_master:
        name = canon(v["name"])
        if name and (name in head or SequenceMatcher(None, name, head[:len(name) + 4]).ratio() >= 0.85):
            return v["vendor_id"]
    return None


class LayoutModel:
    """Line shapes seen in accepted documents of one vendor, per layout epoch."""

    def __init__(self):
        self.counts: Counter = Counter()
        self.n_docs = 0

    def novelty(self, shapes: set[str]) -> list[str]:
        """Lines never seen before, ignoring lines that are just OCR-damaged copies of known ones."""
        if self.n_docs == 0:
            return []
        known = list(self.counts)
        return sorted(s for s in shapes if self.counts[s] == 0 and not _near_any(s, known))

    def missing_core(self, shapes: set[str], min_share: float = 0.95) -> list[str]:
        """Shapes present in almost every past document but absent now (e.g. a renamed label)."""
        if self.n_docs < 3:
            return []
        present = list(shapes)
        return sorted(s for s, c in self.counts.items()
                      if c / self.n_docs >= min_share and s not in shapes and not _near_any(s, present, 0.7))

    def learn(self, shapes: set[str]) -> None:
        self.counts.update(shapes)
        self.n_docs += 1

    def reset(self) -> None:
        self.counts.clear()
        self.n_docs = 0


def _header_tokens(text: str, n_lines: int = 15) -> set[str]:
    lines = [ln for ln in text.splitlines() if ln.strip()][:n_lines]
    return {w.upper() for ln in lines for w in re.findall(r"[A-Za-z]{3,}", ln)}


class LayoutClusterer:
    """Group documents into sources (vendors/merchants) without labels.

    Documents from one source share their letterhead: the words in the first lines.
    A document joins the most similar existing cluster (Jaccard similarity of those
    words) if it clears `threshold`, otherwise it starts a new cluster.
    """

    def __init__(self, threshold: float = 0.5, prefix: str = "C"):
        self.threshold = threshold
        self.prefix = prefix
        self.clusters: list[tuple[str, set[str]]] = []

    def assign(self, text: str) -> str:
        toks = _header_tokens(text)
        best, best_sim = None, 0.0
        for cid, ref in self.clusters:
            if not toks or not ref:
                continue
            sim = len(toks & ref) / len(toks | ref)
            if sim > best_sim:
                best, best_sim = cid, sim
        if best is not None and best_sim >= self.threshold:
            return best
        cid = f"{self.prefix}{len(self.clusters) + 1:03d}"
        self.clusters.append((cid, toks))
        return cid
