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
from typing import Optional

_NUM_TOKEN = re.compile(r"\d[\d.,]*")


def shape(line: str) -> Optional[str]:
    s = " ".join(line.split())
    if not s:
        return None
    if len(_NUM_TOKEN.findall(s)) >= 3:
        return None  # item rows and other value-dense lines vary per document
    s = re.sub(r"([-=_*.|])\1{2,}", r"\1\1\1", s)  # separator rules vary in length
    s = re.sub(r"\d", "9", s)
    s = re.sub(r"9+", "9", s)
    return s


def doc_shapes(text: str) -> set[str]:
    return {sh for ln in text.splitlines() if (sh := shape(ln))}


def identify_vendor(text: str, vendor_master: list[dict]) -> Optional[str]:
    """Match the vendor by its registered tax ID (what an AP system would do)."""
    for v in vendor_master:
        if v["tax_id"] in text:
            return v["vendor_id"]
    return None


class LayoutModel:
    """Line shapes seen in accepted documents of one vendor, per layout epoch."""

    def __init__(self):
        self.counts: Counter = Counter()
        self.n_docs = 0

    def novelty(self, shapes: set[str]) -> list[str]:
        if self.n_docs == 0:
            return []
        return sorted(s for s in shapes if self.counts[s] == 0)

    def missing_core(self, shapes: set[str], min_share: float = 0.95) -> list[str]:
        """Shapes present in almost every past document but absent now (e.g. a renamed label)."""
        if self.n_docs < 3:
            return []
        return sorted(s for s, c in self.counts.items() if c / self.n_docs >= min_share and s not in shapes)

    def learn(self, shapes: set[str]) -> None:
        self.counts.update(shapes)
        self.n_docs += 1

    def reset(self) -> None:
        self.counts.clear()
        self.n_docs = 0
