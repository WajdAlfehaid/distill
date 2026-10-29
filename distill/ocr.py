"""OCR noise: a realistic corruption model, and the tools to recognise it.

Scanned documents fail in characteristic ways: letters and digits that look alike
are swapped (0/O, 1/l, 5/S, 8/B), separators flip (, and .), characters drop out,
and spaces appear or vanish. The generator uses `corrupt` to produce that noise;
the engine uses `ocr_equivalent` and `fuzzy_label` to recognise it, so that one
smudged invoice is treated as noise rather than as evidence that a parser is wrong.
"""
from __future__ import annotations

import random
import re
from difflib import SequenceMatcher
from functools import lru_cache

SUBS = {"0": "O", "O": "0", "1": "l", "l": "1", "I": "1", "5": "S", "S": "5", "8": "B", "B": "8",
        "6": "G", ",": ".", ".": ",", "e": "c", "a": "o", "m": "rn", "-": "~", "/": "1"}

# characters a reader may normalise inside numeric fields
_NUMERIC_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "l": "1", "I": "1", "|": "1", "S": "5", "B": "8", "G": "6"})
_CANON = str.maketrans({"O": "0", "o": "0", "D": "0", "l": "1", "I": "1", "|": "1", "i": "1", "S": "5", "s": "5",
                        "B": "8", "G": "6", ",": ".", "c": "e", "~": "-", "_": "-"})


def corrupt(s: str, rate: float, rng: random.Random) -> tuple[str, set]:
    """Return (noisy string, kinds of damage applied: 'sub', 'drop', 'space')."""
    if rate <= 0 or not s:
        return s, set()
    out, kinds = [], set()
    for ch in s:
        if ch != " " and rng.random() < rate:
            r = rng.random()
            if r < 0.70 and ch in SUBS:
                out.append(SUBS[ch])
                kinds.add("sub")
                continue
            if r < 0.85:
                kinds.add("drop")
                continue
            out.append(ch + " ")
            kinds.add("space")
            continue
        if ch == " " and rng.random() < rate * 2:
            kinds.add("space")
            continue  # two words run together
        out.append(ch)
    return "".join(out), kinds


def numeric_fix(s: str) -> str:
    """Undo letter/digit confusions inside something that should be a number or date."""
    return s.translate(_NUMERIC_FIX)


def canon(s) -> str:
    return re.sub(r"\s+", "", str(s)).translate(_CANON).replace("rn", "m").casefold()


def ocr_equivalent(a, b, allow_drop: bool = True) -> bool:
    """True when a and b differ only in ways OCR damage explains (look-alike characters, spacing, one dropped char)."""
    if a is None or b is None:
        return False
    ca, cb = canon(a), canon(b)
    if ca == cb:
        return True
    if allow_drop and abs(len(ca) - len(cb)) == 1:
        short, long_ = sorted((ca, cb), key=len)
        return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))
    return False


def money_ocr_equivalent(a: float, b: float) -> bool:
    """Two amounts whose printed forms are one OCR slip apart (a dropped or flipped digit or separator)."""
    if a is None or b is None:
        return False
    forms = lambda x: {f"{x:.2f}", f"{x:.2f}".rstrip("0").rstrip("."), f"{x:.2f}".replace(".", "")}
    return any(ocr_equivalent(sa, sb) for sa in forms(a) for sb in forms(b))


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, canon(a), canon(b)).ratio()


@lru_cache(maxsize=65536)
def _label_key(s: str) -> str:
    return canon(re.sub(r"[^\w]+", "", re.sub(r"\d+", "#", s)))


def _edits(a: str, b: str, cap: int) -> int:
    """Levenshtein distance, giving up once it exceeds `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        best = i
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            best = min(best, cur[j])
        if best > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def fuzzy_label(prefix: str, label: str, threshold: float = 0.75) -> bool:
    """Does the text before a value look like the expected label after OCR damage?

    Compared with digits masked and punctuation dropped ('Cate  ;' ~ 'Date :'), allowing
    about one edit per six characters: a damaged label, not a different or truncated one.
    """
    if not label:
        return False
    a, b = _label_key(prefix[-40:]), _label_key(label)
    if not a or not b:
        return False
    if a == b:
        return True
    budget = max(1, len(b) // 6)
    return _edits(a, b, budget) <= budget
