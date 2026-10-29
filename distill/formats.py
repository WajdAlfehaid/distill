"""Locale-aware rendering and parsing of numbers and dates.

The same helpers are used by the data generator (to render) and by compiled
parsers (to parse), so a parser spec only has to name a style, not carry code.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

# ---------------------------------------------------------------- numbers
NUM_STYLES = {
    # style: (thousands sep, decimal sep)
    "en": (",", "."),
    "eu": (".", ","),
    "plain": ("", "."),
    "space": (" ", "."),
}

NUM_REGEX = {
    "en": r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?",
    "eu": r"-?\d{1,3}(?:\.\d{3})*(?:,\d+)?|-?\d+(?:,\d+)?",
    "plain": r"-?\d+(?:\.\d+)?",
    "space": r"-?\d{1,3}(?: \d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?",
}


def fmt_num(x: float, style: str, decimals: int = 2) -> str:
    thou, dec = NUM_STYLES[style]
    neg = x < 0
    s = f"{abs(x):.{decimals}f}"
    if decimals:
        whole, frac = s.split(".")
    else:
        whole, frac = s, ""
    groups = []
    while len(whole) > 3:
        groups.insert(0, whole[-3:])
        whole = whole[:-3]
    groups.insert(0, whole)
    out = thou.join(groups)
    if frac:
        out += dec + frac
    return ("-" if neg else "") + out


def fmt_qty(q: float, style: str) -> str:
    if abs(q - round(q)) < 1e-9:
        return str(int(round(q)))
    return fmt_num(q, style, decimals=2)


def parse_num(s: str, style: str) -> Optional[float]:
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"[^\d,.\- ]", "", s).strip()
    if not s:
        return None
    thou, dec = NUM_STYLES[style]
    if thou:
        s = s.replace(thou, "")
    s = s.replace(" ", "")
    if dec != ".":
        s = s.replace(dec, ".")
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------- dates
DATE_FORMATS = {
    "iso": "%Y-%m-%d",
    "dmy_slash": "%d/%m/%Y",
    "mdy_slash": "%m/%d/%Y",
    "dmy_dot": "%d.%m.%Y",
    "d_mon_y": "%d %b %Y",
    "mon_d_y": "%B %d, %Y",
}

DATE_REGEX = {
    "iso": r"\d{4}-\d{2}-\d{2}",
    "dmy_slash": r"\d{2}/\d{2}/\d{4}",
    "mdy_slash": r"\d{2}/\d{2}/\d{4}",
    "dmy_dot": r"\d{2}\.\d{2}\.\d{4}",
    "d_mon_y": r"\d{2} [A-Z][a-z]{2} \d{4}",
    "mon_d_y": r"[A-Z][a-z]+ \d{2}, \d{4}",
}


def fmt_date(iso: str, style: str) -> str:
    return datetime.strptime(iso, "%Y-%m-%d").strftime(DATE_FORMATS[style])


def parse_date(s: str, style: str) -> Optional[str]:
    if s is None:
        return None
    try:
        return datetime.strptime(s.strip(), DATE_FORMATS[style]).date().isoformat()
    except ValueError:
        return None


def all_date_renderings(iso: str) -> dict[str, str]:
    return {style: fmt_date(iso, style) for style in DATE_FORMATS}


def days_between(a_iso: str, b_iso: str) -> int:
    return (date.fromisoformat(b_iso) - date.fromisoformat(a_iso)).days
