"""Repair a damaged parser reading with the document's own redundancy, before paying for a model call.

An invoice is an error-correcting code: qty x price = amount on every line, lines sum to the subtotal,
subtotal - discount + tax = total, tax = rate x net, due date = invoice date + payment terms. When a scan
smudges one value, the others determine it. Distill recomputes it, and accepts the repair only when it is
the single change that makes every check pass; two different repairs that both work means the copy is
ambiguous and goes to the model as before. Nothing is guessed: the repaired record passes every check
the model's answer would have to pass, and repaired documents remain subject to audits.
"""
from __future__ import annotations

import copy
from datetime import date, timedelta
from typing import Callable, Optional

from .model import Record

MONEY_CHECKS = {"line_math", "subtotal_sum", "grand_total", "tax_rate", "totals_present", "subtotal_present",
                "discount_bounds"}
DATE_CHECKS = {"invoice_date_valid", "due_date_valid", "payment_terms", "due_after_invoice"}
ID_CHECKS = {"tax_id_format", "vendor_tax_id_match", "currency_match", "currency_iso"}
REPAIRABLE = MONEY_CHECKS | DATE_CHECKS | ID_CHECKS
TOL = 0.011


def _cents(x: float) -> Optional[float]:
    """x if it is a whole number of cents (a printed amount), else None."""
    if x is None or x != x or abs(x) > 1e12:
        return None
    r = round(x, 2)
    return r if abs(r - x) < 1e-6 * max(1.0, abs(x)) + 1e-9 else None


def _qty(x: float) -> Optional[float]:
    if x is None or x != x or x <= 0:
        return None
    r = round(x, 3)
    return r if abs(r - x) < 1e-6 * max(1.0, abs(x)) + 1e-9 else None


def _money_candidates(inv: Record, rate: Optional[float]) -> list[tuple[str, Record]]:
    out = []

    def cand(desc: str, f: Callable[[Record], bool]) -> None:
        r = copy.deepcopy(inv)
        try:
            if f(r):
                out.append((desc, r))
        except (TypeError, ZeroDivisionError, ValueError):
            pass

    items = inv.line_items
    sub, disc, tax, tot = inv.subtotal, inv.discount or 0.0, inv.tax, inv.total
    amounts = [li.amount for li in items]
    for i, li in enumerate(items):
        def set_amount(r, i=i):
            v = _cents(r.line_items[i].quantity * r.line_items[i].unit_price)
            if v is None:
                return False
            r.line_items[i].amount = v
            return True

        def set_qty(r, i=i):
            v = _qty(r.line_items[i].amount / r.line_items[i].unit_price)
            if v is None:
                return False
            r.line_items[i].quantity = v
            return True

        def set_price(r, i=i):
            v = _cents(r.line_items[i].amount / r.line_items[i].quantity)
            if v is None:
                return False
            r.line_items[i].unit_price = v
            return True

        def amount_from_subtotal(r, i=i):
            if sub is None:
                return False
            v = _cents(round(sub - sum(a for j, a in enumerate(amounts) if j != i), 2))
            if v is None:
                return False
            r.line_items[i].amount = v
            return True

        cand(f"line {i + 1} amount", set_amount)
        cand(f"line {i + 1} quantity", set_qty)
        cand(f"line {i + 1} unit price", set_price)
        cand(f"line {i + 1} amount", amount_from_subtotal)

    def setf(name, fn):
        def f(r):
            v = fn()
            if v is None:
                return False
            setattr(r, name, round(v, 2))
            return True
        return f

    lines_sum = round(sum(amounts), 2) if items else None
    cand("subtotal", setf("subtotal", lambda: lines_sum))
    cand("subtotal", setf("subtotal", lambda: None if tot is None or tax is None else tot + disc - tax))
    if rate:
        cand("subtotal", setf("subtotal", lambda: None if tax is None else tax / rate + disc))
        cand("tax", setf("tax", lambda: None if sub is None else (sub - disc) * rate))
    cand("tax", setf("tax", lambda: None if tot is None or sub is None else tot - sub + disc))
    cand("total", setf("total", lambda: None if sub is None or tax is None else sub - disc + tax))
    cand("discount", setf("discount", lambda: None if sub is None or tax is None or tot is None else sub + tax - tot))
    return out


def _date_candidates(inv: Record, terms: Optional[int]) -> list[tuple[str, Record]]:
    if terms is None:
        return []
    out = []
    for name, other, sign in (("due_date", "invoice_date", 1), ("invoice_date", "due_date", -1)):
        base = getattr(inv, other)
        try:
            d = date.fromisoformat(base) if base else None
        except ValueError:
            d = None
        if d is None:
            continue
        r = copy.deepcopy(inv)
        setattr(r, name, (d + timedelta(days=sign * terms)).isoformat())
        out.append((name.replace("_", " "), r))
    return out


def _id_candidates(inv: Record, master: Optional[dict]) -> list[tuple[str, Record]]:
    if not master:
        return []
    r = copy.deepcopy(inv)
    changed = []
    if inv.vendor_tax_id != master.get("tax_id"):
        r.vendor_tax_id = master.get("tax_id")
        changed.append("vendor tax id")
    if inv.currency != master.get("currency"):
        r.currency = master.get("currency")
        changed.append("currency")
    return [(" and ".join(changed), r)] if changed else []


def _fix_group(inv: Record, group: set, cands: list[tuple[str, Record]], check: Callable) -> Optional[tuple[str, Record]]:
    """The single candidate change that clears every failing check of this group (and breaks nothing else)."""
    before = {c.name for c in check(inv).fails}
    ok, seen = [], set()
    for desc, r in cands:
        fails = {c.name for c in check(r).fails}
        if fails & group or not fails <= before:
            continue
        key = repr(r.to_dict())
        if key in seen:
            continue
        seen.add(key)
        ok.append((desc, r))
    return ok[0] if len(ok) == 1 else None


def repair(inv: Optional[Record], check: Callable[[Record], object], master: Optional[dict]) -> Optional[tuple[Record, list[str]]]:
    """Return (repaired record, what was repaired) when the damage is uniquely repairable, else None."""
    if inv is None or inv.schema.name != "invoice":
        return None
    fails = {c.name for c in check(inv).fails}
    if not fails or not fails <= REPAIRABLE:
        return None
    cur, done = inv, []
    for group, gen in ((ID_CHECKS, lambda r: _id_candidates(r, master)),
                       (DATE_CHECKS, lambda r: _date_candidates(r, (master or {}).get("terms_days"))),
                       (MONEY_CHECKS, lambda r: _money_candidates(r, (master or {}).get("tax_rate")))):
        if not ({c.name for c in check(cur).fails} & group):
            continue
        fixed = _fix_group(cur, group, gen(cur), check)
        if fixed is None:
            return None
        done.append(fixed[0])
        cur = fixed[1]
    return (cur, done) if not check(cur).fails else None
