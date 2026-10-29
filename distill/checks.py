"""Business checks that run on every extraction, whichever tier produced it.

Four families, cheapest first:
  invariant  arithmetic the invoice must satisfy (line math, subtotal, discount, tax, total)
  format     values must be well-formed (dates, tax-ID shape, ISO currency)
  xref       values must agree with master data (vendor, currency, tax rate, terms, PO, duplicates)
  history    values must look like this vendor's past invoices (prices, quantities, timing)

Invariant/format/xref failures are FAIL (the document cannot be auto-accepted).
History anomalies are WARN: arithmetic cannot catch a swapped quantity and unit
price (3 x 12.50 == 12.50 x 3) but this vendor's price history can.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from .model import Invoice

TOL = 0.011

TAX_ID_FORMATS = {
    "SA": r"3\d{13}3",
    "AE": r"100\d{12}",
    "US": r"\d{2}-\d{7}",
}
ISO_CURRENCIES = {"SAR", "AED", "USD", "EUR", "GBP"}


@dataclass
class Check:
    name: str
    family: str
    ok: bool
    severity: str = "fail"  # fail | warn
    detail: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "family": self.family, "ok": self.ok, "severity": self.severity, "detail": self.detail}


@dataclass
class CheckReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def fails(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and c.severity == "fail"]

    @property
    def warns(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and c.severity == "warn"]

    @property
    def passed(self) -> bool:
        return not self.fails

    def summary(self) -> str:
        if not self.fails and not self.warns:
            return f"all {len(self.checks)} checks passed"
        parts = [f"FAIL {c.name}" for c in self.fails] + [f"WARN {c.name}" for c in self.warns]
        return "; ".join(parts)

    def to_list(self) -> list[dict]:
        return [c.to_dict() for c in self.checks]


class VendorHistory:
    """Rolling memory of a vendor's accepted invoices."""

    def __init__(self, maxlen: int = 200):
        self.prices: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self.qty_integral: deque = deque(maxlen=maxlen)
        self.lags: deque = deque(maxlen=maxlen)
        self.invoice_numbers: set[str] = set()
        self.n = 0

    def learn(self, inv: Invoice, received_at: str) -> None:
        self.n += 1
        for li in inv.line_items:
            self.prices[li.description.casefold()].append(li.unit_price)
            self.qty_integral.append(abs(li.quantity - round(li.quantity)) < 1e-9)
        if inv.invoice_date:
            try:
                self.lags.append((datetime.fromisoformat(received_at).date() - date.fromisoformat(inv.invoice_date)).days)
            except ValueError:
                pass
        if inv.invoice_number:
            self.invoice_numbers.add(inv.invoice_number)


def _ok(name, family, cond, detail="", severity="fail") -> Check:
    return Check(name=name, family=family, ok=bool(cond), severity=severity, detail="" if cond else detail)


def run_checks(inv: Invoice, received_at: str, vendor: Optional[dict], po_master: dict[str, str],
               history: Optional[VendorHistory]) -> CheckReport:
    r = CheckReport()
    C = r.checks.append

    # ---------------------------------------------------------- invariants
    if not inv.line_items:
        C(Check("has_line_items", "invariant", False, detail="no line items extracted"))
    bad_lines = [i for i, li in enumerate(inv.line_items)
                 if not (math.isfinite(li.amount) and abs(li.quantity * li.unit_price - li.amount) <= TOL + 1e-6 * abs(li.amount))]
    C(_ok("line_math", "invariant", not bad_lines, f"qty x price != amount on lines {bad_lines}"))
    if inv.subtotal is None:
        C(Check("subtotal_present", "invariant", False, detail="subtotal missing"))
    else:
        s = round(sum(li.amount for li in inv.line_items), 2)
        C(_ok("subtotal_sum", "invariant", abs(s - inv.subtotal) <= TOL, f"lines sum {s:.2f} != subtotal {inv.subtotal:.2f}"))
    disc = inv.discount or 0.0
    C(_ok("discount_bounds", "invariant", 0 <= disc <= (inv.subtotal or 0) * 0.5 + TOL, "discount out of range"))
    if inv.total is None or inv.tax is None or inv.subtotal is None:
        C(Check("totals_present", "invariant", False, detail="total/tax/subtotal missing"))
    else:
        expect = round(inv.subtotal - disc + inv.tax, 2)
        C(_ok("grand_total", "invariant", abs(expect - inv.total) <= TOL,
              f"subtotal - discount + tax = {expect:.2f} != total {inv.total:.2f}"))

    # ---------------------------------------------------------- format
    try:
        inv_d = date.fromisoformat(inv.invoice_date) if inv.invoice_date else None
        due_d = date.fromisoformat(inv.due_date) if inv.due_date else None
    except ValueError:
        inv_d = due_d = None
    rec = datetime.fromisoformat(received_at).date()
    C(_ok("invoice_date_valid", "format", inv_d is not None, "invoice date missing or invalid"))
    C(_ok("due_date_valid", "format", due_d is not None, "due date missing or invalid"))
    if inv_d:
        C(_ok("not_future_dated", "format", (inv_d - rec).days <= 1, f"invoice dated {inv_d} after receipt {rec}"))
    if inv_d and due_d:
        C(_ok("due_after_invoice", "format", due_d >= inv_d, f"due {due_d} before invoice {inv_d}"))
    C(_ok("invoice_number_present", "format", bool(inv.invoice_number), "invoice number missing"))
    C(_ok("currency_iso", "format", inv.currency in ISO_CURRENCIES, f"currency {inv.currency!r} not recognised"))

    # ---------------------------------------------------------- xref (master data)
    if vendor:
        rx = TAX_ID_FORMATS.get(vendor["country"], vendor["country"] + r"\d{9}")
        C(_ok("tax_id_format", "format", bool(inv.vendor_tax_id and re.fullmatch(rx, inv.vendor_tax_id)),
              f"tax id {inv.vendor_tax_id!r} malformed for {vendor['country']}"))
        C(_ok("vendor_tax_id_match", "xref", inv.vendor_tax_id == vendor["tax_id"], "tax id differs from vendor master"))
        C(_ok("currency_match", "xref", inv.currency == vendor["currency"], f"currency {inv.currency} != master {vendor['currency']}"))
        if inv.subtotal is not None and inv.tax is not None:
            expect_tax = round((inv.subtotal - disc) * vendor["tax_rate"], 2)
            C(_ok("tax_rate", "xref", abs(expect_tax - inv.tax) <= 0.02,
                  f"tax {inv.tax:.2f} != {vendor['tax_rate']:.2%} of net ({expect_tax:.2f})"))
        if inv_d and due_d:
            C(_ok("payment_terms", "xref", (due_d - inv_d).days == vendor["terms_days"],
                  f"due - invoice = {(due_d - inv_d).days} days, master terms {vendor['terms_days']}"))
    if inv.po_number:
        owner = po_master.get(inv.po_number)
        C(_ok("po_exists", "xref", owner is not None, f"PO {inv.po_number} not found"))
        if owner is not None and vendor:
            C(_ok("po_vendor_match", "xref", owner == vendor["vendor_id"], f"PO {inv.po_number} belongs to another vendor"))
    if history and inv.invoice_number:
        C(_ok("not_duplicate", "xref", inv.invoice_number not in history.invoice_numbers,
              f"invoice {inv.invoice_number} already processed"))

    # ---------------------------------------------------------- history (warn)
    if history and history.n >= 5:
        outliers = []
        for li in inv.line_items:
            past = history.prices.get(li.description.casefold())
            if past and len(past) >= 5:
                lo, hi = min(past), max(past)
                if not (lo / 1.6 <= li.unit_price <= hi * 1.6):
                    outliers.append(f"{li.description}: {li.unit_price:g} vs usual {lo:g}-{hi:g}")
        r.checks.append(_ok("price_history", "history", not outliers, "; ".join(outliers[:3]), severity="warn"))
        if len(history.qty_integral) >= 10 and all(history.qty_integral) and inv.line_items:
            frac = [li.quantity for li in inv.line_items if abs(li.quantity - round(li.quantity)) > 1e-9]
            r.checks.append(_ok("qty_pattern", "history", not frac,
                                f"fractional quantities {frac[:3]} from a vendor that always bills whole units", severity="warn"))
        if inv_d and len(history.lags) >= 5:
            lag = (rec - inv_d).days
            r.checks.append(_ok("invoice_timing", "history", -1 <= lag <= max(history.lags) + 10,
                                f"invoice is {lag} days old; usual max {max(history.lags)}", severity="warn"))
    return r
