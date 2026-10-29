"""Core data model: invoices, line items, and field-level comparison.

Money is held as float rounded to cents; every comparison uses a cent tolerance.
Dates are ISO strings (YYYY-MM-DD) so they serialize cleanly and compare exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

HEADER_FIELDS = [
    "invoice_number",
    "invoice_date",
    "due_date",
    "po_number",
    "vendor_tax_id",
    "currency",
    "subtotal",
    "discount",
    "tax",
    "total",
]
MONEY_FIELDS = {"subtotal", "discount", "tax", "total"}
DATE_FIELDS = {"invoice_date", "due_date"}
TEXT_FIELDS = {"invoice_number", "po_number", "vendor_tax_id", "currency"}
LINE_FIELDS = ["description", "quantity", "unit_price", "amount"]

CENT = 0.0051


@dataclass
class LineItem:
    description: str
    quantity: float
    unit_price: float
    amount: float

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "LineItem":
        return LineItem(
            description=str(d.get("description", "")).strip(),
            quantity=_num(d.get("quantity")),
            unit_price=_num(d.get("unit_price")),
            amount=_num(d.get("amount")),
        )


@dataclass
class Invoice:
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    po_number: Optional[str] = None
    vendor_tax_id: Optional[str] = None
    currency: Optional[str] = None
    subtotal: Optional[float] = None
    discount: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    line_items: list[LineItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in HEADER_FIELDS}
        d["line_items"] = [li.to_dict() for li in self.line_items]
        return d

    @staticmethod
    def from_dict(d: dict | None) -> "Invoice":
        d = d or {}
        inv = Invoice()
        for k in HEADER_FIELDS:
            v = d.get(k)
            if v in ("", "null"):
                v = None
            if k in MONEY_FIELDS and v is not None:
                v = _num(v)
            elif v is not None:
                v = str(v).strip()
            setattr(inv, k, v)
        inv.line_items = [LineItem.from_dict(x) for x in (d.get("line_items") or [])]
        return inv

    def value(self) -> float:
        return float(self.total or 0.0)


def _num(v: Any) -> float:
    if v is None or v == "":
        return 0.0
    if isinstance(v, str):  # tolerate "1,234.56" or "SAR 1234.56" from a model
        import re
        v = re.sub(r"[^\d.\-]", "", v.replace(",", "")) or "nan"
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return float("nan")


def _norm_text(v: Any) -> str:
    return " ".join(str(v).split()).casefold() if v is not None else ""


def field_equal(name: str, a: Any, b: Any) -> bool:
    if name in MONEY_FIELDS:
        a = 0.0 if a is None else a
        b = 0.0 if b is None else b
        try:
            return abs(float(a) - float(b)) < CENT
        except (TypeError, ValueError):
            return False
    if a is None or b is None:
        return a is None and b is None
    return _norm_text(a) == _norm_text(b)


def line_equal(a: LineItem, b: LineItem) -> bool:
    return (
        _norm_text(a.description) == _norm_text(b.description)
        and abs(a.quantity - b.quantity) < 1e-6
        and abs(a.unit_price - b.unit_price) < CENT
        and abs(a.amount - b.amount) < CENT
    )


def diff(pred: Invoice, truth: Invoice) -> list[str]:
    """Return the names of fields where pred disagrees with truth ([] = exact match)."""
    bad = [k for k in HEADER_FIELDS if not field_equal(k, getattr(pred, k), getattr(truth, k))]
    if len(pred.line_items) != len(truth.line_items):
        bad.append("line_items")
    else:
        for i, (p, t) in enumerate(zip(pred.line_items, truth.line_items)):
            if not line_equal(p, t):
                bad.append(f"line_items[{i}]")
    return bad


def field_count(inv: Invoice) -> int:
    return len(HEADER_FIELDS) + 4 * len(inv.line_items)
