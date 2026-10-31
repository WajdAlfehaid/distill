"""Generic document records, schemas and field-level comparison.

A *schema* names the fields a document type carries and how each is typed:

  money    amounts, compared to the cent
  date     ISO dates (YYYY-MM-DD), parsed with a learned date style
  code     short upper-case codes such as ISO currencies
  text     identifiers and names, compared case- and whitespace-insensitively
  raw      text kept as printed (e.g. "25/12/2018", "1/02/02"); compared on letters and
           digits only, ignoring case, spacing and punctuation (the public datasets'
           annotators disagree with themselves on commas and full stops)

`Record` holds one document's values. Field names are attributes (`rec.total`),
so domain code stays readable. `Invoice` is the record type of the invoice schema.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

LINE_FIELDS = ["description", "quantity", "unit_price", "amount"]
CENT = 0.0051


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: str  # money | date | code | text | raw
    optional: bool = False
    label: str = ""  # human description, used in LLM prompts


_TYPES_CACHE: dict = {}


@dataclass(frozen=True)
class Schema:
    name: str
    fields: tuple
    items: bool = False
    description: str = ""

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.fields]

    @property
    def types(self) -> dict[str, str]:
        t = _TYPES_CACHE.get(self.name)
        if t is None:
            t = _TYPES_CACHE[self.name] = {f.name: f.type for f in self.fields}
        return t

    @property
    def optional(self) -> set[str]:
        return {f.name for f in self.fields if f.optional}

    def get(self, name: str) -> Optional[FieldSpec]:
        return next((f for f in self.fields if f.name == name), None)


INVOICE = Schema("invoice", (
    FieldSpec("invoice_number", "text", label="invoice number as printed"),
    FieldSpec("invoice_date", "date", label="invoice date, YYYY-MM-DD"),
    FieldSpec("due_date", "date", label="payment due date, YYYY-MM-DD"),
    FieldSpec("po_number", "text", optional=True, label="buyer's purchase order number, or null"),
    FieldSpec("vendor_tax_id", "text", label="supplier's VAT / tax registration number"),
    FieldSpec("currency", "code", label="ISO 4217 currency code"),
    FieldSpec("subtotal", "money", label="sum of line amounts before discount and tax"),
    FieldSpec("discount", "money", optional=True, label="discount amount, 0 when absent"),
    FieldSpec("tax", "money", label="tax amount"),
    FieldSpec("total", "money", label="amount payable"),
), items=True, description="supplier invoices")

# kept for older imports
HEADER_FIELDS = INVOICE.names
MONEY_FIELDS = {f.name for f in INVOICE.fields if f.type == "money"}
DATE_FIELDS = {f.name for f in INVOICE.fields if f.type == "date"}
TEXT_FIELDS = {f.name for f in INVOICE.fields if f.type in ("text", "code")}


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


class Record:
    """Field values of one document under a schema, plus line items when the schema has them."""

    __slots__ = ("schema", "values", "line_items")

    def __init__(self, schema: Schema = INVOICE, line_items: Optional[list] = None, **values):
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "values", {n: None for n in schema.names})
        object.__setattr__(self, "line_items", list(line_items or []))
        for k, v in values.items():
            setattr(self, k, v)

    def __getattr__(self, name: str):
        values = object.__getattribute__(self, "values")
        if name in values:
            return values[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name in ("schema", "values", "line_items"):
            object.__setattr__(self, name, value)
        elif name in self.values:
            self.values[name] = value
        else:
            raise AttributeError(f"{self.schema.name} has no field {name!r}")

    def __deepcopy__(self, memo):
        import copy
        r = Record(self.schema)
        r.values.update(copy.deepcopy(self.values, memo))
        r.line_items = copy.deepcopy(self.line_items, memo)
        return r

    def __repr__(self) -> str:
        return f"Record({self.schema.name}, {self.values}, items={len(self.line_items)})"

    def to_dict(self) -> dict:
        d = dict(self.values)
        if self.schema.items:
            d["line_items"] = [li.to_dict() for li in self.line_items]
        return d

    @classmethod
    def from_dict(cls, d: Optional[dict], schema: Schema = INVOICE) -> "Record":
        d = d or {}
        rec = Record(schema)
        for f in schema.fields:
            v = d.get(f.name)
            if v in ("", "null", "None"):
                v = None
            if v is not None:
                if f.type == "money":
                    v = _num(v)
                else:
                    v = str(v).strip()
            rec.values[f.name] = v
        if schema.items:
            rec.line_items = [LineItem.from_dict(x) for x in (d.get("line_items") or [])]
        return rec

    def value(self) -> float:
        for k in ("total", "amount_total_gross", "amount_due"):
            v = self.values.get(k)
            if isinstance(v, (int, float)) and math.isfinite(v):
                return float(v)
        return 0.0


def Invoice(**kw) -> Record:  # noqa: N802 - reads like the record type it builds
    return Record(INVOICE, **kw)


Invoice.from_dict = lambda d: Record.from_dict(d, INVOICE)  # type: ignore[attr-defined]


def _num(v: Any) -> float:
    if v is None or v == "":
        return 0.0
    if isinstance(v, str):  # tolerate "1,234.56" or "SAR 1234.56" from a model
        v = re.sub(r"[^\d.\-]", "", v.replace(",", "")) or "nan"
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return float("nan")


def _norm_text(v: Any) -> str:
    return " ".join(str(v).split()).casefold() if v is not None else ""


def _norm_raw(v: Any) -> str:
    """Printed values are compared on letters and digits only: annotators disagree on commas and spaces."""
    return re.sub(r"[\W_]+", "", str(v)).casefold() if v is not None else ""


def field_equal(name: str, a: Any, b: Any, ftype: Optional[str] = None) -> bool:
    ftype = ftype or INVOICE.types.get(name, "text")
    if ftype == "money":
        a = 0.0 if a is None else a
        b = 0.0 if b is None else b
        try:
            return abs(float(a) - float(b)) < CENT
        except (TypeError, ValueError):
            return False
    if a is None or b is None:
        return a is None and b is None
    if ftype == "raw":
        return _norm_raw(a) == _norm_raw(b)
    return _norm_text(a) == _norm_text(b)


def line_equal(a: LineItem, b: LineItem) -> bool:
    return (
        _norm_text(a.description) == _norm_text(b.description)
        and abs(a.quantity - b.quantity) < 1e-6
        and abs(a.unit_price - b.unit_price) < CENT
        and abs(a.amount - b.amount) < CENT
    )


def diff(pred: Record, truth: Record, fields: Optional[list[str]] = None) -> list[str]:
    """Names of fields where pred disagrees with truth ([] = exact match).

    `fields` restricts the comparison (e.g. to the fields a dataset actually labels).
    """
    schema = truth.schema
    names = fields or schema.names
    bad = [k for k in names if not field_equal(k, pred.values.get(k), truth.values.get(k), schema.types[k])]
    if schema.items and fields is None:
        if len(pred.line_items) != len(truth.line_items):
            bad.append("line_items")
        else:
            for i, (p, t) in enumerate(zip(pred.line_items, truth.line_items)):
                if not line_equal(p, t):
                    bad.append(f"line_items[{i}]")
    return bad


def field_count(rec: Record) -> int:
    return len(rec.schema.fields) + (4 * len(rec.line_items) if rec.schema.items else 0)
