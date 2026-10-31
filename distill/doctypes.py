"""Document types: what a document contains, how it is checked, how its source is recognised.

  invoice   generated supplier invoices with template changes, vendor master + PO master
  fatura    FATURA invoices (Limam et al. 2023): 50 third-party invoice layouts, fields as printed

Each type supplies the schema, the checks and source identification; the engine,
parser synthesis, drift monitor, routing and audits are shared.
"""
from __future__ import annotations

import re
from typing import Optional

from .checks import Check, CheckReport, VendorHistory, run_checks
from .fingerprint import LayoutClusterer, identify_vendor
from .model import INVOICE, FieldSpec, Record, Schema
from .ocr import numeric_fix

# ---------------------------------------------------------------- schemas
FATURA = Schema("fatura", (
    FieldSpec("invoice_number", "raw", optional=True, label="invoice number"),
    FieldSpec("invoice_date", "raw", optional=True, label="invoice date as printed"),
    FieldSpec("due_date", "raw", optional=True, label="due date as printed"),
    FieldSpec("po_number", "raw", optional=True, label="purchase order number"),
    FieldSpec("subtotal", "raw", optional=True, label="subtotal as printed"),
    FieldSpec("tax", "raw", optional=True, label="tax amount as printed"),
    FieldSpec("discount", "raw", optional=True, label="discount as printed"),
    FieldSpec("total", "raw", optional=True, label="total as printed, with currency if printed"),
    FieldSpec("amount_due", "raw", optional=True, label="amount due as printed"),
), items=False, description="invoices")

SCHEMAS = {s.name: s for s in (INVOICE, FATURA)}

# label words that make a line a better home for a field when several lines carry the same value
FIELD_HINTS = {
    "total": ["total", "amount", "rm", "payable", "due", "nett", "net"],
    "amount_due": ["due", "balance", "pay"],
}

def schema_by_name(name: str) -> Schema:
    return SCHEMAS[name]


# ---------------------------------------------------------------- helpers for loose checks
def loose_amount(s) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    t = numeric_fix(str(s)).replace(" ", "")
    t = re.sub(r"[^\d,.\-]", "", t)
    if not re.search(r"\d", t):
        return None
    if "," in t and "." in t:
        t = t.replace(",", "") if t.rfind(".") > t.rfind(",") else t.replace(".", "").replace(",", ".")
    elif "," in t:
        head, _, tail = t.rpartition(",")
        t = head.replace(",", "") + ("." + tail if len(tail) == 2 else tail)
    try:
        return float(t.rstrip("-")) * (-1 if t.endswith("-") else 1)
    except ValueError:
        return None


# ---------------------------------------------------------------- document types
class DocType:
    schema: Schema
    name: str

    def identify(self, text: str, meta: Optional[dict]) -> Optional[str]:
        raise NotImplementedError

    def master(self, key: str) -> dict:
        return {"vendor_id": key, "name": key, "currency": "USD"}

    def new_history(self) -> VendorHistory:
        return VendorHistory()

    def checks(self, rec: Record, received_at: str, master: Optional[dict], history) -> CheckReport:
        raise NotImplementedError

    def learn(self, history, rec: Record, received_at: str) -> None:
        history.learn(rec, received_at)

    def value(self, rec: Optional[Record], master: dict) -> tuple[float, str]:
        return (rec.value() if rec else 0.0), master.get("currency", "USD")


class InvoiceType(DocType):
    name = "invoice"
    schema = INVOICE

    def __init__(self, vendor_master: list[dict], po_master: dict[str, str]):
        self.vendor_master = vendor_master
        self.by_id = {v["vendor_id"]: v for v in vendor_master}
        self.po_master = po_master

    def identify(self, text, meta=None):
        return identify_vendor(text, self.vendor_master)

    def master(self, key):
        return self.by_id[key]

    def checks(self, rec, received_at, master, history):
        return run_checks(rec, received_at, master, self.po_master, history)


class _ClusteredType(DocType):
    """Sources are recognised by layout clustering, or optionally by an id the dataset provides (like a vendor master)."""

    def __init__(self, use_given_clusters: bool = False, threshold: float = 0.5):
        self.use_given = use_given_clusters
        self.clusterer = LayoutClusterer(threshold=threshold)

    def identify(self, text, meta=None):
        if self.use_given and meta and meta.get("cluster_id") is not None:
            return f"K{meta['cluster_id']}"
        return self.clusterer.assign(text)

    def master(self, key):
        return {"vendor_id": key, "name": f"layout {key}", "currency": "USD"}


class FaturaType(_ClusteredType):
    """FATURA invoices (Limam et al. 2023): 50 invoice templates; the source is the template, as a vendor would be."""
    name = "fatura"
    schema = FATURA

    def checks(self, rec, received_at, master, history):
        r = CheckReport()
        for k in ("subtotal", "tax", "total", "amount_due"):
            if getattr(rec, k) is not None:
                r.checks.append(Check(f"{k}_numeric", "format", loose_amount(getattr(rec, k)) is not None,
                                      detail=f"{k} {getattr(rec, k)!r} is not a number"))
        for k in ("invoice_date", "due_date"):
            if getattr(rec, k):
                r.checks.append(Check(f"{k}_has_digits", "format", bool(re.search(r"\d", getattr(rec, k))),
                                      detail=f"{k} {getattr(rec, k)!r} has no digits"))
        r.checks.append(Check("invoice_number_present", "format", bool(rec.invoice_number) or True))
        return r

    def new_history(self):
        return VendorHistory()
