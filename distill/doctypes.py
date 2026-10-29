"""Document types: what a document contains, how it is checked, how its source is recognised.

  invoice   synthetic supplier invoices (the live drift demo), vendor master + PO master
  receipt   SROIE scanned receipts (ICDAR 2019): company, date, address, total
  docile    DocILE business documents (Rossum, ICDAR 2023): invoices, orders, credit notes,
            utility bills; key fields labelled exactly as printed

Each type supplies the schema, the checks and source identification; the engine,
parser synthesis, drift monitor, routing and audits are shared.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from .checks import Check, CheckReport, VendorHistory, run_checks
from .fingerprint import LayoutClusterer, identify_vendor
from .model import INVOICE, FieldSpec, Record, Schema
from .ocr import numeric_fix

# ---------------------------------------------------------------- schemas
RECEIPT = Schema("receipt", (
    FieldSpec("company", "raw", label="merchant's registered company name as printed"),
    FieldSpec("date", "raw", label="receipt date exactly as printed (e.g. 25/12/2018), without the time"),
    FieldSpec("address", "raw", label="merchant address as printed, lines joined with spaces"),
    FieldSpec("total", "money", label="final amount paid (after rounding), as a number"),
), items=False, description="scanned retail receipts")

DOCILE = Schema("docile", (
    FieldSpec("document_id", "raw", optional=True, label="invoice / document number"),
    FieldSpec("date_issue", "raw", optional=True, label="issue date as printed"),
    FieldSpec("date_due", "raw", optional=True, label="due date as printed"),
    FieldSpec("order_id", "raw", optional=True, label="purchase order number"),
    FieldSpec("customer_id", "raw", optional=True, label="customer / account number"),
    FieldSpec("vendor_name", "raw", optional=True, label="supplier name"),
    FieldSpec("vendor_address", "raw", optional=True, label="supplier address"),
    FieldSpec("vendor_tax_id", "raw", optional=True, label="supplier tax / VAT id"),
    FieldSpec("customer_billing_name", "raw", optional=True, label="name of the customer being billed"),
    FieldSpec("payment_terms", "raw", optional=True, label="payment terms text"),
    FieldSpec("currency_code_amount_due", "raw", optional=True, label="currency of the amount due, as printed"),
    FieldSpec("amount_total_net", "raw", optional=True, label="total before tax, as printed"),
    FieldSpec("amount_total_tax", "raw", optional=True, label="total tax, as printed"),
    FieldSpec("amount_total_gross", "raw", optional=True, label="total including tax, as printed"),
    FieldSpec("amount_due", "raw", optional=True, label="amount due, as printed"),
), items=False, description="business documents (invoices, orders, credit notes, utility bills)")

SCHEMAS = {s.name: s for s in (INVOICE, RECEIPT, DOCILE)}

# label words that make a line a better home for a field when several lines carry the same value
FIELD_HINTS = {
    "total": ["total", "amount", "rm", "payable", "due", "nett", "net"],
    "amount_total_gross": ["total", "gross", "invoice"],
    "amount_due": ["due", "balance", "pay"],
    "amount_total_net": ["net", "sub"],
    "amount_total_tax": ["tax", "vat", "gst"],
    "date": ["date"], "date_issue": ["date", "invoice"], "date_due": ["due"],
    "document_id": ["invoice", "no", "number", "#", "document"],
    "order_id": ["order", "po", "p.o"], "customer_id": ["customer", "account", "acct"],
}


def schema_by_name(name: str) -> Schema:
    return SCHEMAS[name]


# ---------------------------------------------------------------- helpers for loose checks
_DATE_FORMATS = ["%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y", "%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y", "%m/%d/%Y",
                 "%m/%d/%y", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%d-%b-%Y", "%d-%b-%y", "%Y/%m/%d"]


def loose_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    t = re.sub(r"\s+", " ", str(s).strip())
    for cand in (t, numeric_fix(t)):
        for f in _DATE_FORMATS:
            try:
                return datetime.strptime(cand, f).date()
            except ValueError:
                pass
    return None


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
    """Sources are recognised by layout clustering (or, for DocILE, optionally by the given cluster id)."""

    def __init__(self, use_given_clusters: bool = False, threshold: float = 0.5):
        self.use_given = use_given_clusters
        self.clusterer = LayoutClusterer(threshold=threshold)

    def identify(self, text, meta=None):
        if self.use_given and meta and meta.get("cluster_id") is not None:
            return f"K{meta['cluster_id']}"
        return self.clusterer.assign(text)

    def master(self, key):
        return {"vendor_id": key, "name": f"layout {key}", "currency": "USD"}


class ReceiptHistory(VendorHistory):
    pass


class ReceiptType(_ClusteredType):
    name = "receipt"
    schema = RECEIPT

    def checks(self, rec, received_at, master, history):
        r = CheckReport()
        total = rec.total
        r.checks.append(Check("total_present", "format", total is not None and total == total and total > 0,
                              detail=f"total {total!r} missing or not positive"))
        d = loose_date(rec.date)
        r.checks.append(Check("date_valid", "format", d is not None, detail=f"date {rec.date!r} unreadable"))
        if d:
            r.checks.append(Check("date_plausible", "format", 2010 <= d.year <= 2025,
                                  detail=f"date {d} outside the dataset's range"))
        r.checks.append(Check("company_present", "format", bool(rec.company), detail="company missing"))
        if history and history.n >= 5 and total:
            past = list(history.totals)
            hi = max(past)
            r.checks.append(Check("total_history", "history", total <= hi * 5 + 50, severity="warn",
                                  detail=f"total {total} far above this merchant's usual (max {hi})"))
        return r

    def new_history(self):
        h = VendorHistory()
        h.totals = []
        return h

    def learn(self, history, rec, received_at):
        history.n += 1
        if rec.total:
            history.totals.append(rec.total)

    def value(self, rec, master):
        return (rec.total or 0.0) if rec else 0.0, "MYR"


class DocileType(_ClusteredType):
    name = "docile"
    schema = DOCILE

    def checks(self, rec, received_at, master, history):
        r = CheckReport()
        amounts = {k: loose_amount(getattr(rec, k)) for k in
                   ("amount_total_net", "amount_total_tax", "amount_total_gross", "amount_due")}
        for k, v in amounts.items():
            if getattr(rec, k) is not None:
                r.checks.append(Check(f"{k}_numeric", "format", v is not None, detail=f"{k} {getattr(rec, k)!r} is not a number"))
        net, tax, gross = amounts["amount_total_net"], amounts["amount_total_tax"], amounts["amount_total_gross"]
        if None not in (net, tax, gross):
            r.checks.append(Check("net_plus_tax", "invariant", abs(abs(net) + abs(tax) - abs(gross)) <= 0.02 + 0.001 * abs(gross),
                                  detail=f"net {net} + tax {tax} != gross {gross}"))
        for k in ("date_issue", "date_due"):
            if getattr(rec, k):
                r.checks.append(Check(f"{k}_has_digits", "format", bool(re.search(r"\d", getattr(rec, k))),
                                      detail=f"{k} {getattr(rec, k)!r} has no digits"))
        return r

    def new_history(self):
        h = VendorHistory()
        return h

    def learn(self, history, rec, received_at):
        history.n += 1

    def value(self, rec, master):
        if rec is None:
            return 0.0, "USD"
        for k in ("amount_total_gross", "amount_due"):
            v = loose_amount(getattr(rec, k))
            if v is not None:
                return abs(v), "USD"
        return 0.0, "USD"
