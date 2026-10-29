"""Synthetic invoice stream with exact ground truth and controllable layout drift.

Each vendor owns a *template*: labels, label placement, date and number locale,
money formatting, table style, totals block and noise lines (addresses, bank
details, a decoy "Ship Date" line). Documents are rendered as plain text, i.e.
what you get after PDF text extraction or OCR.

Drift is a mutation of one vendor's template from a point in the stream onward:

  LOUD        the parser breaks visibly (labels renamed, table re-flowed, number locale)
  STRUCTURAL  new content appears (discount line, extra reference field)
  SILENT      the parser keeps "working" but returns wrong values
              - date_locale: DD/MM/YYYY -> MM/DD/YYYY (valid dates, wrong meaning)
              - inline_swap: "3 @ 12.50" -> "12.50 @ 3" (qty x price still equals amount)
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Optional

from .formats import fmt_date, fmt_num, fmt_qty
from .model import Invoice, LineItem

DRIFT_KINDS = {
    "relabel": "loud",
    "table_reflow": "loud",
    "number_style": "loud",
    "new_fields": "structural",
    "date_locale": "silent",
    "inline_swap": "silent",
}

VENDOR_NAMES = [
    ("Al-Noor Industrial Supplies", "SA", "SAR"),
    ("Najd Office Solutions", "SA", "SAR"),
    ("Red Sea Logistics Co.", "SA", "SAR"),
    ("Qassim Fresh Foods", "SA", "SAR"),
    ("Desert Peak IT Services", "SA", "SAR"),
    ("Gulf Packaging Factory", "SA", "SAR"),
    ("Baltic Freight GmbH", "DE", "EUR"),
    ("Rheinwerk Maschinenbau AG", "DE", "EUR"),
    ("Lumière Bureau SARL", "FR", "EUR"),
    ("Iberia Componentes S.L.", "ES", "EUR"),
    ("Thames Consulting Ltd", "GB", "GBP"),
    ("Northwind Electrical Ltd", "GB", "GBP"),
    ("Cascade Cloud Inc.", "US", "USD"),
    ("Lone Star Fasteners LLC", "US", "USD"),
    ("Pacific Print Works", "US", "USD"),
    ("Emirates Steel Trading", "AE", "AED"),
    ("Falcon Security Systems", "AE", "AED"),
    ("Riyadh Medical Supply", "SA", "SAR"),
    ("Al-Qassim Cold Chain", "SA", "SAR"),
    ("Hijaz Construction Materials", "SA", "SAR"),
]

CATALOGS = {
    "industrial": [("Steel pipe 2in x 6m", 40, 180), ("Hex bolt M12 (box 100)", 15, 60), ("Hydraulic hose 1m", 25, 90),
                   ("Safety gloves (pair)", 3, 12), ("Welding rods 3.2mm (5kg)", 30, 75), ("Bearing 6205-2RS", 4, 18)],
    "office": [("A4 copy paper (box)", 18, 35), ("Toner cartridge black", 45, 140), ("Ergonomic chair", 150, 600),
               ("Whiteboard markers (12)", 6, 15), ("Filing cabinet 4-drawer", 120, 380), ("Desk lamp LED", 20, 70)],
    "logistics": [("Pallet freight Riyadh-Jeddah", 300, 900), ("Container handling fee", 150, 450), ("Customs clearance", 200, 800),
                  ("Last-mile delivery", 15, 60), ("Warehouse storage (pallet/week)", 8, 25), ("Fuel surcharge", 20, 120)],
    "food": [("Dates Sukkari 5kg", 40, 95), ("Fresh milk 1L (crate 12)", 30, 60), ("Basmati rice 10kg", 45, 90),
             ("Olive oil 5L", 60, 140), ("Chicken breast 10kg", 120, 260), ("Mixed vegetables 5kg", 25, 55)],
    "it": [("Laptop 14in i7", 900, 2400), ("Monitor 27in 4K", 250, 700), ("Network switch 24-port", 180, 650),
           ("Cloud backup (TB/month)", 20, 60), ("On-site support visit", 150, 400), ("Software license (seat/yr)", 60, 300)],
    "consulting": [("Senior consultant (hours)", 120, 260), ("Analyst (hours)", 60, 140), ("Workshop facilitation (day)", 900, 2200),
                   ("Travel expenses", 50, 400), ("Project management (hours)", 90, 180)],
    "medical": [("Nitrile gloves (box 100)", 8, 20), ("Surgical masks (box 50)", 5, 15), ("Syringes 5ml (box 100)", 12, 30),
                ("Blood pressure monitor", 60, 180), ("Disinfectant 5L", 20, 50), ("Gauze rolls (pack 12)", 6, 18)],
}

KEYWORD_CATALOG = [("consult", "consulting"), ("food", "food"), ("cold chain", "food"), ("medical", "medical"),
                   ("logistics", "logistics"), ("freight", "logistics"), ("office", "office"), ("bureau", "office"),
                   ("print", "office"), ("it services", "it"), ("cloud", "it"), ("security", "it"),
                   ("industrial", "industrial"), ("packaging", "industrial"), ("construction", "industrial"),
                   ("steel", "industrial"), ("fasteners", "industrial"), ("electrical", "industrial"),
                   ("maschinenbau", "industrial"), ("componentes", "industrial")]

TAX_RATES = {"SA": 0.15, "AE": 0.05, "DE": 0.19, "FR": 0.20, "ES": 0.21, "GB": 0.20, "US": 0.0825}
CURRENCY_SYMBOL = {"USD": "$", "EUR": "€", "GBP": "£", "SAR": "SAR", "AED": "AED"}


def make_tax_id(country: str, rng: random.Random) -> str:
    if country == "SA":
        return "3" + "".join(str(rng.randint(0, 9)) for _ in range(13)) + "3"
    if country == "AE":
        return "100" + "".join(str(rng.randint(0, 9)) for _ in range(12))
    if country == "US":
        return f"{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}"
    return country + "".join(str(rng.randint(0, 9)) for _ in range(9))


@dataclass
class Vendor:
    vendor_id: str
    name: str
    country: str
    currency: str
    tax_id: str
    tax_rate: float
    terms_days: int
    catalog: str
    decimal_qty: bool
    discount_prob: float
    lines_range: tuple[int, int]
    inv_prefix: str
    inv_counter: int
    po_numbers: list[str]
    template: dict
    template_rev: int = 0
    drift_log: list = field(default_factory=list)
    address: str = ""
    phone: str = ""
    iban: str = ""


@dataclass
class Doc:
    doc_id: str
    received_at: str
    text: str
    truth: Invoice
    vendor_id: str  # hidden: used only for evaluation and the dashboard
    template_rev: int
    drift_tag: Optional[str] = None  # set when the template is drifted


# ------------------------------------------------------------------ templates
LABEL_SETS = [
    {"invoice_number": "Invoice No:", "invoice_date": "Invoice Date:", "due_date": "Due Date:", "po_number": "PO Number:",
     "vendor_tax_id": "VAT No:", "currency": "Currency:", "subtotal": "Subtotal", "discount": "Discount", "tax": "VAT",
     "total": "TOTAL"},
    {"invoice_number": "INVOICE #", "invoice_date": "Date", "due_date": "Payment Due", "po_number": "Your Order Ref",
     "vendor_tax_id": "Tax Registration", "currency": "All amounts in", "subtotal": "Net Amount", "discount": "Less Discount",
     "tax": "Tax", "total": "Total Due"},
    {"invoice_number": "Bill Number", "invoice_date": "Bill Date", "due_date": "Pay By", "po_number": "Purchase Order",
     "vendor_tax_id": "TRN", "currency": "Curr.", "subtotal": "Sub-total", "discount": "Discount", "tax": "Sales Tax",
     "total": "Grand Total"},
    {"invoice_number": "Rechnung Nr.", "invoice_date": "Rechnungsdatum", "due_date": "Fällig am", "po_number": "Bestellnummer",
     "vendor_tax_id": "USt-IdNr.", "currency": "Währung", "subtotal": "Zwischensumme", "discount": "Rabatt", "tax": "MwSt.",
     "total": "Gesamtbetrag"},
    {"invoice_number": "Document No.", "invoice_date": "Issued", "due_date": "Due", "po_number": "P.O.",
     "vendor_tax_id": "Tax ID", "currency": "Currency", "subtotal": "Items total", "discount": "Promo discount",
     "tax": "Tax", "total": "AMOUNT PAYABLE"},
]

TABLE_HEADERS = [
    {"description": "Description", "quantity": "Qty", "unit_price": "Unit Price", "amount": "Amount"},
    {"description": "Item", "quantity": "Quantity", "unit_price": "Rate", "amount": "Line Total"},
    {"description": "Leistung", "quantity": "Menge", "unit_price": "Einzelpreis", "amount": "Betrag"},
    {"description": "Product / Service", "quantity": "Units", "unit_price": "Price", "amount": "Value"},
]

DATE_STYLE_BY_COUNTRY = {
    "SA": ["dmy_slash", "iso", "d_mon_y"],
    "AE": ["dmy_slash", "d_mon_y"],
    "DE": ["dmy_dot", "iso"],
    "FR": ["dmy_slash", "iso"],
    "ES": ["dmy_slash"],
    "GB": ["dmy_slash", "d_mon_y"],
    "US": ["mdy_slash", "mon_d_y", "iso"],
}
NUM_STYLE_BY_COUNTRY = {"SA": ["en", "plain"], "AE": ["en"], "DE": ["eu"], "FR": ["space", "eu"], "ES": ["eu"],
                        "GB": ["en"], "US": ["en"]}


def make_template(country: str, currency: str, rng: random.Random, force_table: Optional[str] = None,
                  force_date: Optional[str] = None) -> dict:
    labels = copy.deepcopy(rng.choice(LABEL_SETS[:3] if country not in ("DE",) else [LABEL_SETS[3]]))
    if rng.random() < 0.25:
        labels = copy.deepcopy(LABEL_SETS[4])
    header_order = ["invoice_number", "invoice_date", "due_date", "po_number", "currency"]
    rng.shuffle(header_order)
    label_mode = {f: ("next" if rng.random() < 0.2 else "same") for f in header_order}
    table_style = force_table or rng.choice(["pipe", "spaces", "spaces", "inline", "semicolon"])
    columns = ["description", "quantity", "unit_price", "amount"]
    if rng.random() < 0.3:
        columns = ["description", "unit_price", "quantity", "amount"]
    return {
        "labels": labels,
        "label_mode": label_mode,
        "sep": rng.choice([": ", "  ", " : "]),
        "header_order": header_order,
        "date_style": force_date or rng.choice(DATE_STYLE_BY_COUNTRY[country]),
        "num_style": rng.choice(NUM_STYLE_BY_COUNTRY[country]),
        "money_fmt": rng.choice(["code_before", "bare", "symbol" if currency in ("USD", "EUR", "GBP") else "code_after"]),
        "table": {
            "style": table_style,
            "columns": columns,
            "headers": copy.deepcopy(rng.choice(TABLE_HEADERS)),
            "inline_fmt": rng.choice(["{description} - {quantity} @ {unit_price} = {amount}",
                                      "{description} :: {quantity} x {unit_price} -> {amount}"]),
        },
        "tax_label_rate": rng.random() < 0.5,
        "show_ship_date": rng.random() < 0.5,
        "show_customer_ref": False,
        "footer": rng.choice(["Thank you for your business.", "Payment by bank transfer only.", "Page 1 of 1"]),
    }


# ------------------------------------------------------------------ rendering
def _money(x: float, tpl: dict, currency: str) -> str:
    n = fmt_num(x, tpl["num_style"])
    f = tpl["money_fmt"]
    if f == "code_before":
        return f"{currency} {n}"
    if f == "code_after":
        return f"{n} {currency}"
    if f == "symbol":
        sym = CURRENCY_SYMBOL[currency]
        return f"{sym}{n}" if len(sym) == 1 else f"{sym} {n}"
    return n


def render(vendor: Vendor, inv: Invoice, rng: random.Random) -> str:
    t = vendor.template
    sep = t["sep"]
    L = {k: (v.rstrip(":").rstrip() if ":" in sep else v) for k, v in t["labels"].items()}
    out: list[str] = []
    out.append(vendor.name.upper())
    out.append(vendor.address)
    out.append(vendor.phone)
    out.append(f"{L['vendor_tax_id']}{sep}{inv.vendor_tax_id}")
    out.append("")
    out.append("TAX INVOICE" if vendor.country in ("SA", "AE") else "INVOICE")
    out.append("")

    values = {
        "invoice_number": inv.invoice_number,
        "invoice_date": fmt_date(inv.invoice_date, t["date_style"]),
        "due_date": fmt_date(inv.due_date, t["date_style"]),
        "po_number": inv.po_number,
        "currency": inv.currency,
    }
    for f in t["header_order"]:
        if f == "po_number" and inv.po_number is None:
            continue
        if t["label_mode"].get(f) == "next":
            out.append(L[f])
            out.append(values[f])
        else:
            out.append(f"{L[f]}{sep}{values[f]}")
        if f == "invoice_date" and t["show_ship_date"]:
            ship = (date.fromisoformat(inv.invoice_date) - timedelta(days=rng.randint(1, 4))).isoformat()
            out.append(f"Ship Date{sep}{fmt_date(ship, t['date_style'])}")
    if t.get("show_customer_ref"):
        out.append(f"Customer Ref{sep}CR-{rng.randint(10000, 99999)}")
    out.append("")
    out.append("Bill To: Qassim Holding Co., Procurement Dept., Buraydah")
    out.append("")

    tb = t["table"]
    ns = t["num_style"]

    def cell(col: str, li: LineItem) -> str:
        if col == "description":
            return li.description
        if col == "quantity":
            return fmt_qty(li.quantity, ns)
        return fmt_num(getattr(li, col), ns)

    if tb["style"] == "inline":
        out.append("Items:")
        for li in inv.line_items:
            out.append(tb["inline_fmt"].format(description=li.description, quantity=fmt_qty(li.quantity, ns),
                                               unit_price=fmt_num(li.unit_price, ns), amount=fmt_num(li.amount, ns)))
    else:
        cols = tb["columns"]
        heads = [tb["headers"][c] for c in cols]
        rows = [[cell(c, li) for c in cols] for li in inv.line_items]
        if tb["style"] == "pipe":
            out.append("| " + " | ".join(heads) + " |")
            out.append("|" + "|".join("-" * (len(h) + 2) for h in heads) + "|")
            for r in rows:
                out.append("| " + " | ".join(r) + " |")
        elif tb["style"] == "semicolon":
            out.append(";".join(heads))
            for r in rows:
                out.append(";".join(r))
        else:  # spaces, aligned
            widths = [max(len(x) for x in [h] + [r[i] for r in rows]) for i, h in enumerate(heads)]
            out.append("   ".join(h.ljust(w) for h, w in zip(heads, widths)).rstrip())
            out.append("-" * (sum(widths) + 3 * (len(widths) - 1)))
            for r in rows:
                out.append("   ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths))).rstrip())
    out.append("")

    cur = inv.currency
    out.append(f"{L['subtotal']}{sep}{_money(inv.subtotal, t, cur)}")
    if inv.discount:
        out.append(f"{L['discount']}{sep}{_money(inv.discount, t, cur)}")
    tax_label = L["tax"] + (f" ({round(vendor.tax_rate * 100, 2):g}%)" if t["tax_label_rate"] else "")
    out.append(f"{tax_label}{sep}{_money(inv.tax, t, cur)}")
    out.append(f"{L['total']}{sep}{_money(inv.total, t, cur)}")
    out.append("")
    out.append(vendor.iban)
    out.append(f"Payment terms: Net {vendor.terms_days}")
    out.append(t["footer"])
    return "\n".join(out)


# ------------------------------------------------------------------ the stream
class InvoiceStream:
    def __init__(self, seed: int = 7, n_vendors: int = 20, start: str = "2026-01-04T08:00:00",
                 vendor_overrides: Optional[dict] = None):
        self.rng = random.Random(seed)
        self.clock = datetime.fromisoformat(start)
        self.seq = 0
        self.vendors: list[Vendor] = []
        self.po_master: dict[str, str] = {}
        overrides = vendor_overrides or {}
        catalog_by_country = {"SA": ["industrial", "office", "logistics", "food", "it", "medical"],
                              "AE": ["industrial", "it"], "DE": ["industrial", "logistics"], "FR": ["office"],
                              "ES": ["industrial"], "GB": ["consulting", "industrial"], "US": ["it", "industrial", "office"]}
        for i, (name, country, cur) in enumerate(VENDOR_NAMES[:n_vendors]):
            vid = f"V{i + 1:02d}"
            ov = overrides.get(vid, {})
            catalog = self.rng.choice(catalog_by_country[country])
            for kw, cat in KEYWORD_CATALOG:
                if kw in name.lower():
                    catalog = cat
                    break
            tpl = make_template(country, cur, self.rng, force_table=ov.get("table"), force_date=ov.get("date"))
            v = Vendor(
                vendor_id=vid, name=name, country=country, currency=cur, tax_id=make_tax_id(country, self.rng),
                tax_rate=TAX_RATES[country], terms_days=self.rng.choice([15, 30, 30, 45, 60]), catalog=catalog,
                decimal_qty=catalog in ("consulting", "food"), discount_prob=self.rng.choice([0.0, 0.0, 0.2, 0.4]),
                lines_range=(1, self.rng.choice([3, 5, 8])),
                inv_prefix=self.rng.choice(["INV-2026-", "A/", "", "BL", "RE-26-"]),
                inv_counter=self.rng.randint(100, 90000), po_numbers=[], template=tpl,
            )
            r = self.rng
            v.address = f"Building {r.randint(1000, 9999)}, {r.choice(['King Fahd Rd', 'Industrial Area 2', 'Hauptstrasse 12', 'Main St 400', 'Olaya St'])}"
            v.phone = f"Tel +{r.randint(10, 999)} {r.randint(100, 999)} {r.randint(1000, 9999)}"
            v.iban = f"IBAN: {v.country}{r.randint(10, 99)} {r.randint(1000, 9999)} {r.randint(1000, 9999)} {r.randint(1000, 9999)}"
            while len(v.po_numbers) < 40:
                po = f"PO-{self.rng.randint(400000, 499999)}"
                if po in self.po_master:
                    continue
                v.po_numbers.append(po)
                self.po_master[po] = vid
            self.vendors.append(v)
        # Zipf-like volume: a few vendors send most invoices
        self.weights = [1.0 / (i + 1) ** 0.8 for i in range(len(self.vendors))]
        self.rng.shuffle(self.weights)

    # ---------------------------------------------------------- master data
    def vendor_master(self) -> list[dict]:
        return [{"vendor_id": v.vendor_id, "name": v.name, "country": v.country, "currency": v.currency,
                 "tax_id": v.tax_id, "tax_rate": v.tax_rate, "terms_days": v.terms_days} for v in self.vendors]

    def vendor(self, vid: str) -> Vendor:
        return next(v for v in self.vendors if v.vendor_id == vid)

    # ---------------------------------------------------------- drift
    def apply_drift(self, vid: str, kind: str) -> str:
        """Mutate a vendor's template from now on. Returns a human description."""
        v = self.vendor(vid)
        t = copy.deepcopy(v.template)
        desc = ""
        if kind == "relabel":
            t["labels"]["total"] = "Amount Due" if t["labels"]["total"] != "Amount Due" else "Balance Payable"
            t["labels"]["invoice_number"] = "Invoice Ref."
            t["label_mode"]["due_date"] = "next" if t["label_mode"].get("due_date") == "same" else "same"
            desc = "renamed total and invoice-number labels; moved due date"
        elif kind == "table_reflow":
            t["table"]["style"] = "pipe" if t["table"]["style"] != "pipe" else "spaces"
            c = t["table"]["columns"]
            t["table"]["columns"] = ["description", "amount", "quantity", "unit_price"] if c[-1] == "amount" else \
                ["description", "quantity", "unit_price", "amount"]
            desc = "line-item table re-flowed with new column order"
        elif kind == "number_style":
            t["num_style"] = "eu" if t["num_style"] != "eu" else "en"
            desc = f"number format switched to {t['num_style']}"
        elif kind == "new_fields":
            t["show_customer_ref"] = True
            t["show_ship_date"] = True
            v.discount_prob = max(v.discount_prob, 0.6)
            desc = "added customer reference, ship date and discounts"
        elif kind == "date_locale":
            if t["date_style"] != "dmy_slash":
                raise ValueError("date_locale drift needs a vendor using DD/MM/YYYY")
            t["date_style"] = "mdy_slash"
            desc = "billing system switched dates from DD/MM/YYYY to MM/DD/YYYY"
        elif kind == "inline_swap":
            if t["table"]["style"] != "inline":
                raise ValueError("inline_swap drift needs a vendor with inline line items")
            f = t["table"]["inline_fmt"]
            t["table"]["inline_fmt"] = f.replace("{quantity}", "{TMP}").replace("{unit_price}", "{quantity}").replace("{TMP}", "{unit_price}")
            desc = "line items now print price before quantity"
        else:
            raise ValueError(f"unknown drift kind {kind}")
        v.template = t
        v.template_rev += 1
        v.drift_log.append({"at_doc": self.seq, "kind": kind, "class": DRIFT_KINDS[kind], "desc": desc})
        return desc

    def eligible_drifts(self, vid: str) -> list[str]:
        v = self.vendor(vid)
        kinds = ["relabel", "table_reflow", "number_style", "new_fields"]
        if v.template["date_style"] == "dmy_slash":
            kinds.append("date_locale")
        if v.template["table"]["style"] == "inline":
            kinds.append("inline_swap")
        return kinds

    # ---------------------------------------------------------- generation
    def _invoice(self, v: Vendor) -> Invoice:
        rng = self.rng
        inv_date = (self.clock - timedelta(days=rng.randint(0, 4))).date()
        due = inv_date + timedelta(days=v.terms_days)
        items = []
        catalog = CATALOGS[v.catalog]
        for desc, lo, hi in rng.sample(catalog, k=min(len(catalog), rng.randint(*v.lines_range))):
            price = round(rng.uniform(lo, hi), 2)
            qty = round(rng.uniform(1, 40) * 4) / 4 if v.decimal_qty else rng.randint(1, 60)
            if v.decimal_qty and abs(qty - round(qty)) < 1e-9 and rng.random() < 0.5:
                qty += 0.5
            items.append(LineItem(desc, float(qty), price, round(qty * price, 2)))
        subtotal = round(sum(li.amount for li in items), 2)
        discount = round(subtotal * rng.choice([0.02, 0.05, 0.10]), 2) if rng.random() < v.discount_prob else 0.0
        tax = round((subtotal - discount) * v.tax_rate, 2)
        total = round(subtotal - discount + tax, 2)
        v.inv_counter += rng.randint(1, 7)
        po = rng.choice(v.po_numbers) if rng.random() < 0.85 else None
        return Invoice(invoice_number=f"{v.inv_prefix}{v.inv_counter}", invoice_date=inv_date.isoformat(),
                       due_date=due.isoformat(), po_number=po, vendor_tax_id=v.tax_id, currency=v.currency,
                       subtotal=subtotal, discount=discount, tax=tax, total=total, line_items=items)

    def next_doc(self, vendor_id: Optional[str] = None) -> Doc:
        self.seq += 1
        self.clock += timedelta(minutes=self.rng.randint(2, 9))
        v = self.vendor(vendor_id) if vendor_id else self.rng.choices(self.vendors, weights=self.weights, k=1)[0]
        inv = self._invoice(v)
        text = render(v, inv, self.rng)
        tag = v.drift_log[-1]["kind"] if v.drift_log else None
        return Doc(doc_id=f"D{self.seq:05d}", received_at=self.clock.isoformat(), text=text, truth=inv,
                   vendor_id=v.vendor_id, template_rev=v.template_rev, drift_tag=tag)


# Reproducible demo/benchmark set-up: guarantees one vendor of each silent kind.
DEMO_OVERRIDES = {"V01": {"table": "inline", "date": "dmy_slash"}, "V02": {"table": "spaces", "date": "dmy_slash"},
                  "V03": {"table": "pipe"}}
