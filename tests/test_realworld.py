"""Tests for the real-world machinery: OCR noise, page registration, geometric templates, datasets."""
import random

from distill.datagen import DEMO_OVERRIDES, InvoiceStream
from distill.doctypes import DOCILE, RECEIPT
from distill.fingerprint import identify_vendor
from distill.model import Record, field_equal
from distill.ocr import corrupt, fuzzy_label, money_ocr_equivalent, ocr_equivalent
from distill.parser import execute
from distill.realdata import score
from distill.synth import Example, best_candidate, generate_candidates, validate
from distill.zones import fit_transform, learn_zones, register, signature


# ------------------------------------------------------------------ OCR noise
def test_noise_is_recognised_as_noise():
    assert ocr_equivalent("INV-2026-10234", "INV-2O26-1O234")
    assert ocr_equivalent("TD01167104", "TDO1167104")
    assert not ocr_equivalent("2026-01-04", "2026-04-01")  # swapped day/month is not a scanning slip
    assert money_ocr_equivalent(8905.17, 8905.1)  # one dropped digit
    assert not money_ocr_equivalent(37.50, 12.50)


def test_damaged_labels_still_match():
    assert fuzzy_label("Cate  ;", "Date :")
    assert fuzzy_label("Subtota1:", "Subtotal:")
    assert not fuzzy_label("Ship Date:", "Invoice No:")


def test_corruption_is_reproducible_and_rate_controlled():
    a, _ = corrupt("Grand Total: 1,234.56", 0.0, random.Random(1))
    assert a == "Grand Total: 1,234.56"
    b1, k1 = corrupt("Grand Total: 1,234.56", 0.3, random.Random(1))
    b2, _ = corrupt("Grand Total: 1,234.56", 0.3, random.Random(1))
    assert b1 == b2 and b1 != a and k1


# ------------------------------------------------------------------ layouts real documents have
def _clean_stream():
    return InvoiceStream(seed=11, vendor_overrides=DEMO_OVERRIDES, realistic=False)


def test_tables_continued_on_a_second_page_and_wrapped_rows():
    s = _clean_stream()
    vid = next(v.vendor_id for v in s.vendors if v.template["table"]["style"] == "spaces" and v.template["wrap_at"]
               and v.template["paginate_after"])
    s.vendor(vid).lines_range = (6, 8)
    train = [Example(d.text, d.truth, d.received_at, d.doc_id) for d in (s.next_doc(vid) for _ in range(5))]
    best, _ = best_candidate(generate_candidates(train), train)
    assert best is not None
    test = [Example(d.text, d.truth, d.received_at, d.doc_id) for d in (s.next_doc(vid) for _ in range(20))]
    assert any("Page 1 of 2" in e.text for e in test)
    assert validate(best, test).accuracy == 1.0


def test_vendor_recognised_despite_damaged_tax_id():
    s = _clean_stream()
    vm = s.vendor_master()
    d = s.next_doc("V05")
    tid = vm[4]["tax_id"]
    damaged = d.text.replace(tid, tid[:3] + ("O" if tid[3] == "0" else "0") + tid[4:])
    assert tid not in damaged
    assert identify_vendor(damaged, vm) == "V05"


# ------------------------------------------------------------------ geometric templates
def _page(dx=0.0, dy=0.0, doc_no="INV-001", total="1,234.50"):
    """A tiny synthetic page: a letterhead, a few labels and two values, shifted by (dx, dy)."""
    W = [
        (0.10, 0.05, "ACME"), (0.18, 0.05, "Industrial"), (0.32, 0.05, "Supplies"),
        (0.10, 0.10, "Warehouse"), (0.25, 0.10, "Road"), (0.60, 0.12, "Invoice"), (0.70, 0.12, "Number"),
        (0.60, 0.15, doc_no), (0.10, 0.80, "Thank"), (0.18, 0.80, "you"), (0.60, 0.70, "Total"),
        (0.72, 0.70, total), (0.10, 0.90, "Payment"), (0.25, 0.90, "terms"), (0.10, 0.40, "Description"),
    ]
    return [[0, x + dx, y + dy, x + dx + 0.06, y + dy + 0.012, t] for x, y, t in W]


def test_registration_aligns_shifted_scans():
    ref, shifted = _page(), _page(dx=0.03, dy=-0.05)
    T = fit_transform(signature(ref), signature(shifted))
    assert T is not None
    ax, bx, ay, by = T
    assert abs(bx + 0.03) < 0.005 and abs(by - 0.05) < 0.005


def test_zone_template_learned_on_shifted_pages():
    pages = [(_page(0.0, 0.0, "INV-001", "1,234.50"), {"document_id": "INV-001", "amount_due": "1,234.50"}),
             (_page(0.02, -0.04, "INV-002", "88.00"), {"document_id": "INV-002", "amount_due": "88.00"}),
             (_page(-0.01, 0.03, "INV-003", "990.10"), {"document_id": "INV-003", "amount_due": "990.10"})]
    registration, rules = learn_zones(pages, ["document_id", "amount_due"])
    assert set(rules) == {"document_id", "amount_due"}
    spec = {"schema": "docile", "fields": {k: {**r, "type": "raw"} for k, r in rules.items()},
            "registration": registration, "required": [], "num_style": "en", "date_style": "iso"}
    new = _page(0.04, -0.06, "INV-777", "42.00")
    r = execute(spec, "", DOCILE, new)
    assert r.record.document_id == "INV-777" and r.record.amount_due == "42.00"
    words, ok = register(new, registration)
    assert ok


# ------------------------------------------------------------------ dataset scoring
def test_scoring_accepts_any_annotated_value_and_ignores_punctuation():
    item = {"truth": Record(DOCILE, document_id="A-1"), "alts": {"document_id": ["A-1", "A1 "]}}
    assert score(Record(DOCILE, document_id="A1"), item) == []
    assert score(Record(DOCILE, document_id="A2"), item) == ["document_id"]
    assert score(Record(DOCILE, document_id="A1", order_id="X"), item) == ["order_id"]  # extra value is wrong
    assert field_equal("address", "NO 290, JALAN AIR PANAS. SETAPAK", "NO 290 JALAN AIR PANAS, SETAPAK", "raw")


def test_receipt_schema_round_trip():
    r = Record(RECEIPT, company="X SDN BHD", date="25/12/2018", address="NO 1", total=9.0)
    assert Record.from_dict(r.to_dict(), RECEIPT).values == r.values
