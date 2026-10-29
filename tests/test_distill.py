"""Core behaviour tests. Run with:  python -m pytest -q"""
import pytest

from distill.audit import audit_rate, error_upper_bound
from distill.checks import VendorHistory, run_checks
from distill.datagen import DEMO_OVERRIDES, InvoiceStream
from distill.doctypes import InvoiceType
from distill.engine import Config, Engine
from distill.fingerprint import doc_shapes, identify_vendor
from distill.formats import fmt_num, parse_date, parse_num
from distill.llm import SimulatedLLM
from distill.model import Invoice, LineItem, diff
from distill.parser import execute
from distill.policy import RouteInput, decide
from distill.synth import Example, best_candidate, generate_candidates, validate


@pytest.fixture
def stream():
    # parser mechanics are unit-tested on clean prints; scan noise is covered by the end-to-end tests
    return InvoiceStream(seed=11, vendor_overrides=DEMO_OVERRIDES, realistic=False)


def examples(stream, vid, n):
    out = []
    for _ in range(n):
        d = stream.next_doc(vid)
        out.append(Example(d.text, d.truth, d.received_at, d.doc_id))
    return out


# ------------------------------------------------------------------ formats
def test_number_roundtrip():
    for style in ("en", "eu", "plain", "space"):
        assert parse_num(fmt_num(1234567.89, style), style) == pytest.approx(1234567.89)


def test_ambiguous_dates_parse_differently():
    assert parse_date("03/04/2026", "dmy_slash") == "2026-04-03"
    assert parse_date("03/04/2026", "mdy_slash") == "2026-03-04"


# ------------------------------------------------------------------ synthesis
@pytest.mark.parametrize("vid", ["V01", "V02", "V03", "V10", "V11", "V14", "V20"])
def test_synthesized_parser_generalises(stream, vid):
    train = examples(stream, vid, 6)
    best, _ = best_candidate(generate_candidates(train), train)
    assert best is not None, "no parser synthesized"
    test = examples(stream, vid, 40)
    # fields never seen in training (e.g. a first discount) are allowed to miss; everything else must match
    v = validate(best, [e for e in test if not (e.labels.discount and not any(t.labels.discount for t in train))
                        and not (e.labels.po_number and not any(t.labels.po_number for t in train))])
    assert v.accuracy == 1.0, v.errors


def test_spec_is_data_not_code(stream):
    train = examples(stream, "V01", 4)
    best, _ = best_candidate(generate_candidates(train), train)
    import json
    assert json.loads(json.dumps(best)) == best  # fully serialisable, nothing executable


def test_inline_swap_fools_parser_silently(stream):
    """The silent drift really is silent: the old parser still 'works' and arithmetic still passes."""
    train = examples(stream, "V01", 5)
    best, _ = best_candidate(generate_candidates(train), train)
    stream.apply_drift("V01", "inline_swap")
    d = stream.next_doc("V01")
    r = execute(best, d.text)
    assert r.anchors_ok
    assert any(k.startswith("line_items") for k in diff(r.invoice, d.truth))
    for li in r.invoice.line_items:
        assert abs(li.quantity * li.unit_price - li.amount) < 0.011


# ------------------------------------------------------------------ checks
def _inv(**kw):
    base = dict(invoice_number="A1", invoice_date="2026-01-10", due_date="2026-02-09", po_number=None,
                vendor_tax_id="300000000000003", currency="SAR", subtotal=100.0, discount=0.0, tax=15.0, total=115.0,
                line_items=[LineItem("Widget", 4, 25.0, 100.0)])
    base.update(kw)
    return Invoice(**base)


VENDOR = {"vendor_id": "V99", "country": "SA", "currency": "SAR", "tax_id": "300000000000003", "tax_rate": 0.15,
          "terms_days": 30}


def test_checks_pass_on_clean_invoice():
    rep = run_checks(_inv(), "2026-01-11T09:00:00", VENDOR, {}, None)
    assert rep.passed, rep.summary()


def test_checks_catch_arithmetic_and_master_data():
    assert not run_checks(_inv(total=116.0), "2026-01-11T09:00:00", VENDOR, {}, None).passed
    assert not run_checks(_inv(due_date="2026-02-20"), "2026-01-11T09:00:00", VENDOR, {}, None).passed  # terms
    assert not run_checks(_inv(po_number="PO-1"), "2026-01-11T09:00:00", VENDOR, {}, None).passed  # unknown PO


def test_history_catches_swap_that_arithmetic_cannot():
    h = VendorHistory()
    for i in range(10):
        h.learn(_inv(invoice_number=f"A{i}"), "2026-01-11T09:00:00")
    swapped = _inv(invoice_number="B1", line_items=[LineItem("Widget", 25.0, 4.0, 100.0)])
    rep = run_checks(swapped, "2026-01-11T09:00:00", VENDOR, {}, h)
    assert rep.passed  # invariants and master data are all satisfied...
    assert {c.name for c in rep.warns} & {"price_history"}  # ...but the vendor's history is not


def test_duplicate_invoice_detected():
    h = VendorHistory()
    h.learn(_inv(), "2026-01-11T09:00:00")
    assert "not_duplicate" in {c.name for c in run_checks(_inv(), "2026-01-11T09:00:00", VENDOR, {}, h).fails}


# ------------------------------------------------------------------ audit statistics & policy
def test_rule_of_three():
    assert error_upper_bound(0, 300) == pytest.approx(0.00994, rel=0.02)
    assert error_upper_bound(0, 0) == 1.0
    assert audit_rate(0) == 1.0 and audit_rate(1000) == 0.02


def test_policy_routes():
    assert decide(RouteInput("D1", True, False, "learning")).path == "llm"
    assert decide(RouteInput("D1", True, True, "stable", anchors_ok=False)).path == "llm"
    assert decide(RouteInput("D1", True, True, "suspect", parser_clean_audits=500)).path == "audit"
    assert decide(RouteInput("D1", True, True, "stable", value=10, currency="USD", parser_clean_audits=5000)).path == "code"


# ------------------------------------------------------------------ fingerprints
def test_vendor_identified_by_tax_id(stream):
    d = stream.next_doc("V05")
    assert identify_vendor(d.text, stream.vendor_master()) == "V05"


def test_layout_shapes_stable_across_invoices(stream):
    a, b = stream.next_doc("V03"), stream.next_doc("V03")
    assert len(doc_shapes(a.text) ^ doc_shapes(b.text)) <= 2  # at most an optional line (PO/discount) differs


# ------------------------------------------------------------------ end to end
def _inline_vendor():
    s = InvoiceStream(seed=11, vendor_overrides=DEMO_OVERRIDES)
    by_volume = [v.vendor_id for v, _ in sorted(zip(s.vendors, s.weights), key=lambda x: -x[1])]
    return next(v for v in by_volume if "inline_swap" in s.eligible_drifts(v))


def _run(mode, drifts, n=1600):
    """Realistic (scan-damaged) stream. Returns the engine and per-vendor wrong counts after each drift."""
    s = InvoiceStream(seed=11, vendor_overrides=DEMO_OVERRIDES)
    truth, damage = {}, {}
    eng = Engine(InvoiceType(s.vendor_master(), s.po_master),
                 SimulatedLLM(lambda d: truth[d], damage=lambda d: damage.get(d)), Config(mode=mode))
    wrong_after = {}
    for i in range(1, n + 1):
        if i in drifts:
            s.apply_drift(*drifts[i])
        d = s.next_doc()
        truth[d.doc_id] = d.truth
        damage[d.doc_id] = d.damage
        r = eng.process(d.doc_id, d.text, d.received_at)
        bad = diff(Invoice.from_dict(r["final"]) if r["final"] else Invoice(), d.truth)
        for at, (vid, _) in drifts.items():
            if d.vendor_id == vid and i >= at:
                wrong_after[vid] = wrong_after.get(vid, 0) + bool(bad)
    return eng, wrong_after


def test_end_to_end_silent_drift_detected_and_recovered():
    vid = _inline_vendor()
    eng, _ = _run("distill", {800: (vid, "inline_swap")})
    kinds = [(e["kind"], e["vendor_id"]) for e in eng.events if e["seq"] >= 800]
    assert ("drift", vid) in kinds and ("promoted", vid) in kinds
    first = next(e["seq"] for e in eng.events if e["seq"] >= 800 and e["kind"] == "drift" and e["vendor_id"] == vid)
    assert first <= 860  # caught within a few of this vendor's invoices


def test_distill_beats_static_baseline_on_silent_drift():
    vid = _inline_vendor()
    _, wrong_static = _run("static", {800: (vid, "inline_swap")})
    _, wrong_distill = _run("distill", {800: (vid, "inline_swap")})
    assert wrong_distill.get(vid, 0) < wrong_static.get(vid, 0)
