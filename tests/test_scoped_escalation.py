from distill.checks import Check, CheckReport
from distill.engine import escalation_scope
from distill.model import Invoice, LineItem


def _rep(*names):
    return CheckReport(checks=[Check(n, "invariant", False) for n in names])


def _inv():
    return Invoice(line_items=[LineItem("a", 2, 5.0, 10.0), LineItem("b", 3, 4.0, 13.0)])


def test_totals_failure_asks_only_for_totals():
    assert escalation_scope(_rep("grand_total"), _inv()) == (["subtotal", "discount", "tax", "total"], [])


def test_broken_line_is_reread_alone():
    fields, rows = escalation_scope(_rep("line_math", "subtotal_sum"), _inv())
    assert rows == [1] and fields == ["subtotal"]


def test_unknown_check_needs_full_read():
    assert escalation_scope(_rep("has_line_items"), _inv()) == ([], [])


def test_simulated_scoped_answer_is_small():
    from distill.llm import SimulatedLLM
    inv = _inv()
    inv.total = 23.0
    llm = SimulatedLLM(lambda d: inv, profiles={"large": {"p_err": 0, "lat": 1, "per_tok": 0, "recover": 1}})
    full = llm.extract("d", "text", "large")
    part = llm.extract("d", "text", "large", fields=["total", "line_items[1]"])
    assert part.invoice.total == 23.0 and len(part.invoice.line_items) == 1
    assert part.output_tokens < full.output_tokens
