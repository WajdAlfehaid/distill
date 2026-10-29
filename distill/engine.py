"""The Distill engine: one document at a time, end to end, for any document type.

    identify source -> fingerprint layout -> run active parser -> checks
      -> drift monitor -> routing policy -> (code | audit | LLM)
      -> compare, and tell OCR noise from real drift
      -> learn (examples, history, layout) -> synthesise / shadow / promote

Noise versus drift
------------------
Scanned documents are damaged in ordinary ways: look-alike characters, smudged
labels, values that never made it through OCR. When the parser and a verified
extraction disagree, the engine first asks whether OCR damage explains it (the
values are one look-alike character apart, the parser had to repair or fuzzily
match that field, or a field is missing while the layout itself is unchanged).
Only disagreements that noise cannot explain count as evidence that the parser
is wrong, and only that evidence suspends a parser.

Modes (used by the benchmark to compare like for like):
  distill   full system
  static    "compile once" baseline: first validated parser per source is kept
            forever; LLM only when the parser visibly breaks
  llm_only  every document goes to a model, escalating to the large model when its
            output fails the checks
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .audit import describe_bound, error_upper_bound
from .checks import CheckReport
from .doctypes import DocType, InvoiceType
from .drift import DriftMonitor
from .fingerprint import LayoutModel, doc_shapes, structural
from .llm import LLMResult
from .model import Record, diff
from .ocr import money_ocr_equivalent, ocr_equivalent
from .parser import ExecResult, execute
from .policy import RouteInput, decide
from .registry import ParserVersion, Registry
from .synth import Example, best_candidate, generate_candidates

DSL_DOC = (__import__("distill.parser", fromlist=["__doc__"]).__doc__ or "")


def _item_close(x, y) -> bool:
    """Same line item, up to OCR damage."""
    desc_ok = ocr_equivalent(x.description, y.description) or x.description.split()[:2] == y.description.split()[:2]
    nums = [(x.quantity, y.quantity), (x.unit_price, y.unit_price), (x.amount, y.amount)]
    return desc_ok and all(abs(p - q) < 0.006 or money_ocr_equivalent(p, q) for p, q in nums)


@dataclass
class Config:
    mode: str = "distill"  # distill | static | llm_only
    profile: str = "balanced"
    min_examples: int = 3
    shadow_agreements: int = 3
    max_examples: int = 12
    use_llm_spec_proposals: bool = False
    llm_only_tier: str = "small"  # llm_only baseline: which model reads every document


@dataclass
class VendorState:
    master: dict
    monitor: DriftMonitor
    history: object
    layout: LayoutModel = field(default_factory=LayoutModel)
    examples: deque = field(default_factory=lambda: deque(maxlen=12))
    epoch: int = 0
    active: Optional[ParserVersion] = None
    shadow: Optional[ParserVersion] = None
    versions: int = 0
    code_pool: list = field(default_factory=list)  # doc_ids accepted by code since the last verified document
    examples_added: int = 0
    last_synth_fail_seq: int = -1  # value of examples_added when synthesis last failed
    docs: int = 0


class Engine:
    def __init__(self, doctype: DocType, llm, config: Optional[Config] = None, registry: Optional[Registry] = None):
        self.doctype: DocType = doctype
        self.schema = doctype.schema
        self.cfg = config or Config()
        self.llm = llm
        self.reg = registry or Registry()
        self.vendors: dict[str, VendorState] = {}
        for v in getattr(doctype, "vendor_master", []):
            self._vs(v["vendor_id"])
        self.seq = 0
        self.records: dict[str, dict] = {}
        self.order: list[str] = []
        self.events: list[dict] = []
        self.listeners: list = []
        self.revised: list[str] = []  # documents whose final result was changed after the fact (quarantine)
        self.totals = {"docs": 0, "cost_usd": 0.0, "llm_calls": 0, "code_docs": 0, "audit_docs": 0, "llm_docs": 0,
                       "human_review": 0, "quarantined": 0, "quarantine_corrected": 0, "latency_ms": 0.0,
                       "synth_cost_usd": 0.0, "noise_disagreements": 0, "llm_corrected_by_parser": 0}

    def _vs(self, key: str) -> VendorState:
        vs = self.vendors.get(key)
        if vs is None:
            vs = VendorState(master=self.doctype.master(key), monitor=DriftMonitor(key),
                             history=self.doctype.new_history())
            vs.examples = deque(maxlen=self.cfg.max_examples)
            self.vendors[key] = vs
        return vs

    # ------------------------------------------------------------------ events
    def _event(self, kind: str, vendor_id: Optional[str], message: str, **data) -> None:
        ev = {"seq": self.seq, "kind": kind, "vendor_id": vendor_id, "message": message, **data}
        self.events.append(ev)
        self.reg.log(self.seq, vendor_id, kind, message, data)
        for fn in self.listeners:
            try:
                fn(ev)
            except Exception:
                pass

    # ------------------------------------------------------------------ helpers
    def _checks(self, rec: Optional[Record], received_at: str, vs: Optional[VendorState]) -> CheckReport:
        if rec is None:
            return CheckReport()
        return self.doctype.checks(rec, received_at, vs.master if vs else None, vs.history if vs else None)

    def _llm(self, rec: dict, doc_id: str, text: str, tier: str, purpose: str) -> LLMResult:
        res = self.llm.extract(doc_id, text, tier, self.schema)
        rec["llm_calls"].append({"model": res.model, "tier": tier, "purpose": purpose, "in": res.input_tokens,
                                 "out": res.output_tokens, "cost_usd": res.cost_usd, "ms": res.latency_ms,
                                 "simulated": res.simulated, "error": res.error})
        rec["cost_usd"] += res.cost_usd
        rec["latency_ms"] += res.latency_ms
        self.totals["llm_calls"] += 1
        return res

    def _llm_extract_with_escalation(self, rec, doc_id, text, received_at, vs, first_tier):
        res = self._llm(rec, doc_id, text, first_tier, "extract")
        rep = self._checks(res.invoice, received_at, vs) if res.invoice else None
        tier = first_tier
        if (rep is None or not rep.passed) and first_tier == "small":
            rec["trace"].append(f"small model output failed checks ({rep.summary() if rep else res.error}) -> large model")
            res2 = self._llm(rec, doc_id, text, "large", "escalate")
            rep2 = self._checks(res2.invoice, received_at, vs) if res2.invoice else None
            if rep2 is not None and (rep is None or len(rep2.fails) <= len(rep.fails)):
                res, rep, tier = res2, rep2, "large"
        return res.invoice, rep or CheckReport(), tier

    def _classify(self, bad: list[str], code: Record, verified: Record, er: Optional[ExecResult],
                  layout_changed: bool, code_failed: bool = False) -> tuple[list[str], list[str]]:
        """Split disagreements into (real, noise). Noise = explained by OCR damage on an unchanged layout.

        When the parser's own output already failed the business checks on an unchanged layout,
        the copy is visibly damaged: a misread there is noise. A wrong parser shows up instead as
        clean, check-passing disagreements, or as the same field disagreeing copy after copy
        (see _systematic)."""
        if code_failed and not layout_changed:
            return [], list(bad)
        real, noise = [], []
        types = self.schema.types
        items_noise = None
        for f in bad:
            if f.startswith("line_items"):
                if items_noise is None:
                    items_noise = self._items_noise(code, verified, er, layout_changed)
                (noise if items_noise else real).append(f)
                continue
            cv, vv = code.values.get(f), verified.values.get(f)
            if er and (f in er.fuzzy or f in er.ocr_fixed):
                noise.append(f)
            elif cv is None or vv is None:
                (real if layout_changed else noise).append(f)
            elif types.get(f) == "money":
                (noise if money_ocr_equivalent(cv, vv) else real).append(f)
            elif ocr_equivalent(cv, vv):
                noise.append(f)
            else:
                real.append(f)
        return real, noise

    def _systematic(self, vs: VendorState, bad: list[str], window: int = 6, need: int = 3) -> Optional[str]:
        """Record which fields disagreed on this copy; return a field that disagreed on most recent copies."""
        hist = getattr(vs, "disagreements", None)
        if hist is None:
            hist = vs.disagreements = deque(maxlen=window)
        hist.append(set(bad))
        counts = {}
        for fs in hist:
            for f in fs:
                counts[f] = counts.get(f, 0) + 1
        hot = [f for f, c in counts.items() if c >= need]
        return hot[0] if hot else None

    @staticmethod
    def _items_noise(code: Record, verified: Record, er: Optional[ExecResult], layout_changed: bool) -> bool:
        """Line-item differences explained by a damaged scan rather than a wrong parser.

        Noise: on an unchanged layout, the parser read fewer rows (a damaged row stopped matching),
        or every differing value is a look-alike/one-slip misread. Real: rows read cleanly with
        different values (e.g. quantity and price swapped), or the layout itself changed.
        """
        if layout_changed:
            return False
        a, b = code.line_items, verified.line_items
        if er and "line_items" in er.ocr_fixed:
            return True
        if len(a) < len(b):
            # every row the parser did read must agree with one of the verified rows
            return all(any(_item_close(x, y) for y in b) for x in a)
        if len(a) != len(b):
            return False
        return all(_item_close(x, y) for x, y in zip(a, b))

    # ------------------------------------------------------------------ main entry
    def process(self, doc_id: str, text: str, received_at: str, meta: Optional[dict] = None) -> dict:
        self.seq += 1
        t0 = time.perf_counter()
        rec = {"doc_id": doc_id, "seq": self.seq, "received_at": received_at, "vendor_id": None, "source": None,
               "parser_id": None, "route": None, "final": None, "checks": [], "cost_usd": 0.0, "latency_ms": 0.0,
               "flags": [], "trace": [], "llm_calls": [], "drift_state": None, "signals": {}, "_text": text,
               "_words": (meta or {}).get("words")}
        words = rec["_words"]
        vid = self.doctype.identify(text, meta)
        rec["vendor_id"] = vid
        if vid is None:
            inv, rep, tier = self._unknown_vendor(rec, doc_id, text, received_at)
            return self._finish(rec, inv, rep, f"llm_{tier}", t0)
        vs = self._vs(vid)
        vs.docs += 1

        if self.cfg.mode == "llm_only":
            inv, rep, tier = self._llm_extract_with_escalation(rec, doc_id, text, received_at, vs, self.cfg.llm_only_tier)
            rec["route"] = {"path": "llm", "model": self.cfg.llm_only_tier, "reasons": ["baseline: LLM on every document"]}
            self.totals["llm_docs"] += 1
            if not rep.passed:
                rec["flags"].append("human_review")
            if inv is not None and rep.passed:
                self.doctype.learn(vs.history, inv, received_at)
            return self._finish(rec, inv, rep, f"llm_{tier}", t0)

        shapes = doc_shapes(text)
        novelty = vs.layout.novelty(shapes) if self.cfg.mode == "distill" else []
        missing = vs.layout.missing_core(shapes) if self.cfg.mode == "distill" else []
        # A changed template replaces label lines: new label-like lines appear AND familiar ones vanish.
        # A damaged copy usually does only one of the two, and prose notes are not labels at all.
        new_labels = [x for x in novelty if structural(x)]
        layout_changed = bool(new_labels and missing) or len(missing) >= 2

        # ---- run the active parser (if any) and check its output
        code_inv, code_rep, anchors_ok, er = None, None, True, None
        active = vs.active if (vs.active and vs.active.status == "active") else None
        if active:
            c0 = time.perf_counter()
            er = execute(active.spec, text, self.schema, words)
            rec["latency_ms"] += (time.perf_counter() - c0) * 1000
            code_inv, anchors_ok = er.record, er.anchors_ok
            code_rep = self._checks(code_inv, received_at, vs)
            notes = []
            if er.fuzzy:
                notes.append(f"damaged label(s) recovered: {er.fuzzy}")
            if er.ocr_fixed:
                notes.append(f"look-alike characters repaired in {er.ocr_fixed}")
            rec["trace"].append(f"parser {active.parser_id}: anchors {'ok' if anchors_ok else 'MISSING ' + str(er.missing_required + er.parse_errors)}"
                                f"; {code_rep.summary()}" + (f"; {'; '.join(notes)}" if notes else ""))

        if self.cfg.mode == "static":
            return self._process_static(rec, vs, doc_id, text, received_at, code_inv, anchors_ok, t0)

        # ---- drift signals (hard = the parser looks wrong on a clean copy; soft = noise or warnings)
        hard, medium, soft = [], [], []
        if active:
            noisy_copy = bool(er.fuzzy or er.ocr_fixed)
            if not anchors_ok:
                (hard if layout_changed else soft).append("anchors_missing" if layout_changed else "unreadable_field")
            for c in code_rep.fails:
                (soft if noisy_copy and not layout_changed else hard).append(c.name)
            soft += [c.name for c in code_rep.warns]
            if novelty:
                medium.append("novel_layout_lines")
            if missing:
                medium.append("missing_layout_lines")
            if vs.monitor.observe(self.seq, hard, medium, soft) == "drift" and active.status == "active":
                self._on_drift(vs, rec, f"repeated hard failures ({', '.join(hard[:3]) or 'see history'})")
                active, code_inv, code_rep, anchors_ok = None, None, None, True
        rec["signals"] = {"hard": hard, "medium": medium, "soft": soft, "novelty": novelty[:5], "missing": missing[:5]}

        # ---- route
        value, currency = self.doctype.value(code_inv, vs.master)
        route = decide(RouteInput(
            doc_id=doc_id, vendor_known=True, has_parser=active is not None, drift_state=vs.monitor.state,
            anchors_ok=anchors_ok, code_fails=[c.name for c in code_rep.fails] if code_rep else [],
            code_warns=[c.name for c in code_rep.warns] if code_rep else [], novelty=novelty + missing,
            value=value, currency=currency, parser_clean_audits=active.audits_clean if active else 0),
            self.cfg.profile)
        rec["route"] = route.to_dict()
        rec["trace"].append(f"route: {route.path} ({'; '.join(route.reasons)})")

        verified = False  # the final answer was confirmed independently of the parser
        if route.path == "code":
            inv, rep, source = code_inv, code_rep, "code"
            if not code_rep.warns:
                self._systematic(vs, [])  # a clean copy is evidence the parser is still fine
            rec["parser_id"] = active.parser_id
            active.docs += 1
            vs.code_pool.append(doc_id)
            self.totals["code_docs"] += 1
        elif route.path == "audit":
            inv, rep, source, verified = self._audit(rec, vs, active, doc_id, text, received_at, code_inv, code_rep,
                                                     er, layout_changed)
            self.totals["audit_docs"] += 1
        else:
            inv, rep, tier = self._llm_extract_with_escalation(rec, doc_id, text, received_at, vs, route.model)
            source = f"llm_{tier}"
            self.totals["llm_docs"] += 1
            verified = rep.passed
            if not rep.passed:
                rec["flags"].append("human_review")
            if active and code_inv is not None and inv is not None and rep.passed:
                inv, rep, source, verified = self._compare_with_parser(rec, vs, active, doc_id, text, code_inv,
                                                                       code_rep, anchors_ok, er, inv, rep, source,
                                                                       verified, layout_changed)

        # ---- learn from trusted results
        if inv is not None and rep.passed and source != "code":
            self._add_example(vs, doc_id, text, inv, received_at, words)
        if inv is not None and rep.passed:
            self.doctype.learn(vs.history, inv, received_at)
            if source != "code" or verified or not novelty:
                vs.layout.learn(shapes)
        if verified:
            vs.code_pool.clear()

        # ---- shadow evaluation, synthesis, promotion
        if inv is not None and rep.passed and source != "code":
            self._shadow_step(vs, rec, text, inv)
        self._maybe_synthesize(vs, rec)
        if rec.get("_corrected_final") is not None:
            inv = rec.pop("_corrected_final")
            source = "code"
            rep = self._checks(inv, received_at, vs)
            rep.checks = [c for c in rep.checks if c.name != "not_duplicate"]
        rec["drift_state"] = vs.monitor.state
        return self._finish(rec, inv, rep, source, t0)

    def _compare_with_parser(self, rec, vs, active, doc_id, text, code_inv, code_rep, anchors_ok, er, inv, rep,
                             source, verified, layout_changed):
        """The document went to a model although a parser exists. Was the parser wrong, or was this copy noisy?"""
        bad = diff(code_inv, inv)
        if not bad:
            self._systematic(vs, [])
            rec["trace"].append("parser agrees with the verified extraction; the unusual signals were harmless")
            return inv, rep, source, True
        real, noise = self._classify(bad, code_inv, inv, er, layout_changed, bool(code_rep.fails))
        if not real:
            sys_f = self._systematic(vs, bad)
            if sys_f:
                real = [sys_f]
        else:
            self._systematic(vs, bad)
        rules = set(active.spec.get("fields", {}))
        unknown = [f for f in real if not f.startswith("line_items") and f not in rules]
        if real and len(unknown) == len(real):
            # The parser never learned these fields (e.g. the first discount this vendor ever printed):
            # it is incomplete, not wrong. Keep it and learn an extended version in the background.
            vs.needs_extension = True
            rec["trace"].append(f"parser has no rule for {unknown}; it keeps running while an extended version is learned")
            self._event("extend", vs.master["vendor_id"], f"{active.parser_id} has never seen {', '.join(unknown)}; "
                        "learning an extended version")
            return inv, rep, source, verified
        if real:
            # Before blaming the parser, ask the arbiter: the model that read this document may have slipped.
            arb = self._llm(rec, doc_id, text, "large", "arbiter")
            if arb.invoice is not None:
                still = [f for f in diff(code_inv, arb.invoice) if f in real]
                if not still:
                    rec["trace"].append(f"model disagreed with parser on {real[:4]}; arbiter sided with the parser")
                    rec["flags"].append("llm_error_corrected_by_parser")
                    self.totals["llm_corrected_by_parser"] += 1
                    if not code_rep.fails:
                        return code_inv, code_rep, "code", True
                    real = []
        if real:
            rec["trace"].append(f"parser disagrees with the verified extraction on {real[:4]} (not explained by OCR damage)")
            self._on_drift(vs, rec, f"parser output disagrees with verified extraction on {', '.join(real[:3])}")
        else:
            self.totals["noise_disagreements"] += 1
            rec["flags"].append("ocr_noise")
            rec["trace"].append(f"parser misread this copy on {noise[:4]}: OCR damage, not drift; the model's reading is used")
        return inv, rep, source, verified

    # ------------------------------------------------------------------ paths
    def _unknown_vendor(self, rec, doc_id, text, received_at):
        res = self._llm(rec, doc_id, text, "large", "unknown vendor")
        rep = self._checks(res.invoice, received_at, None)
        rec["flags"].append("human_review")
        rec["route"] = {"path": "llm", "model": "large", "reasons": ["source not recognised"]}
        self.totals["llm_docs"] += 1
        return res.invoice, rep, "large"

    def _process_static(self, rec, vs, doc_id, text, received_at, code_inv, anchors_ok, t0):
        if vs.active and anchors_ok:
            rec["route"] = {"path": "code", "reasons": ["static parser"]}
            vs.active.docs += 1
            self.totals["code_docs"] += 1
            rep = self._checks(code_inv, received_at, vs)
            return self._finish(rec, code_inv, rep, "code", t0)
        inv, rep, tier = self._llm_extract_with_escalation(rec, doc_id, text, received_at, vs, "small")
        rec["route"] = {"path": "llm", "reasons": ["no parser" if not vs.active else "parser broke"]}
        self.totals["llm_docs"] += 1
        if inv is not None and rep.passed:
            self.doctype.learn(vs.history, inv, received_at)
            if vs.active is None:
                self._add_example(vs, doc_id, text, inv, received_at, rec.get("_words"))
                ex = self._epoch_examples(vs)
                if len(ex) >= self.cfg.min_examples and vs.last_synth_fail_seq != vs.examples_added:
                    best, _ = best_candidate(generate_candidates(ex), ex)
                    if best:
                        vs.versions += 1
                        vs.active = ParserVersion(f"{vs.master['vendor_id']}-p{vs.versions}", vs.master["vendor_id"],
                                                  vs.versions, best, best.get("generator", "?"), status="active",
                                                  created_seq=self.seq, promoted_seq=self.seq)
                        self._event("promoted", vs.master["vendor_id"], f"static parser {vs.active.parser_id} compiled")
                    else:
                        vs.last_synth_fail_seq = vs.examples_added
        return self._finish(rec, inv, rep, f"llm_{tier}", t0)

    def _audit(self, rec, vs, active, doc_id, text, received_at, code_inv, code_rep, er, layout_changed):
        res = self._llm(rec, doc_id, text, "small", "audit")
        rec["parser_id"] = active.parser_id
        if res.invoice is not None and not diff(code_inv, res.invoice):
            active.audits_clean += 1
            active.docs += 1
            self._systematic(vs, [])
            rec["trace"].append(f"audit: small model agrees with parser ({describe_bound(active.audits_failed, active.audits_clean + active.audits_failed)})")
            return code_inv, code_rep, "code", True
        bad = diff(code_inv, res.invoice) if res.invoice else ["llm_error"]
        rec["trace"].append(f"audit: small model disagrees on {bad[:4]} -> arbitrating with large model")
        arb = self._llm(rec, doc_id, text, "large", "arbiter")
        if arb.invoice is not None and not diff(code_inv, arb.invoice):
            active.audits_clean += 1
            active.docs += 1
            rec["trace"].append("arbiter sides with the parser (small model slipped)")
            return code_inv, code_rep, "code", True
        arb_rep = self._checks(arb.invoice, received_at, vs)
        if arb.invoice is not None and arb_rep.passed:
            bad = diff(code_inv, arb.invoice)
            real, noise = self._classify(bad, code_inv, arb.invoice, er, layout_changed, bool(code_rep.fails))
            if not real:
                sys_f = self._systematic(vs, bad)
                if sys_f:
                    real = [sys_f]
            else:
                self._systematic(vs, bad)
            if real:
                active.audits_failed += 1
                rec["trace"].append(f"arbiter confirms parser error on {real[:4]}")
                rec["flags"].append("parser_error_caught")
                self._on_drift(vs, rec, f"audit proved parser wrong on {', '.join(real[:3])}")
            else:
                self.totals["noise_disagreements"] += 1
                rec["flags"].append("ocr_noise")
                rec["trace"].append(f"parser misread this damaged copy ({noise[:4]}); counted as noise, not a parser error")
            return arb.invoice, arb_rep, "llm_large", True
        active.audits_failed += 1
        rec["flags"].append("human_review")
        return arb.invoice or code_inv, arb_rep if arb.invoice else code_rep, "llm_large", False

    # ------------------------------------------------------------------ drift handling
    def _on_drift(self, vs: VendorState, rec: dict, reason: str) -> None:
        vid = vs.master["vendor_id"]
        if vs.active and vs.active.status == "active":
            vs.active.status = "suspended"
            vs.active.retire_reason = reason
            self.reg.save_parser(vs.active)
        vs.monitor.confirm_drift(self.seq, reason)
        vs.epoch += 1
        vs.layout.reset()
        vs.shadow = None
        self._event("drift", vid, f"Drift confirmed for {vs.master['name']}: {reason}. Parser suspended.",
                    parser_id=vs.active.parser_id if vs.active else None)
        rec["flags"].append("drift_confirmed")
        self._quarantine(vs, rec)

    def _quarantine(self, vs: VendorState, rec: dict) -> None:
        """Re-check every document the parser accepted since the last verified one."""
        pool, vs.code_pool = list(vs.code_pool), []
        if not pool:
            return
        corrected = 0
        for qid in pool:
            q = self.records.get(qid)
            if q is None:
                continue
            res = self.llm.extract(qid, q["_text"], "small", self.schema)
            q["llm_calls"].append({"model": res.model, "tier": "small", "purpose": "quarantine re-check",
                                   "in": res.input_tokens, "out": res.output_tokens, "cost_usd": res.cost_usd,
                                   "ms": res.latency_ms, "simulated": res.simulated})
            q["cost_usd"] += res.cost_usd
            self.totals["cost_usd"] += res.cost_usd
            self.totals["llm_calls"] += 1
            self.totals["quarantined"] += 1
            q["flags"].append("quarantined")
            if res.invoice is None:
                continue
            rep = self._checks(res.invoice, q["received_at"], vs)
            rep.checks = [c for c in rep.checks if c.name != "not_duplicate"]
            prev = Record.from_dict(q["final"], self.schema) if q["final"] else None
            if rep.passed and (prev is None or diff(prev, res.invoice)):
                q["final"] = res.invoice.to_dict()
                q["source"] = "llm_small"
                q["flags"].append("corrected_by_quarantine")
                self.revised.append(qid)
                q["trace"].append(f"quarantine: corrected after drift detected at #{self.seq}")
                corrected += 1
        self.totals["quarantine_corrected"] += corrected
        self._event("quarantine", vs.master["vendor_id"],
                    f"Quarantined {len(pool)} recent document(s); corrected {corrected}.", n=len(pool), corrected=corrected)

    # ------------------------------------------------------------------ learning
    def _add_example(self, vs: VendorState, doc_id: str, text: str, inv: Record, received_at: str,
                     words: Optional[list] = None) -> None:
        vs.examples.append((vs.epoch, Example(text=text, labels=inv, received_at=received_at, doc_id=doc_id, words=words)))
        vs.examples_added += 1

    def _epoch_examples(self, vs: VendorState) -> list[Example]:
        return [e for ep, e in vs.examples if ep == vs.epoch]

    def _maybe_synthesize(self, vs: VendorState, rec: dict) -> None:
        if self.cfg.mode != "distill" or vs.shadow is not None:
            return
        extending = vs.active is not None and vs.active.status == "active" and getattr(vs, "needs_extension", False)
        if vs.active is not None and vs.active.status == "active" and not extending:
            return
        fresh = self._epoch_examples(vs)
        if len(fresh) < self.cfg.min_examples:
            return
        # after a failed attempt, wait for more evidence: retry after 1, 2, 4, 8 ... new examples (max 8)
        if vs.last_synth_fail_seq >= 0:
            wait = min(8, max(1, getattr(vs, "synth_failures", 1)))
            if vs.examples_added - vs.last_synth_fail_seq < wait:
                return
        older = [e for ep, e in vs.examples if ep != vs.epoch]
        proposals, pcost = [], 0.0
        if self.cfg.use_llm_spec_proposals:
            proposals, pcost = self.llm.propose_spec(fresh, DSL_DOC)
            self.totals["synth_cost_usd"] += pcost
            self.totals["cost_usd"] += pcost
        attempts = [("fresh+older", fresh + older)] if older else []
        attempts.append(("fresh", fresh))
        if len(fresh) >= self.cfg.min_examples + 1:  # tolerate one mislabelled or unreadable example
            for i in range(len(fresh)):
                attempts.append((f"leave-out {fresh[i].doc_id}", fresh[:i] + fresh[i + 1:]))
        for k in range(len(fresh) - 1, self.cfg.min_examples - 1, -1):  # the layout may have moved while learning
            attempts.append((f"latest {k}", fresh[-k:]))
        chosen = None
        for label, ex in attempts:
            cands = generate_candidates(ex, proposals)
            if not cands:
                continue
            best, scored = best_candidate(cands, ex)
            if best is None and label == "fresh+older":
                best, scored = best_candidate(cands, fresh)
            if best is not None and self._covers_schema(best, ex):
                chosen = (label, best, scored, len(cands))
                break
        if chosen is None:
            vs.last_synth_fail_seq = vs.examples_added
            vs.synth_failures = getattr(vs, "synth_failures", 0) * 2 or 1
            rec["trace"].append(f"synthesis: no candidate fits all {len(fresh)} fresh examples yet")
            return
        label, best, scored, n_cands = chosen
        vid = vs.master["vendor_id"]
        vs.versions += 1
        parent = vs.active.parser_id if vs.active else None
        v = scored[0][1]
        p = ParserVersion(f"{vid}-p{vs.versions}", vid, vs.versions, best, best.get("generator", "?"), status="shadow",
                          parent_id=parent, created_seq=self.seq,
                          validation={"examples": v.n, "exact": v.exact, "candidates": n_cands, "basis": label})
        if label.startswith("leave-out"):
            p.validation["suspect_label"] = label.split()[-1]
        vs.shadow = p
        self.reg.save_parser(p)
        self._event("candidate", vid, f"{p.parser_id} synthesized ({best.get('generator')}), validated {v.exact}/{v.n} "
                    f"in isolation from {n_cands} candidates; now in shadow mode", parser_id=p.parser_id)
        rec["trace"].append(f"synthesis: {p.parser_id} -> shadow ({label}, {v.exact}/{v.n} exact)")

    def _covers_schema(self, spec: dict, examples: list) -> bool:
        """A parser must extract every field that all its examples carry; otherwise its output is incomplete."""
        have = set(spec.get("fields", {}))
        for f in self.schema.fields:
            present_everywhere = all(ex.labels.values.get(f.name) not in (None, "") for ex in examples)
            if f.name in ("discount", "po_number"):
                continue
            if present_everywhere and f.name not in have:
                return False
        return True

    def _shadow_step(self, vs: VendorState, rec: dict, text: str, verified: Record) -> None:
        sh = vs.shadow
        if sh is None:
            return
        r = execute(sh.spec, text, self.schema, rec.get("_words"))
        vid = vs.master["vendor_id"]
        if not r.anchors_ok and not rec.get("signals", {}).get("novelty") and vs.layout.n_docs:
            rec["trace"].append(f"shadow {sh.parser_id} could not read this damaged copy; not held against it")
            return
        bad = diff(r.record, verified) if r.anchors_ok else ["anchors"]
        if bad and r.anchors_ok:
            # Who is wrong, the candidate parser or the model that labelled this document? Ask the arbiter.
            arb = self._llm(rec, rec["doc_id"], text, "large", "shadow arbiter")
            if arb.invoice is not None and not diff(r.record, arb.invoice):
                rec["trace"].append(f"shadow {sh.parser_id} disagreed on {bad[:3]}; arbiter sided with the parser "
                                    "-> model's label corrected")
                rec["flags"].append("llm_error_corrected_by_parser")
                self.totals["llm_corrected_by_parser"] += 1
                rec["_corrected_final"] = r.record
                bad = []
            elif arb.invoice is not None:
                real, noise = self._classify(diff(r.record, arb.invoice), r.record, arb.invoice, r, False)
                if not real:
                    rec["trace"].append(f"shadow {sh.parser_id} misread this damaged copy ({noise[:3]}); not held against it")
                    return
        if bad:
            sh.shadow_disagree += 1
            sh.status = "rejected"
            sh.retired_seq = self.seq
            sh.retire_reason = f"shadow disagreed on {bad[:3]}"
            self.reg.save_parser(sh)
            vs.shadow = None
            vs.last_synth_fail_seq = -1
            self._event("rejected", vid, f"{sh.parser_id} rejected in shadow (disagreed on {', '.join(bad[:3])}); re-synthesizing")
            rec["trace"].append(f"shadow {sh.parser_id} disagreed on {bad[:3]} -> rejected")
            return
        sh.shadow_agree += 1
        rec["trace"].append(f"shadow {sh.parser_id} agrees ({sh.shadow_agree}/{self.cfg.shadow_agreements})")
        if sh.shadow_agree >= self.cfg.shadow_agreements:
            old = vs.active
            if old is not None:
                old.status = "retired"
                old.retired_seq = self.seq
                old.retire_reason = old.retire_reason or f"replaced by {sh.parser_id}"
                self.reg.save_parser(old)
            sh.status = "active"
            sh.promoted_seq = self.seq
            vs.needs_extension = False
            sh.audits_clean = sh.shadow_agree  # shadow agreements are pre-promotion audits
            vs.active, vs.shadow = sh, None
            self.reg.save_parser(sh)
            vs.monitor.recovered(self.seq, f"{sh.parser_id} promoted")
            vs.disagreements = deque(maxlen=6)
            vs.code_pool.clear()
            what = "replaces " + old.parser_id if old else "first parser for this source"
            self._event("promoted", vid, f"{sh.parser_id} promoted ({what}) after {sh.validation.get('exact')}/"
                        f"{sh.validation.get('examples')} isolated + {sh.shadow_agree} shadow agreements",
                        parser_id=sh.parser_id)

    # ------------------------------------------------------------------ manual controls
    def rollback(self, vendor_id: str) -> Optional[str]:
        """Re-activate the most recent retired parser for a source (dashboard button)."""
        vs = self.vendors[vendor_id]
        hist = [p for p in self.reg.parser_history(vendor_id) if p["status"] == "retired"]
        if not hist:
            return None
        pid = hist[-1]["parser_id"]
        row = self.reg.db.execute("SELECT spec FROM parsers WHERE parser_id=?", (pid,)).fetchone()
        vs.versions += 1
        p = ParserVersion(f"{vendor_id}-p{vs.versions}", vendor_id, vs.versions, json.loads(row[0]), "rollback",
                          status="active", parent_id=pid, created_seq=self.seq, promoted_seq=self.seq)
        if vs.active:
            vs.active.status = "retired"
            vs.active.retire_reason = "manual rollback"
            self.reg.save_parser(vs.active)
        vs.active, vs.shadow = p, None
        self.reg.save_parser(p)
        vs.monitor.recovered(self.seq, "manual rollback")
        self._event("rollback", vendor_id, f"Rolled back to a copy of {pid} as {p.parser_id}")
        return p.parser_id

    # ------------------------------------------------------------------ bookkeeping
    def _finish(self, rec: dict, inv: Optional[Record], rep: CheckReport, source: str, t0: float) -> dict:
        rec["final"] = inv.to_dict() if inv is not None else None
        rec["source"] = source
        rec["checks"] = rep.to_list() if rep else []
        rec["checks_summary"] = rep.summary() if rep else "no output"
        rec["checks_passed"] = rep.passed if rep else False
        rec["wall_ms"] = (time.perf_counter() - t0) * 1000
        if "human_review" in rec["flags"]:
            self.totals["human_review"] += 1
        self.totals["docs"] += 1
        self.totals["cost_usd"] += rec["cost_usd"]
        self.totals["latency_ms"] += rec["latency_ms"]
        self.records[rec["doc_id"]] = rec
        self.order.append(rec["doc_id"])
        self.reg.save_doc(rec)
        return rec

    def parser_table(self) -> list[dict]:
        rows = []
        for vid, vs in self.vendors.items():
            a = vs.active
            n = (a.audits_clean + a.audits_failed) if a else 0
            rows.append({
                "vendor_id": vid, "vendor": vs.master["name"], "state": vs.monitor.state,
                "active": a.parser_id if a and a.status == "active" else None,
                "shadow": vs.shadow.parser_id if vs.shadow else None,
                "generator": a.generator if a else None,
                "docs": a.docs if a else 0, "seen": vs.docs, "audits": n, "audit_errors": a.audits_failed if a else 0,
                "error_bound": error_upper_bound(a.audits_failed, n) if a and n else None,
                "bound_text": describe_bound(a.audits_failed, n) if a else "",
                "examples": len(self._epoch_examples(vs)), "versions": vs.versions,
            })
        return rows
