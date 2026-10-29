"""The Distill engine: one document at a time, end to end.

    identify vendor -> fingerprint layout -> run active parser -> business checks
      -> drift monitor -> routing policy -> (code | audit | LLM)
      -> learn (examples, history, layout) -> synthesise / shadow / promote

Modes (used by the benchmark to compare like for like):
  distill   full system
  static    "compile once" baseline: first validated parser per vendor is kept
            forever; LLM only when the parser visibly breaks (no checks gate,
            no drift monitor, no audits, no re-synthesis)
  llm_only  every document goes to the small model, escalating to the large
            model when its output fails the checks
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .audit import describe_bound, error_upper_bound
from .checks import VendorHistory, run_checks, CheckReport
from .drift import DriftMonitor
from .fingerprint import LayoutModel, doc_shapes, identify_vendor
from .llm import LLMResult
from .model import Invoice, diff
from .parser import execute
from .policy import Route, RouteInput, decide
from .registry import ParserVersion, Registry
from .synth import Example, best_candidate, generate_candidates

DSL_DOC = (__import__("distill.parser", fromlist=["__doc__"]).__doc__ or "")


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
    layout: LayoutModel = field(default_factory=LayoutModel)
    history: VendorHistory = field(default_factory=VendorHistory)
    examples: deque = field(default_factory=lambda: deque(maxlen=12))
    epoch: int = 0
    active: Optional[ParserVersion] = None
    shadow: Optional[ParserVersion] = None
    versions: int = 0
    code_pool: list = field(default_factory=list)  # doc_ids accepted by code since the last verified document
    examples_added: int = 0
    last_synth_fail_seq: int = -1  # value of examples_added when synthesis last failed


class Engine:
    def __init__(self, vendor_master: list[dict], po_master: dict[str, str], llm, config: Optional[Config] = None,
                 registry: Optional[Registry] = None):
        self.cfg = config or Config()
        self.master = vendor_master
        self.po_master = po_master
        self.llm = llm
        self.reg = registry or Registry()
        self.vendors: dict[str, VendorState] = {
            v["vendor_id"]: VendorState(master=v, monitor=DriftMonitor(v["vendor_id"])) for v in vendor_master}
        for vs in self.vendors.values():
            vs.examples = deque(maxlen=self.cfg.max_examples)
        self.seq = 0
        self.records: dict[str, dict] = {}
        self.order: list[str] = []
        self.events: list[dict] = []
        self.listeners: list = []
        self.totals = {"docs": 0, "cost_usd": 0.0, "llm_calls": 0, "code_docs": 0, "audit_docs": 0, "llm_docs": 0,
                       "human_review": 0, "quarantined": 0, "quarantine_corrected": 0, "latency_ms": 0.0,
                       "synth_cost_usd": 0.0}

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
    def _checks(self, inv: Invoice, received_at: str, vs: VendorState) -> CheckReport:
        return run_checks(inv, received_at, vs.master, self.po_master, vs.history)

    def _llm(self, rec: dict, doc_id: str, text: str, tier: str, purpose: str) -> LLMResult:
        res = self.llm.extract(doc_id, text, tier)
        rec["llm_calls"].append({"model": res.model, "tier": tier, "purpose": purpose, "in": res.input_tokens,
                                 "out": res.output_tokens, "cost_usd": res.cost_usd, "ms": res.latency_ms,
                                 "simulated": res.simulated, "error": res.error})
        rec["cost_usd"] += res.cost_usd
        rec["latency_ms"] += res.latency_ms
        self.totals["llm_calls"] += 1
        return res

    def _llm_extract_with_escalation(self, rec: dict, doc_id: str, text: str, received_at: str, vs: VendorState,
                                     first_tier: str) -> tuple[Optional[Invoice], CheckReport, str]:
        res = self._llm(rec, doc_id, text, first_tier, "extract")
        rep = self._checks(res.invoice, received_at, vs) if res.invoice else None
        tier = first_tier
        if (rep is None or not rep.passed) and first_tier == "small":
            rec["trace"].append(f"small model output failed checks ({rep.summary() if rep else res.error}) -> large model")
            res2 = self._llm(rec, doc_id, text, "large", "escalate")
            rep2 = self._checks(res2.invoice, received_at, vs) if res2.invoice else None
            if rep2 is not None and (rep is None or len(rep2.fails) <= len(rep.fails)):
                res, rep, tier = res2, rep2, "large"
        if rep is None:
            rep = CheckReport()
        return res.invoice, rep, tier

    # ------------------------------------------------------------------ main entry
    def process(self, doc_id: str, text: str, received_at: str) -> dict:
        self.seq += 1
        t0 = time.perf_counter()
        rec = {"doc_id": doc_id, "seq": self.seq, "received_at": received_at, "vendor_id": None, "source": None,
               "parser_id": None, "route": None, "final": None, "checks": [], "cost_usd": 0.0, "latency_ms": 0.0,
               "flags": [], "trace": [], "llm_calls": [], "drift_state": None, "signals": {}, "_text": text}
        vid = identify_vendor(text, self.master)
        rec["vendor_id"] = vid
        if vid is None:
            inv, rep, tier = self._unknown_vendor(rec, doc_id, text, received_at)
            return self._finish(rec, inv, rep, f"llm_{tier}", t0)

        if self.cfg.mode == "llm_only":
            vs = self.vendors[vid]
            inv, rep, tier = self._llm_extract_with_escalation(rec, doc_id, text, received_at, vs, self.cfg.llm_only_tier)
            rec["route"] = {"path": "llm", "model": self.cfg.llm_only_tier, "reasons": ["baseline: LLM on every document"]}
            self.totals["llm_docs"] += 1
            if not rep.passed:
                rec["flags"].append("human_review")
            if inv and rep.passed:
                vs.history.learn(inv, received_at)
            return self._finish(rec, inv, rep, f"llm_{tier}", t0)

        vs = self.vendors[vid]
        shapes = doc_shapes(text)
        novelty = vs.layout.novelty(shapes) if self.cfg.mode == "distill" else []
        missing = vs.layout.missing_core(shapes) if self.cfg.mode == "distill" else []

        # ---- run the active parser (if any) and check its output
        code_inv, code_rep, anchors_ok, code_ms = None, None, True, 0.0
        active = vs.active if (vs.active and vs.active.status == "active") else None
        if active:
            c0 = time.perf_counter()
            r = execute(active.spec, text)
            code_ms = (time.perf_counter() - c0) * 1000
            code_inv, anchors_ok = r.invoice, r.anchors_ok
            code_rep = self._checks(code_inv, received_at, vs)
            rec["trace"].append(f"parser {active.parser_id}: anchors {'ok' if anchors_ok else 'MISSING ' + str(r.missing_required + r.parse_errors)}; {code_rep.summary()}")
            rec["latency_ms"] += code_ms

        # ---- static baseline: trust the parser unless it visibly breaks
        if self.cfg.mode == "static":
            return self._process_static(rec, vs, doc_id, text, received_at, code_inv, anchors_ok, t0)

        # ---- drift signals
        hard, medium, soft = [], [], []
        if active:
            if not anchors_ok:
                hard.append("anchors_missing")
            hard += [c.name for c in code_rep.fails]
            soft += [c.name for c in code_rep.warns]
            if novelty:
                medium.append("novel_layout_lines")
            if missing:
                medium.append("missing_layout_lines")
            state = vs.monitor.observe(self.seq, hard, medium, soft)
            if state == "drift" and active.status == "active":
                self._on_drift(vs, rec, f"repeated hard failures ({', '.join(hard[:3]) or 'see history'})")
                active = None
        rec["signals"] = {"hard": hard, "medium": medium, "soft": soft, "novelty": novelty[:5], "missing": missing[:5]}

        # ---- route
        value = (code_inv.total if code_inv and code_inv.total else 0.0)
        route = decide(RouteInput(
            doc_id=doc_id, vendor_known=True, has_parser=active is not None, drift_state=vs.monitor.state,
            anchors_ok=anchors_ok, code_fails=[c.name for c in code_rep.fails] if code_rep else [],
            code_warns=[c.name for c in code_rep.warns] if code_rep else [], novelty=novelty + missing,
            value=value, currency=vs.master["currency"], parser_clean_audits=active.audits_clean if active else 0),
            self.cfg.profile)
        rec["route"] = route.to_dict()
        rec["trace"].append(f"route: {route.path} ({'; '.join(route.reasons)})")

        verified = False  # the final answer was confirmed independently of the parser
        if route.path == "code":
            inv, rep, source = code_inv, code_rep, "code"
            rec["parser_id"] = active.parser_id
            active.docs += 1
            vs.code_pool.append(doc_id)
            self.totals["code_docs"] += 1
        elif route.path == "audit":
            inv, rep, source, verified = self._audit(rec, vs, active, doc_id, text, received_at, code_inv, code_rep)
            self.totals["audit_docs"] += 1
        else:
            inv, rep, tier = self._llm_extract_with_escalation(rec, doc_id, text, received_at, vs, route.model)
            source = f"llm_{tier}"
            self.totals["llm_docs"] += 1
            verified = rep.passed
            if not rep.passed:
                rec["flags"].append("human_review")
            # the parser was skipped because of novelty/failed checks: was it actually wrong?
            if active and code_inv is not None and inv is not None and rep.passed:
                bad = diff(code_inv, inv)
                if bad:
                    rec["trace"].append(f"parser disagrees with verified LLM output on {bad[:4]}")
                    self._on_drift(vs, rec, f"parser output disagrees with verified extraction on {', '.join(bad[:3])}")
                    active = None
                else:
                    rec["trace"].append("parser agrees with LLM output; unseen lines were harmless")
                    verified = True

        # ---- learn from trusted results
        if inv is not None and rep.passed and source != "code":
            self._add_example(vs, doc_id, text, inv, received_at)
        if inv is not None and rep.passed:
            vs.history.learn(inv, received_at)
            if source != "code" or verified:
                vs.layout.learn(shapes)
            elif not novelty:
                vs.layout.learn(shapes)
        if verified:
            vs.code_pool.clear()

        # ---- shadow evaluation, synthesis, promotion
        if inv is not None and rep.passed and source != "code":
            self._shadow_step(vs, rec, text, inv)
        self._maybe_synthesize(vs, rec)
        if rec.get("_corrected_final") is not None:
            if inv is not None and inv.invoice_number:
                vs.history.invoice_numbers.discard(inv.invoice_number)
            inv = rec.pop("_corrected_final")
            source = "code"
            rep = self._checks(inv, received_at, vs)
            if inv.invoice_number:
                vs.history.invoice_numbers.add(inv.invoice_number)
        rec["drift_state"] = vs.monitor.state
        return self._finish(rec, inv, rep, source, t0)

    # ------------------------------------------------------------------ paths
    def _unknown_vendor(self, rec, doc_id, text, received_at):
        res = self._llm(rec, doc_id, text, "large", "unknown vendor")
        rep = run_checks(res.invoice, received_at, None, self.po_master, None) if res.invoice else CheckReport()
        rec["flags"].append("human_review")
        rec["route"] = {"path": "llm", "model": "large", "reasons": ["vendor not in master data"]}
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
            vs.history.learn(inv, received_at)
            if vs.active is None:
                self._add_example(vs, doc_id, text, inv, received_at)
                if len(vs.examples) >= self.cfg.min_examples:
                    ex = self._epoch_examples(vs)
                    best, scored = best_candidate(generate_candidates(ex), ex)
                    if best:
                        vs.versions += 1
                        vs.active = ParserVersion(f"{vs.master['vendor_id']}-p{vs.versions}", vs.master["vendor_id"],
                                                  vs.versions, best, best.get("generator", "?"), status="active",
                                                  created_seq=self.seq, promoted_seq=self.seq)
                        self._event("promoted", vs.master["vendor_id"], f"static parser {vs.active.parser_id} compiled")
        return self._finish(rec, inv, rep, f"llm_{tier}", t0)

    def _audit(self, rec, vs, active, doc_id, text, received_at, code_inv, code_rep):
        res = self._llm(rec, doc_id, text, "small", "audit")
        rec["parser_id"] = active.parser_id
        if res.invoice is not None and not diff(code_inv, res.invoice):
            active.audits_clean += 1
            active.docs += 1
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
        arb_rep = self._checks(arb.invoice, received_at, vs) if arb.invoice else CheckReport()
        active.audits_failed += 1
        if arb.invoice is not None and arb_rep.passed:
            bad = diff(code_inv, arb.invoice)
            rec["trace"].append(f"arbiter confirms parser error on {bad[:4]}")
            rec["flags"].append("parser_error_caught")
            self._on_drift(vs, rec, f"audit proved parser wrong on {', '.join(bad[:3])}")
            return arb.invoice, arb_rep, "llm_large", True
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
            res = self.llm.extract(qid, q["_text"], "small")
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
            prev = Invoice.from_dict(q["final"]) if q["final"] else None
            if rep.passed and (prev is None or diff(prev, res.invoice)):
                q["final"] = res.invoice.to_dict()
                q["source"] = "llm_small"
                q["flags"].append("corrected_by_quarantine")
                q["trace"].append(f"quarantine: corrected after drift detected at seq {self.seq}")
                corrected += 1
        self.totals["quarantine_corrected"] += corrected
        self._event("quarantine", vs.master["vendor_id"],
                    f"Quarantined {len(pool)} recent document(s); corrected {corrected}.", n=len(pool), corrected=corrected)

    # ------------------------------------------------------------------ learning
    def _add_example(self, vs: VendorState, doc_id: str, text: str, inv: Invoice, received_at: str) -> None:
        vs.examples.append((vs.epoch, Example(text=text, labels=inv, received_at=received_at, doc_id=doc_id)))
        vs.examples_added += 1

    def _epoch_examples(self, vs: VendorState) -> list[Example]:
        return [e for ep, e in vs.examples if ep == vs.epoch]

    def _maybe_synthesize(self, vs: VendorState, rec: dict) -> None:
        if self.cfg.mode != "distill" or vs.shadow is not None:
            return
        needs = vs.active is None or vs.active.status != "active"
        if not needs:
            return
        fresh = self._epoch_examples(vs)
        if len(fresh) < self.cfg.min_examples or vs.last_synth_fail_seq == vs.examples_added:
            return
        older = [e for ep, e in vs.examples if ep != vs.epoch]
        proposals, pcost = [], 0.0
        if self.cfg.use_llm_spec_proposals:
            proposals, pcost = self.llm.propose_spec(fresh, DSL_DOC)
            self.totals["synth_cost_usd"] += pcost
            self.totals["cost_usd"] += pcost
        # 1) all fresh examples (+ compatible older ones keep optional fields alive)
        attempts = [("fresh+older", fresh + older), ("fresh", fresh)]
        # 2) leave-one-out: tolerate one mislabeled example (an undetectable LLM slip)
        if len(fresh) >= self.cfg.min_examples + 1:
            for i in range(len(fresh)):
                attempts.append((f"leave-out {fresh[i].doc_id}", fresh[:i] + fresh[i + 1:]))
        # 3) most recent examples only: the layout may have changed while we were still learning
        for k in range(len(fresh) - 1, self.cfg.min_examples - 1, -1):
            attempts.append((f"latest {k}", fresh[-k:]))
        chosen = None
        for label, ex in attempts:
            cands = generate_candidates(ex, proposals)
            if not cands:
                continue
            best, scored = best_candidate(cands, ex)
            if best is None and label == "fresh+older":
                best, scored = best_candidate(cands, fresh)
            if best is not None:
                chosen = (label, best, scored, len(cands))
                break
        if chosen is None:
            vs.last_synth_fail_seq = vs.examples_added
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

    def _shadow_step(self, vs: VendorState, rec: dict, text: str, verified: Invoice) -> None:
        sh = vs.shadow
        if sh is None:
            return
        r = execute(sh.spec, text)
        bad = diff(r.invoice, verified) if r.anchors_ok else ["anchors"]
        vid = vs.master["vendor_id"]
        if bad and r.anchors_ok:
            # Who is wrong, the candidate parser or the LLM that labelled this document? Ask the arbiter.
            arb = self._llm(rec, rec["doc_id"], text, "large", "shadow arbiter")
            if arb.invoice is not None and not diff(r.invoice, arb.invoice):
                rec["trace"].append(f"shadow {sh.parser_id} disagreed on {bad[:3]}; arbiter sided with the parser "
                                    "-> LLM label corrected")
                rec["flags"].append("llm_error_corrected_by_parser")
                rec["_corrected_final"] = r.invoice
                bad = []
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
            vs.active, vs.shadow = sh, None
            self.reg.save_parser(sh)
            vs.monitor.recovered(self.seq, f"{sh.parser_id} promoted")
            vs.code_pool.clear()
            what = "replaces " + old.parser_id if old else "first parser for this vendor"
            self._event("promoted", vid, f"{sh.parser_id} promoted ({what}) after {sh.validation.get('exact')}/"
                        f"{sh.validation.get('examples')} isolated + {sh.shadow_agree} shadow agreements",
                        parser_id=sh.parser_id)

    # ------------------------------------------------------------------ manual controls
    def rollback(self, vendor_id: str) -> Optional[str]:
        """Re-activate the most recent retired parser for a vendor (dashboard button)."""
        vs = self.vendors[vendor_id]
        hist = [p for p in self.reg.parser_history(vendor_id) if p["status"] == "retired"]
        if not hist:
            return None
        pid = hist[-1]["parser_id"]
        row = self.reg.db.execute("SELECT spec FROM parsers WHERE parser_id=?", (pid,)).fetchone()
        import json as _json
        vs.versions += 1
        p = ParserVersion(f"{vendor_id}-p{vs.versions}", vendor_id, vs.versions, _json.loads(row[0]), "rollback",
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
    def _finish(self, rec: dict, inv: Optional[Invoice], rep: CheckReport, source: str, t0: float) -> dict:
        rec["final"] = inv.to_dict() if inv else None
        rec["source"] = source
        rec["checks"] = rep.to_list() if rep else []
        rec["checks_summary"] = rep.summary() if rep else "no output"
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
                "docs": a.docs if a else 0, "audits": n, "audit_errors": a.audits_failed if a else 0,
                "error_bound": error_upper_bound(a.audits_failed, n) if a and n else None,
                "bound_text": describe_bound(a.audits_failed, n) if a else "",
                "examples": len(self._epoch_examples(vs)), "versions": vs.versions,
            })
        return rows
