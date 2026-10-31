"""Benchmark: the same drifting invoice stream through four systems.

  llm_only_large   every document read by the large model (what many teams ship first)
  llm_only_small   every document read by the small model, large model when checks fail
  static           compile-once parsers (Evaporate-style), LLM only when a parser visibly breaks
  distill          the full system

Usage
  python -m distill.bench                     # simulated LLM tiers, 8,000 documents
  python -m distill.bench --n 300 --real      # real Claude calls (needs ANTHROPIC_API_KEY, costs money)
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .datagen import DEMO_OVERRIDES, DRIFT_KINDS, InvoiceStream
from .doctypes import InvoiceType
from .engine import Config, Engine
from .llm import AnthropicLLM, SimulatedLLM
from .model import Invoice, diff, field_count

DEFAULT_SCHEDULE = [
    # (doc number, vendor, drift kind) - after every vendor has had time to get a parser
    (1500, "V14", "inline_swap"),    # silent: price and quantity swap places
    (1800, "V18", "date_locale"),    # silent: DD/MM -> MM/DD
    (2100, "V20", "relabel"),        # loud
    (2400, "V11", "table_reflow"),   # loud
    (2700, "V10", "number_style"),   # loud
    (3000, "V12", "new_fields"),     # structural
]

def auto_schedule(seed: int, n: int) -> list:
    """Pick, for each drift kind, the busiest vendor it applies to (works for any seed)."""
    st = InvoiceStream(seed=seed, vendor_overrides=DEMO_OVERRIDES)
    by_volume = [v.vendor_id for v, _ in sorted(zip(st.vendors, st.weights), key=lambda x: -x[1])]
    plan, used = [], set()
    for frac, kind in [(0.1875, "inline_swap"), (0.225, "date_locale"), (0.2625, "relabel"),
                       (0.30, "table_reflow"), (0.3375, "number_style"), (0.375, "new_fields")]:
        vid = next((v for v in by_volume if v not in used and kind in st.eligible_drifts(v)), None)
        if vid:
            used.add(vid)
            plan.append((int(frac * n), vid, kind))
    return plan


SYSTEMS = {
    "llm_only_large": Config(mode="llm_only", llm_only_tier="large"),
    "llm_only_small": Config(mode="llm_only", llm_only_tier="small"),
    # ablation: the small-model baseline with Distill's invariant repair, to show what compiling adds on top
    "llm_only_small_repair": Config(mode="llm_only", llm_only_tier="small", repair_in_baseline=True),
    "static": Config(mode="static"),
    "distill": Config(mode="distill"),
}


def run_system(name: str, cfg: Config, n: int, seed: int, schedule: list, real: bool = False, progress: bool = False,
               realistic: bool = True) -> dict:
    stream = InvoiceStream(seed=seed, vendor_overrides=DEMO_OVERRIDES, realistic=realistic)
    truth: dict[str, Invoice] = {}
    damage: dict[str, dict] = {}
    llm = AnthropicLLM() if real else SimulatedLLM(lambda d: truth[d], seed=seed, damage=lambda d: damage.get(d))
    eng = Engine(InvoiceType(stream.vendor_master(), stream.po_master), llm, cfg)
    sched = {k: (v, kind) for k, v, kind in schedule if k <= n}
    docs, curve, drift_meta = [], [], {}
    t0 = time.perf_counter()
    for i in range(1, n + 1):
        if i in sched:
            vid, kind = sched[i]
            desc = stream.apply_drift(vid, kind)
            drift_meta[vid] = {"at": i, "kind": kind, "class": DRIFT_KINDS[kind], "desc": desc}
        d = stream.next_doc()
        truth[d.doc_id] = d.truth
        damage[d.doc_id] = d.damage
        docs.append(d)
        eng.process(d.doc_id, d.text, d.received_at)
        if i % 50 == 0:
            curve.append({"doc": i, "cost_usd": round(eng.totals["cost_usd"], 6), "llm_calls": eng.totals["llm_calls"]})
        if progress and i % 100 == 0:
            print(f"  {name}: {i}/{n} docs, ${eng.totals['cost_usd']:.4f}", flush=True)
    wall = time.perf_counter() - t0

    wrong_docs, wrong_fields, total_fields, lat = 0, 0, 0, []
    repaired, repaired_wrong = 0, 0
    per_vendor_after = {vid: {"docs": 0, "wrong": 0, "first_wrong_accept": None} for vid in drift_meta}
    for d in docs:
        r = eng.records[d.doc_id]
        pred = Invoice.from_dict(r["final"]) if r["final"] else Invoice()
        bad = diff(pred, d.truth)
        total_fields += field_count(d.truth)
        wrong_fields += len(bad) if "line_items" not in bad else len(bad) + 3 * len(d.truth.line_items)
        if bad:
            wrong_docs += 1
        if "repaired" in r["flags"]:
            repaired += 1
            repaired_wrong += bool(bad)
        lat.append(r["latency_ms"])
        m = drift_meta.get(d.vendor_id)
        if m and r["seq"] >= m["at"]:
            pv = per_vendor_after[d.vendor_id]
            pv["docs"] += 1
            pv["wrong"] += 1 if bad else 0

    drift_rows = []
    for vid, m in drift_meta.items():
        row = {"vendor": vid, "kind": m["kind"], "class": m["class"], "at_doc": m["at"],
               "vendor_docs_after": per_vendor_after[vid]["docs"], "wrong_after": per_vendor_after[vid]["wrong"]}
        if cfg.mode == "distill":
            evs = [e for e in eng.events if e["vendor_id"] == vid and e["seq"] >= m["at"]]
            det = next((e for e in evs if e["kind"] == "drift"), None)
            prom = next((e for e in evs if e["kind"] == "promoted"), None)
            vend_seqs = [eng.records[d.doc_id]["seq"] for d in docs if d.vendor_id == vid and eng.records[d.doc_id]["seq"] >= m["at"]]
            row["detected_after_vendor_docs"] = sum(1 for s in vend_seqs if det and s <= det["seq"]) if det else None
            row["recovered_after_vendor_docs"] = sum(1 for s in vend_seqs if prom and s <= prom["seq"]) if prom else None
            row["detection_reason"] = det["message"].split(": ", 1)[-1] if det else None
        drift_rows.append(row)

    last = [eng.records[d.doc_id] for d in docs[-1000:]]
    T = eng.totals
    calls = [c for d in docs for c in eng.records[d.doc_id]["llm_calls"]]
    tok_in, tok_out = sum(c["in"] for c in calls), sum(c["out"] for c in calls)
    esc_full = [c["out"] for c in calls if c["purpose"] == "escalate"]
    esc_scoped = [c["out"] for c in calls if c["purpose"] == "escalate (fields)"]
    return {
        "system": name, "mode": cfg.mode, "simulated_llm": not real, "docs": n,
        "doc_accuracy": round(1 - wrong_docs / n, 5), "wrong_docs": wrong_docs,
        "field_accuracy": round(1 - wrong_fields / max(total_fields, 1), 6),
        "cost_usd": round(T["cost_usd"], 6), "cost_per_1k_docs": round(1000 * T["cost_usd"] / n, 5),
        "cost_per_1k_docs_last_1000": round(sum(r["cost_usd"] for r in last), 5),
        "llm_calls": T["llm_calls"], "llm_calls_per_doc": round(T["llm_calls"] / n, 4),
        "code_share": round(T["code_docs"] / n, 4), "audit_share": round(T["audit_docs"] / n, 4),
        "code_share_last_1000": round(sum(1 for r in last if r["source"] == "code") / len(last), 4) if last else 0,
        "cost_per_1k_correct_docs": round(1000 * T["cost_usd"] / max(n - wrong_docs, 1), 5),
        "tokens_per_doc": round((tok_in + tok_out) / n, 1), "input_tokens": tok_in, "output_tokens": tok_out,
        "escalations_full": len(esc_full), "escalations_scoped": len(esc_scoped),
        "escalation_out_tokens_full": round(statistics.fmean(esc_full), 1) if esc_full else None,
        "escalation_out_tokens_scoped": round(statistics.fmean(esc_scoped), 1) if esc_scoped else None,
        "scoped_resolved": T.get("scoped_escalations", 0),
        "human_review": T["human_review"], "repaired": repaired, "repaired_wrong": repaired_wrong,
        "quarantined": T["quarantined"],
        "quarantine_corrected": T["quarantine_corrected"],
        "latency_ms_p50": round(statistics.median(lat), 2), "latency_ms_mean": round(statistics.fmean(lat), 2),
        "wall_seconds": round(wall, 2), "drifts": drift_rows, "curve": curve,
        "events": [e for e in eng.events if e["kind"] in ("drift", "quarantine", "promoted", "rejected", "candidate")][-200:],
        "parsers": eng.parser_table() if cfg.mode != "llm_only" else [],
    }


def to_markdown(results: list[dict]) -> str:
    sim = results[0]["simulated_llm"]
    out = [f"# Distill benchmark ({results[0]['docs']:,} documents, {'SIMULATED LLM tiers' if sim else 'real Claude calls'})", ""]
    out.append("| System | Doc accuracy | Wrong docs | LLM cost | Cost / 1k correct docs | Tokens / doc | Cost / 1k docs (last 1k) | LLM calls / doc | Code share (last 1k) | p50 latency | Human review |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        out.append(f"| {r['system']} | {r['doc_accuracy']:.2%} | {r['wrong_docs']} | ${r['cost_usd']:.4f} | "
                   f"${r['cost_per_1k_correct_docs']:.4f} | {r['tokens_per_doc']:,.0f} | "
                   f"${r['cost_per_1k_docs_last_1000']:.4f} | {r['llm_calls_per_doc']:.3f} | {r['code_share_last_1000']:.0%} | "
                   f"{r['latency_ms_p50']:.1f} ms | {r['human_review']} |")
    out.append("")
    out.append("## Drift events (wrong documents from the drifted vendor after the change)")
    out.append("")
    names = [r["system"] for r in results]
    out.append("| Vendor | Drift | Class | " + " | ".join(names) + " | Distill: detected after | Distill: recovered after |")
    out.append("|---|---|---|" + "---|" * len(names) + "---|---|")
    dist = next((r for r in results if r["mode"] == "distill"), None)
    for i, row in enumerate(results[0]["drifts"]):
        cells = []
        for r in results:
            rr = r["drifts"][i]
            cells.append(f"{rr['wrong_after']}/{rr['vendor_docs_after']}")
        dd = dist["drifts"][i] if dist else {}
        det = dd.get("detected_after_vendor_docs")
        rec = dd.get("recovered_after_vendor_docs")
        absorbed = det is None and sum(int(c.split("/")[0]) for c in cells[-1:]) == 0
        det_s = "not needed" if absorbed else (f"{det} docs" if det is not None else "-")
        rec_s = "not needed" if absorbed else (f"{rec} docs" if rec is not None else "-")
        out.append(f"| {row['vendor']} | {row['kind']} | {row['class']} | " + " | ".join(cells) + f" | {det_s} | {rec_s} |")
    out.append("")
    out.append("'Not needed': the change added lines the parser did not use; Distill checked those documents with the "
               "model, found the parser still right, and learned the new layout without replacing the parser.")
    dr = next((r for r in results if r["system"] == "distill"), None)
    if dr and dr.get("repaired"):
        out.append("")
        out.append(f"Repaired from the document's own invariants instead of a (further) model call: {dr['repaired']:,} of "
                   f"{dr['docs']:,} documents ({dr['repaired'] / dr['docs']:.1%}); {dr['repaired'] - dr['repaired_wrong']:,} "
                   f"of them exactly right ({1 - dr['repaired_wrong'] / dr['repaired']:.1%}).")
    if dr and dr.get("escalations_scoped"):
        out.append("")
        out.append(f"Scoped escalation: when the small model's answer fails only some checks, the large model is asked "
                   f"only for the fields (or the line-item rows) behind them, not the whole record. "
                   f"{dr['escalations_scoped']} such calls, {dr['scoped_resolved']} of them settled every check; answers "
                   f"averaged {dr['escalation_out_tokens_scoped']:.0f} output tokens against "
                   f"{dr['escalation_out_tokens_full']:.0f} for a full re-read ({dr['escalations_full']} full re-reads remained).")
    if sim:
        out.append("")
        out.append("_LLM tiers are simulated (ground truth + injected errors: small 5%, large 1.2% of documents; tokens billed "
                   "at published Haiku 5.5 / Sonnet 5.5 prices). Parsers, checks, drift detection, routing and audits are real._")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--real", action="store_true", help="call the real Claude API (ANTHROPIC_API_KEY)")
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    if args.real and args.n > 1000:
        print("note: --real with many documents costs money; consider --n 300 and adjust the drift schedule")
    schedule = auto_schedule(args.seed, args.n)
    results = []
    for name in args.systems.split(","):
        print(f"running {name} ...", flush=True)
        results.append(run_system(name, SYSTEMS[name], args.n, args.seed, schedule, real=args.real, progress=args.real))
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(results, indent=1))
    md = to_markdown(results)
    (out / "benchmark.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
