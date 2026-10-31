"""Evaluate Distill on FATURA, a public set of third-party invoices (Limam et al. 2023).

    python -m distill.realdata fatura            # 10,000 invoices from 50 layouts, text layer as provided
    python -m distill.realdata fatura --real     # use real Claude as the teacher (needs ANTHROPIC_API_KEY)

Without --real the model tier is a *perfect teacher*: the dataset's human labels, billed
with token counts from the real documents at published Claude prices. That isolates the
question Distill adds to an LLM pipeline: how much of real traffic can compiled parsers take
over, how often do they disagree with the teacher, and what does that save?

Scoring: a field is right when it matches any of the human-annotated values for that field
(letters and digits, ignoring case, spacing and punctuation), or when both say "absent".
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from .doctypes import FATURA, FaturaType, loose_amount
from .engine import Config, Engine
from .llm import AnthropicLLM, CachedLLM, SimulatedLLM, cost_of, LARGE_MODEL, SMALL_MODEL
from .model import Record, field_equal, _norm_raw

PERFECT = {"small": {"p_err": 0.0, "per_line": 0.0, "lat": 700, "per_tok": 4, "recover": 1.0},
           "large": {"p_err": 0.0, "per_line": 0.0, "lat": 1800, "per_tok": 12, "recover": 1.0}}


# ---------------------------------------------------------------- loaders
_FATURA_LABEL = re.compile(r"^\s*[A-Za-z][A-Za-z .()#/&%\d_-]{0,40}?\s*(?::|#|\s{2,})\s*")


_LABEL_WORDS = {"invoice", "number", "no", "id", "#", "date", "due", "bill", "total", "sub", "subtotal", "tax", "vat",
                "gst", "discount", "amount", "po", "order", "purchase", "payment", "issue", "issued", "balance",
                "grand", "ref", "reference", "num"}


def fatura_value(text: str) -> str:
    """A FATURA region's value: its text after the printed label(s) ('TAX:VAT (3.85%): 10.26 $' -> '10.26 $')."""
    t = " ".join(str(text).split())
    for _ in range(2):
        m = _FATURA_LABEL.match(t)
        if not (m and m.end() < len(t) and re.search(r"[A-Za-z]", m.group(0))):
            break
        t = t[m.end():].strip()
    # labels printed without a colon ('INVOICE ID 0923-334', 'Invoice number 7Y5M7d-024'): drop leading label words
    toks = t.split()
    while len(toks) > 1 and re.sub(r"[^a-z#]", "", toks[0].lower()) in _LABEL_WORDS:
        toks = toks[1:]
    return " ".join(toks)


def load_fatura(path: str = "data/fatura_invoices.jsonl.gz", per_template: Optional[int] = None) -> list[dict]:
    """FATURA (Limam et al. 2023, CC BY 4.0): 10,000 invoices from 50 templates, with the page's text layer.
    Prepared by scripts/prepare_fatura.py. The source is the template, as a vendor would be in AP."""
    names = FATURA.names
    items, seen = [], Counter()
    for line in gzip.open(path, "rt", encoding="utf-8"):
        d = json.loads(line)
        if per_template and seen[d["template"]] >= per_template:
            continue
        seen[d["template"]] += 1
        vals = {k: fatura_value(v) for k, v in d["fields"].items() if k in names and str(v).strip()}
        items.append({"doc_id": d["docid"], "text": d["text"], "truth": Record(FATURA, **vals),
                      "alts": {k: [v] for k, v in vals.items()},
                      "meta": {"cluster_id": d["template"], "document_type": "invoice", "words": None},
                      "source": f"T{d['template']}"})
    return items


# ---------------------------------------------------------------- scoring
_CURRENCY = {"$": "USD", "us$": "USD", "usd": "USD", "dollars": "USD", "€": "EUR", "eur": "EUR", "euro": "EUR",
             "£": "GBP", "gbp": "GBP", "chf": "CHF", "kč": "CZK", "czk": "CZK", "sar": "SAR"}
AMOUNT_FIELDS = {"amount_due", "total", "subtotal", "tax", "discount"}


def same_value(field: str, p, a) -> bool:
    """Do a predicted and an annotated value mean the same thing? Exact after dropping spacing and punctuation,
    or equal as amounts / currencies. Used for FATURA, whose labels are the printed text."""
    if p is None or a is None:
        return False
    if _norm_raw(str(p)) == _norm_raw(str(a)):
        return True
    if field in AMOUNT_FIELDS:
        x, y = loose_amount(p), loose_amount(a)
        return x is not None and y is not None and abs(x - y) < 0.006
    if field.startswith("currency"):
        c = lambda v: _CURRENCY.get(str(v).strip().lower(), str(v).strip().upper())
        return c(p) == c(a)
    return False


def answerable(item: dict, field: str) -> bool:
    """Is any annotated value for this field present in the text we send the model? Labels come from the
    original document; a damaged text layer can cut or garble them, and a field the text does not contain cannot be read."""
    key = "_norm_text"
    if key not in item:
        item[key] = _norm_raw(item["text"])
    return any(_norm_raw(a) and _norm_raw(a) in item[key] for a in item["alts"].get(field, []))


def score(pred: Optional[Record], item: dict, annotated_only: bool = False, answerable_only: bool = False) -> list[str]:
    """Wrong fields of a prediction, accepting any annotated alternative.

    annotated_only: ignore fields the annotators did not label on this document.
    answerable_only: also ignore labelled fields whose value does not appear in the OCR text.
    """
    schema = item["truth"].schema
    value_aware = schema.name == "fatura"
    bad = []
    for f in schema.fields:
        p = pred.values.get(f.name) if pred is not None else None
        alts = item["alts"].get(f.name)
        if not alts and (annotated_only or answerable_only):
            continue
        if answerable_only and not answerable(item, f.name):
            continue
        if not alts:
            if p not in (None, "") and not (f.type == "money" and p == 0):
                bad.append(f.name)
            continue
        if p in (None, ""):
            bad.append(f.name)
            continue
        ok = any(field_equal(f.name, p, a if f.type != "money" else loose_amount(a), f.type) for a in alts)
        if not ok and value_aware:
            ok = any(same_value(f.name, p, a) for a in alts)
        if not ok:
            bad.append(f.name)
    return bad


def rescore(pred_path: str, dataset_items: list[dict]) -> dict:
    """Score a saved predictions file again (no model calls)."""
    by_id = {it["doc_id"]: it for it in dataset_items}
    schema = dataset_items[0]["truth"].schema
    tot = Counter()
    for line in open(pred_path, encoding="utf-8"):
        r = json.loads(line)
        it = by_id[r["doc_id"]]
        pred = Record.from_dict(r["prediction"], schema) if r["prediction"] else None
        for name, kw in (("strict", {}), ("labelled", {"annotated_only": True}), ("answerable", {"answerable_only": True})):
            bad = score(pred, it, **kw)
            n = len(schema.fields) if name == "strict" else sum(
                1 for f in schema.fields if it["alts"].get(f.name) and (name == "labelled" or answerable(it, f.name)))
            tot[name + "_fields"] += n
            tot[name + "_wrong"] += len(bad)
            tot[name + "_docs_right"] += not bad
        tot["docs"] += 1
    return {k: (1 - tot[k + "_wrong"] / max(1, tot[k + "_fields"]), tot[k + "_docs_right"] / tot["docs"])
            for k in ("strict", "labelled", "answerable")}


# ---------------------------------------------------------------- run one system
def make_llm(dataset: str, real: bool, truth: dict, seed: int, out_dir: str = "results"):
    if not real:
        return SimulatedLLM(lambda d: truth[d], seed=seed, profiles=PERFECT)
    Path(out_dir).mkdir(exist_ok=True)
    return CachedLLM(AnthropicLLM(), str(Path(out_dir) / f"llm_cache_{dataset}.jsonl"))


def prefetch(llm, dataset: str, order: list[dict], workers: int = 8) -> None:
    """Real runs: read every document with the small model in parallel (cached), then re-read with the
    large model the ones whose small-model answer fails the checks. The systems then replay from cache."""
    if not isinstance(llm, CachedLLM):
        return
    doctype = FaturaType()
    schema = doctype.schema
    say = lambda tier: (lambda d, n: print(f"  {tier} model: {d}/{n} documents read", flush=True))
    print(f"reading {len(order)} documents with the small model ({workers} at a time; cached in results/) ...", flush=True)
    llm.prefetch([(it["doc_id"], it["text"]) for it in order], "small", schema, workers, say("small"))
    retry = []
    for it in order:
        r = llm.extract(it["doc_id"], it["text"], "small", schema)
        if r.invoice is None or not doctype.checks(r.invoice, "2019-01-01T00:00:00", None, None).passed:
            retry.append((it["doc_id"], it["text"]))
    if retry:
        print(f"re-reading {len(retry)} documents whose answers failed checks with the large model ...", flush=True)
        llm.prefetch(retry, "large", schema, workers, say("large"))


def run(dataset: str, items: list[dict], mode: str, real: bool = False, seed: int = 0, given_clusters: bool = True,
        llm_only_tier: str = "small", llm=None, predictions_path: Optional[str] = None) -> dict:
    order = items[:]
    random.Random(seed).shuffle(order)
    truth = {it["doc_id"]: it["truth"] for it in order}
    doctype = FaturaType(use_given_clusters=given_clusters)
    llm = llm or make_llm(dataset, real, truth, seed)
    eng = Engine(doctype, llm, Config(mode=mode, llm_only_tier=llm_only_tier))
    seen = Counter()
    rows = []
    t0 = time.perf_counter()
    pf = open(predictions_path, "w", encoding="utf-8") if predictions_path else None
    for k, it in enumerate(order, 1):
        r = eng.process(it["doc_id"], it["text"], "2019-01-01T00:00:00", it["meta"])
        pred = Record.from_dict(r["final"], doctype.schema) if r["final"] else None
        bad = score(pred, it)
        bad_annotated = score(pred, it, annotated_only=True)
        bad_answerable = score(pred, it, answerable_only=True)
        prior = seen[r["vendor_id"]]
        seen[r["vendor_id"]] += 1
        rows.append({"doc_id": it["doc_id"], "source": r["vendor_id"], "prior": prior, "path": (r["route"] or {}).get("path"),
                     "by_code": r["source"] == "code", "bad": bad, "bad_annotated": bad_annotated,
                     "n_annotated": sum(1 for f in doctype.schema.fields if it["alts"].get(f.name)),
                     "bad_answerable": bad_answerable,
                     "n_answerable": sum(1 for f in doctype.schema.fields if it["alts"].get(f.name) and answerable(it, f.name)),
                     "cost": r["cost_usd"], "flags": r["flags"]})
        if pf:
            pf.write(json.dumps({"doc_id": it["doc_id"], "system": mode, "source": r["vendor_id"], "by_code": r["source"] == "code",
                                 "prediction": r["final"], "annotations": it["alts"], "wrong": bad,
                                 "wrong_annotated_only": bad_annotated, "wrong_answerable_only": bad_answerable}, ensure_ascii=False) + "\n")
        if real and k % 50 == 0:
            print(f"  {mode}: {k}/{len(order)} documents, ${eng.totals['cost_usd']:.3f} so far", flush=True)
    if pf:
        pf.close()
    wall = time.perf_counter() - t0
    # what reading every document with one model would cost (same token accounting)
    cf = {"small": 0.0, "large": 0.0}
    for it in order:
        tin, tout = SimulatedLLM.tokens_for(it["text"], it["truth"])
        cf["small"] += cost_of(SMALL_MODEL, tin, tout)
        cf["large"] += cost_of(LARGE_MODEL, tin, tout)
    if isinstance(llm, CachedLLM):  # real runs: use the measured cost of real calls instead of the token model
        ver = llm._ver(doctype.schema)
        for tier in ("small", "large"):
            got = [llm.cache[(it["doc_id"], tier, ver)]["cost"] for it in order if (it["doc_id"], tier, ver) in llm.cache]
            if got:
                cf[tier] = sum(got) / len(got) * len(order)
    return summarize(dataset, mode, real, rows, eng, cf, wall, doctype.schema)


def summarize(dataset, mode, real, rows, eng, cf, wall, schema) -> dict:
    n = len(rows)
    nf = len(schema.fields)
    code = [r for r in rows if r["by_code"]]
    buckets = {"first 3 from a source": lambda p: p < 3, "4th-10th": lambda p: 3 <= p < 10, "11th onward": lambda p: p >= 10}
    by_bucket = {}
    for name, fn in buckets.items():
        rs = [r for r in rows if fn(r["prior"])]
        if rs:
            by_bucket[name] = {"docs": len(rs), "code_share": sum(r["by_code"] for r in rs) / len(rs),
                               "exact": sum(not r["bad"] for r in rs) / len(rs)}
    field_err = Counter(b for r in rows for b in r["bad"])
    T = eng.totals
    return {
        "dataset": dataset, "system": mode, "teacher": "real Claude" if real else "human labels (perfect teacher)",
        "docs": n, "sources": len({r["source"] for r in rows}),
        "doc_exact": sum(not r["bad"] for r in rows) / n,
        "field_accuracy": 1 - sum(len(r["bad"]) for r in rows) / (n * nf),
        "doc_exact_annotated": sum(not r["bad_annotated"] for r in rows) / n,
        "field_accuracy_annotated": 1 - sum(len(r["bad_annotated"]) for r in rows) / max(1, sum(r["n_annotated"] for r in rows)),
        "doc_exact_answerable": sum(not r["bad_answerable"] for r in rows) / n,
        "field_accuracy_answerable": 1 - sum(len(r["bad_answerable"]) for r in rows) / max(1, sum(r["n_answerable"] for r in rows)),
        "answerable_share": sum(r["n_answerable"] for r in rows) / max(1, sum(r["n_annotated"] for r in rows)),
        "code_share": len(code) / n,
        "code_doc_exact": (sum(not r["bad"] for r in code) / len(code)) if code else None,
        "code_field_accuracy": (1 - sum(len(r["bad"]) for r in code) / (len(code) * nf)) if code else None,
        "silent_wrong_docs": sum(1 for r in code if r["bad"]),
        "llm_calls": T["llm_calls"], "llm_calls_per_doc": T["llm_calls"] / n,
        "cost_usd": T["cost_usd"], "cost_all_small_usd": cf["small"], "cost_all_large_usd": cf["large"],
        "noise_disagreements": T["noise_disagreements"], "drifts": sum(1 for e in eng.events if e["kind"] == "drift"),
        "quarantined": T["quarantined"], "quarantine_corrected": T["quarantine_corrected"],
        "parsers_promoted": sum(1 for e in eng.events if e["kind"] == "promoted"),
        "by_position": by_bucket, "field_errors": dict(field_err.most_common()), "wall_seconds": wall,
    }


def to_markdown(results: list[dict]) -> str:
    r0 = results[0]
    kind = "third-party template invoices (FATURA, Limam et al. 2023), text layer as provided"
    out = [f"# {r0['dataset'].upper()}: {r0['docs']:,} {kind}, {r0['sources']} sources",
           f"Teacher: {r0['teacher']}.", ""]
    out.append("| System | Docs fully right (answerable fields) | Fields right | Labelled fields right | Answerable labelled fields right | Handled by code | Right when handled by code | Wrong but accepted by code | LLM calls / doc | LLM cost | Cost if every doc went to Haiku / Sonnet |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        ce = "-" if r["code_doc_exact"] is None else f"{r['code_doc_exact']:.1%}"
        out.append(f"| {r['system']} | {r.get('doc_exact_answerable', r['doc_exact']):.1%} | {r['field_accuracy']:.1%} | "
                   f"{r.get('field_accuracy_annotated', r['field_accuracy']):.1%} | {r.get('field_accuracy_answerable', r['field_accuracy']):.1%} | {r['code_share']:.1%} | {ce} | "
                   f"{r['silent_wrong_docs']} | {r['llm_calls_per_doc']:.2f} | ${r['cost_usd']:.4f} | "
                   f"${r['cost_all_small_usd']:.4f} / ${r['cost_all_large_usd']:.2f} |")
    d = next((r for r in results if r["system"] == "distill"), None)
    if d:
        out += ["", "## Distill by how many documents it had already seen from that source", "",
                "| Position | Docs | Handled by code | Fully right |", "|---|---|---|---|"]
        for k, v in d["by_position"].items():
            out.append(f"| {k} | {v['docs']:,} | {v['code_share']:.1%} | {v['exact']:.1%} |")
        out += ["", f"Parsers promoted: {d['parsers_promoted']}. Disagreements recognised as OCR noise: "
                    f"{d['noise_disagreements']}. Drifts confirmed: {d['drifts']}.",
                "", "_FATURA's text layer has its own OCR-like damage (e.g. 'INVO1036008' for 'INV01036008'); the model "
                    "is given the same text. "
                    "_Fields right_ counts a value the annotators did not label as wrong; _labelled fields right_ scores only "
                    "the fields labelled on that document; _answerable_ further "
                    "drops labelled values that do not appear in the text the model is given "
                    f"({1 - d.get('answerable_share', 1):.0%} of labelled fields here). "
                    "Amounts and currencies are compared by value ($ = USD). "
                    "Per-document predictions are in the matching `_predictions.jsonl` file._",
                "", "Fields Distill got wrong most often: " +
                ", ".join(f"{k} ({v})" for k, v in list(d["field_errors"].items())[:6])]
    if not r0["teacher"].startswith("real"):
        out += ["", "_With a perfect teacher, reading every document with the model is 100% right by construction; "
                    "the question is how much of that Distill keeps while replacing the model with compiled code. "
                    "Run with `--real` to measure accuracy against real Claude._"]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", default="fatura", choices=["fatura"])
    ap.add_argument("--fatura", default="data/fatura_invoices.jsonl.gz")
    ap.add_argument("--per-template", type=int, help="FATURA: invoices per template (default all 200)")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--clusters", help="only documents from the N largest layout clusters (e.g. 10), or a rank range "
                                       "(e.g. 11-20, for clusters the prompt was not tuned on)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cluster", choices=["given", "learned"], default="given",
                    help="identify the source by the dataset's template id (like a vendor master) or by unsupervised layout clustering")
    ap.add_argument("--systems", default="llm_only,static,distill")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=8, help="parallel model calls in --real runs")
    a = ap.parse_args(argv)
    items = load_fatura(a.fatura, a.per_template)
    if a.clusters:
        from collections import Counter as _C
        lo, _, hi = a.clusters.partition("-")
        lo, hi = (1, int(lo)) if not hi else (int(lo), int(hi))
        top = {k for k, _ in _C(it["source"] for it in items).most_common(hi)[lo - 1:]}
        items = [it for it in items if it["source"] in top]
    if a.limit:
        random.Random(a.seed).shuffle(items)
        items = items[:a.limit]
    Path(a.out).mkdir(exist_ok=True)
    tag = f"{a.dataset}{'_learned' if a.cluster == 'learned' else ''}{'_real' if a.real else ''}"
    llm = None
    if a.real:
        llm = make_llm(a.dataset, True, {}, a.seed, a.out)
        order = items[:]
        random.Random(a.seed).shuffle(order)
        prefetch(llm, a.dataset, order, workers=a.workers)
    results = []
    for sysname in a.systems.split(","):
        print(f"running {sysname} on {len(items)} {a.dataset} documents ...", flush=True)
        results.append(run(a.dataset, items, sysname, real=a.real, seed=a.seed, given_clusters=a.cluster == "given",
                           llm=llm, predictions_path=str(Path(a.out) / f"{tag}_{sysname}_predictions.jsonl")))
        # save after every system so an interrupted run keeps what finished
        (Path(a.out) / f"{tag}.json").write_text(json.dumps(results, indent=1))
        (Path(a.out) / f"{tag}.md").write_text(to_markdown(results))
    print(to_markdown(results))

if __name__ == "__main__":
    main()
