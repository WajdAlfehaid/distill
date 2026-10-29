"""Evaluate Distill on public, real-world document datasets.

    python -m distill.realdata docile            # DocILE (Rossum): real business documents, DocTR OCR
    python -m distill.realdata sroie             # SROIE (ICDAR 2019): scanned receipts, Tesseract OCR
    python -m distill.realdata docile --real     # use real Claude as the teacher (needs ANTHROPIC_API_KEY)

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
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from .doctypes import DOCILE, RECEIPT, DocileType, ReceiptType, loose_amount
from .engine import Config, Engine
from .llm import AnthropicLLM, SimulatedLLM, cost_of, LARGE_MODEL, SMALL_MODEL
from .model import Record, field_equal

PERFECT = {"small": {"p_err": 0.0, "per_line": 0.0, "lat": 700, "per_tok": 4, "recover": 1.0},
           "large": {"p_err": 0.0, "per_line": 0.0, "lat": 1800, "per_tok": 12, "recover": 1.0}}


# ---------------------------------------------------------------- loaders
def load_docile(path: str = "data/docile_trainval_words.jsonl.gz") -> list[dict]:
    names = DOCILE.names
    items = []
    for line in gzip.open(path, "rt", encoding="utf-8"):
        d = json.loads(line)
        if not d["text"].strip():
            continue
        alts = {k: [v.replace("\n", " ") for v in vs] for k, vs in d["fields"].items() if k in names}
        truth = Record(DOCILE, **{k: vs[0] for k, vs in alts.items()})
        items.append({"doc_id": d["docid"], "text": d["text"], "truth": truth, "alts": alts,
                      "meta": {"cluster_id": d["cluster_id"], "document_type": d["document_type"],
                               "words": d.get("words")},
                      "source": f"K{d['cluster_id']}"})
    return items


def load_sroie(ocr_dir: str, key_dir: str) -> list[dict]:
    items = []
    for f in sorted(Path(ocr_dir).glob("*.txt")):
        text = f.read_text(errors="replace")
        kp = Path(key_dir) / (f.stem + ".json")
        if not text.strip() or not kp.exists():
            continue
        k = json.loads(kp.read_text(errors="replace"))
        vals = {"company": (k.get("company") or "").strip() or None, "date": (k.get("date") or "").strip() or None,
                "address": (k.get("address") or "").strip() or None, "total": loose_amount(k.get("total"))}
        truth = Record(RECEIPT, **vals)
        items.append({"doc_id": f.stem, "text": text, "truth": truth,
                      "alts": {n: [v] for n, v in vals.items() if v is not None}, "meta": {},
                      "source": vals["company"] or "?"})
    return items


# ---------------------------------------------------------------- scoring
def score(pred: Optional[Record], item: dict) -> list[str]:
    """Wrong fields of a prediction, accepting any annotated alternative."""
    schema = item["truth"].schema
    bad = []
    for f in schema.fields:
        p = pred.values.get(f.name) if pred is not None else None
        alts = item["alts"].get(f.name)
        if not alts:
            if p not in (None, "") and not (f.type == "money" and p == 0):
                bad.append(f.name)
            continue
        if p in (None, ""):
            bad.append(f.name)
            continue
        if not any(field_equal(f.name, p, a if f.type != "money" else loose_amount(a), f.type) for a in alts):
            bad.append(f.name)
    return bad


# ---------------------------------------------------------------- run one system
def run(dataset: str, items: list[dict], mode: str, real: bool = False, seed: int = 0, given_clusters: bool = True,
        llm_only_tier: str = "small") -> dict:
    order = items[:]
    random.Random(seed).shuffle(order)
    truth = {it["doc_id"]: it["truth"] for it in order}
    doctype = DocileType(use_given_clusters=given_clusters) if dataset == "docile" else ReceiptType(threshold=0.4)
    llm = AnthropicLLM() if real else SimulatedLLM(lambda d: truth[d], seed=seed, profiles=PERFECT)
    eng = Engine(doctype, llm, Config(mode=mode, llm_only_tier=llm_only_tier))
    seen = Counter()
    rows = []
    t0 = time.perf_counter()
    for it in order:
        r = eng.process(it["doc_id"], it["text"], "2019-01-01T00:00:00", it["meta"])
        pred = Record.from_dict(r["final"], doctype.schema) if r["final"] else None
        bad = score(pred, it)
        prior = seen[r["vendor_id"]]
        seen[r["vendor_id"]] += 1
        rows.append({"doc_id": it["doc_id"], "source": r["vendor_id"], "prior": prior, "path": (r["route"] or {}).get("path"),
                     "by_code": r["source"] == "code", "bad": bad, "cost": r["cost_usd"], "flags": r["flags"]})
    wall = time.perf_counter() - t0
    # what reading every document with one model would cost (same token accounting)
    cf = {"small": 0.0, "large": 0.0}
    for it in order:
        tin, tout = SimulatedLLM.tokens_for(it["text"], it["truth"])
        cf["small"] += cost_of(SMALL_MODEL, tin, tout)
        cf["large"] += cost_of(LARGE_MODEL, tin, tout)
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
    out = [f"# {r0['dataset'].upper()}: {r0['docs']:,} real documents, {r0['sources']} sources",
           f"Teacher: {r0['teacher']}.", ""]
    out.append("| System | Docs fully right | Fields right | Handled by code | Right when handled by code | Wrong but accepted by code | LLM calls / doc | LLM cost | Cost if every doc went to Haiku / Sonnet |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        ce = "-" if r["code_doc_exact"] is None else f"{r['code_doc_exact']:.1%}"
        out.append(f"| {r['system']} | {r['doc_exact']:.1%} | {r['field_accuracy']:.1%} | {r['code_share']:.1%} | {ce} | "
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
                "", "Fields Distill got wrong most often: " +
                ", ".join(f"{k} ({v})" for k, v in list(d["field_errors"].items())[:6])]
    if not r0["teacher"].startswith("real"):
        out += ["", "_With a perfect teacher, reading every document with the model is 100% right by construction; "
                    "the question is how much of that Distill keeps while replacing the model with compiled code. "
                    "Run with `--real` to measure accuracy against real Claude._"]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["docile", "sroie"])
    ap.add_argument("--docile", default="data/docile_trainval_words.jsonl.gz")
    ap.add_argument("--sroie-ocr", default="data/sroie_ocr")
    ap.add_argument("--sroie-keys", default="data/sroie_keys")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cluster", choices=["given", "learned"], default="given",
                    help="DocILE: identify the source by DocILE's layout cluster (like a vendor master) or by unsupervised clustering")
    ap.add_argument("--systems", default="llm_only,static,distill")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    items = load_docile(a.docile) if a.dataset == "docile" else load_sroie(a.sroie_ocr, a.sroie_keys)
    if a.limit:
        random.Random(a.seed).shuffle(items)
        items = items[:a.limit]
    results = []
    for sysname in a.systems.split(","):
        print(f"running {sysname} on {len(items)} {a.dataset} documents ...", flush=True)
        results.append(run(a.dataset, items, sysname, real=a.real, seed=a.seed, given_clusters=a.cluster == "given"))
    Path(a.out).mkdir(exist_ok=True)
    tag = f"{a.dataset}{'_learned' if a.cluster == 'learned' and a.dataset == 'docile' else ''}{'_real' if a.real else ''}"
    (Path(a.out) / f"{tag}.json").write_text(json.dumps(results, indent=1))
    md = to_markdown(results)
    (Path(a.out) / f"{tag}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
