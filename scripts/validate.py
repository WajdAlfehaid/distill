"""Validation beyond the main benchmark: held-out streams and ablations.

    python scripts/validate.py            # writes results/validation.md (about 15 minutes, no API calls)

1. Held-out streams. The main benchmark uses seed 11, the stream Distill was developed on. Seeds 101, 202 and 303
   generate new suppliers, layouts, scan damage and template-change schedules that were never seen during
   development. The same systems run on each.
2. Ablations. Distill with one component switched off (invariant repair, scoped escalation), on the main stream,
   to show that each one contributes.
"""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HELD_OUT = [101, 202, 303]
HELD_OUT_SYSTEMS = "llm_only_large,llm_only_small,static,distill"
ABLATION_SYSTEMS = "distill,distill_no_repair,distill_no_scoped"
NAMES = {"llm_only_large": "Sonnet on every invoice", "llm_only_small": "Haiku (Sonnet when checks fail)",
         "static": "Compile once", "distill": "Distill", "distill_no_repair": "Distill without invariant repair",
         "distill_no_scoped": "Distill without scoped escalation"}


def run(seed: int, systems: str, out: str) -> list:
    subprocess.run([sys.executable, "-m", "distill.bench", "--seed", str(seed), "--systems", systems, "--out", out],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return json.loads((ROOT / out / "benchmark.json").read_text())


def main() -> None:
    jobs = {f"seed {s}": (s, HELD_OUT_SYSTEMS, f"results/heldout_seed{s}") for s in HELD_OUT}
    jobs["ablation"] = (11, ABLATION_SYSTEMS, "results/ablation")
    with ThreadPoolExecutor(max_workers=2) as ex:
        done = {k: ex.submit(run, *v) for k, v in jobs.items()}
        res = {k: f.result() for k, f in done.items()}

    out = ["# Validation: held-out streams and ablations", "",
           "`python scripts/validate.py`. Model tiers are simulated as in the main benchmark (tokens billed at Claude "
           "Haiku 5.5 / Sonnet 5.5 prices); parsers, checks, drift detection, routing and audits are real.", "",
           "## 1. Held-out streams (never seen during development)", "",
           "Seeds 101, 202 and 303 each generate 8,000 invoices from new suppliers, layouts, scan damage and template "
           "changes. Development used seed 11 only.", "",
           "| Stream | System | Exactly right | AI cost | Cost per 1k correct | Wrong after template changes | Changes caught (median invoices) |",
           "|---|---|---|---|---|---|---|"]
    summary = {}
    for s in HELD_OUT:
        for r in res[f"seed {s}"]:
            after = sum(d["wrong_after"] for d in r["drifts"])
            total_after = sum(d["vendor_docs_after"] for d in r["drifts"])
            det = sorted(d["detected_after_vendor_docs"] for d in r["drifts"] if d.get("detected_after_vendor_docs") is not None)
            det_s = f"{det[len(det) // 2]}" if det else "-"
            out.append(f"| seed {s} | {NAMES[r['system']]} | {r['doc_accuracy']:.1%} | ${r['cost_usd']:.2f} | "
                       f"${r['cost_per_1k_correct_docs']:.3f} | {after} / {total_after} | {det_s if r['mode'] == 'distill' else ''} |")
            summary.setdefault(r["system"], []).append(r)
    d, h, l = summary["distill"], summary["llm_only_small"], summary["llm_only_large"]
    ratio_h = [y["cost_per_1k_correct_docs"] / x["cost_per_1k_correct_docs"] for x, y in zip(d, h)]
    ratio_l = [y["cost_per_1k_correct_docs"] / x["cost_per_1k_correct_docs"] for x, y in zip(d, l)]
    acc = [(x["doc_accuracy"], y["doc_accuracy"]) for x, y in zip(d, h)]
    out += ["", f"Across the three held-out streams Distill costs {min(ratio_h):.1f}–{max(ratio_h):.1f}x less than Haiku "
                f"and {min(ratio_l):.0f}–{max(ratio_l):.0f}x less than Sonnet per correct invoice, and is "
                f"{'more' if all(a > b for a, b in acc) else 'not always more'} accurate than Haiku on every stream "
                f"({', '.join(f'{a:.1%} vs {b:.1%}' for a, b in acc)})."]

    out += ["", "## 2. Ablations (main stream, seed 11)", "",
            "| System | Exactly right | Wrong invoices | AI cost | Cost per 1k correct | Model calls / invoice |",
            "|---|---|---|---|---|---|"]
    for r in res["ablation"]:
        out.append(f"| {NAMES[r['system']]} | {r['doc_accuracy']:.2%} | {r['wrong_docs']} | ${r['cost_usd']:.2f} | "
                   f"${r['cost_per_1k_correct_docs']:.3f} | {r['llm_calls_per_doc']:.2f} |")
    full = next(r for r in res["ablation"] if r["system"] == "distill")
    for r in res["ablation"]:
        if r["system"] != "distill":
            out.append(f"\n- Switching off {NAMES[r['system']].split('without ')[1]}: cost {r['cost_usd'] / full['cost_usd']:.2f}x, "
                       f"{r['wrong_docs'] - full['wrong_docs']:+d} wrong invoices.")
    out += ["", "The main benchmark (`results/benchmark.md`) also includes the reverse ablation: Haiku with Distill's "
                "invariant repair but no compiling, which separates what repair adds from what compiling adds."]
    (ROOT / "results" / "validation.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
