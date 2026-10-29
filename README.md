# Distill

**The first invoice from a supplier costs a fraction of a cent of AI. The thousandth costs nothing, and Distill keeps proving it is still right.**

Companies send the same kinds of documents through large language models millions of times, even though each supplier's layout barely changes. Distill reads a supplier's first few invoices with Claude, compiles what it learned into a small parser (data, not code), and from then on extracts that supplier's invoices without AI. The hard part, and the part this project is about, is what happens next: layouts change, sometimes loudly and sometimes silently, and a compiled parser that is wrong but confident is worse than no parser at all.

Distill treats every compiled parser like a production release:

1. **Detect drift.** Every result goes through business checks (line math, totals, tax, payment terms, PO and vendor master data, duplicates) and the vendor's own history. Layout fingerprints notice unseen or missing lines. A per-vendor monitor turns these signals into *stable → suspect → drift*.
2. **Generate candidate parsers** from fresh, check-verified examples. Ambiguity (is `03/04/2026` DD/MM or MM/DD?) produces several candidates instead of a guess.
3. **Test them in isolation** on stored examples, then in **shadow mode** next to live traffic. Disagreements are arbitrated by a larger model, so a wrong LLM label can't kill a correct parser, and a correct parser can correct a wrong LLM label.
4. **Deploy only a validated version**, with lineage, evidence and one-click rollback. Documents the old parser accepted since its last verified result are **quarantined** and re-checked.
5. **Keep proving reliability.** Audits compare the parser with the model at a rate that adapts to the evidence (100% for a new parser, halving every 3 clean audits, 2% floor) and spike on risk: high value, history warnings, a suspicious monitor. Every parser carries an exact 95% upper bound on its error rate (Clopper–Pearson; ≈3/n with zero errors, the rule of three).
6. **Route by policy, not by a fixed cascade.** Each document's path (code, code + audit, small model, large model, person) is chosen from risk, uncertainty, cost and latency, and the reasons are recorded with the result.

## Results

8,000 synthetic invoices from 20 vendors, six template changes injected mid-stream (`python -m distill.bench`). The LLM tiers are **simulated** in this run (ground truth plus injected errors: 5% of documents for the small model, 1.2% for the large one; tokens billed at published Claude Haiku 5.5 and Sonnet 5.5 prices). Everything else is real: parser synthesis, checks, drift detection, routing, audits, quarantine.

| System | Exactly right | AI cost | Cost per 1k invoices, steady state | LLM calls per invoice | p50 latency |
|---|---|---|---|---|---|
| Claude Sonnet on every invoice | 98.81% | $19.09 | $2.39 | 1.00 | 3.8 s |
| Claude Haiku on every invoice (Sonnet on failed checks) | 98.15% | $1.54 | $0.19 | 1.03 | 1.4 s |
| Compile once (Evaporate-style, no drift handling) | 87.89% | $0.48 | $0.085 | 0.32 | <1 ms |
| **Distill** | **99.95%** | **$0.26** | **$0.024** | **0.14** | **<1 ms** |

| Template change | Kind | Wrong after change: compile-once | Wrong after change: Distill | Distill detected after | New parser promoted after |
|---|---|---|---|---|---|
| Price printed before quantity | silent | 788 of 788 | 0 | 1 invoice | 6 invoices |
| Dates switch DD/MM → MM/DD | silent | 112 of 543 | 0 | 1 | 6 |
| Labels renamed, due date moved | loud | 17 of 1,222 | 0 | 1 | 7 |
| Item table re-flowed | loud | 5 of 412 | 0 | 1 | 6 |
| Number format 1,234.56 → 1.234,56 | loud | 7 of 300 | 0 | 1 | 8 |
| New fields and discounts | structural | 0 of 250 | 0 | not needed | not needed |

Results hold across seeds (3, 5, 11, 23): 99.95–99.99% exact, 5–7x cheaper than Haiku-on-everything overall and 7–14x in steady state, every drift detected within one invoice of the changed vendor.

**Before you quote these numbers, run the real thing:** `python -m distill.bench --real --n 400` (see below). The simulated error rates are assumptions; real Claude may be more or less accurate on these documents.

## Why the silent drifts matter

"Price before quantity" is the case arithmetic can't catch: `12.50 @ 3 = 37.50` and `3 @ 12.50 = 37.50` both pass every invariant. Distill catches it because the vendor's price history says this vendor never sells fasteners at 3.00 or in fractional quantities: a history warning raises the risk, the policy orders a targeted audit, the audit disagrees, the arbiter confirms, and the parser is suspended, its recent output quarantined and re-checked, and a replacement synthesized, validated and shadow-tested. The compile-once baseline gets every one of those invoices wrong.

## How it relates to prior work

Stanford's Evaporate (Arora et al., VLDB 2024) showed that having an LLM synthesize extraction functions instead of reading every document cuts the tokens processed by about 110x, and Evaporate-Code+ already generates many candidate functions and aggregates them with weak supervision. Distill does not claim the idea of code synthesis. Its contribution is the operating discipline around it for a **stream** of business documents that **changes over time**: drift detection (including drift that passes arithmetic), parser versioning with isolated validation, shadow testing and promotion, quarantine of affected results, risk-based routing, and continuously updated statistical evidence that each parser is still right.

## Quick start

Python 3.10+. No API key needed for the simulated mode.

```bash
pip install -r requirements.txt
python -m pytest -q                 # 21 tests
python -m distill.bench             # benchmark -> results/benchmark.md and .json (about 15 s)
python -m distill.server            # live dashboard on http://127.0.0.1:8000
```

With real Claude (costs money; Haiku 5.5 is $0.10 / $0.50 per million input/output tokens, Sonnet 5.5 $2 / $10):

```bash
export ANTHROPIC_API_KEY=sk-ant-...           # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
python scripts/smoke_real.py 3                # 3 invoices x 2 models, prints accuracy and cost
python -m distill.bench --real --n 400 --systems distill,llm_only_small
python -m distill.server --real
```

Models are set with `DISTILL_SMALL_MODEL` (default `claude-haiku-5-5`) and `DISTILL_LARGE_MODEL` (default `claude-sonnet-5-5`).

## The live demo

Open the dashboard, press **Start stream**, and let it run to about 1,000 invoices (speed 30–60/s). Then hand a judge the **Change a vendor's template** panel: they pick any vendor and any change, including the silent ones, and Distill isn't told which. Within seconds the feed shows the drift, the quarantine, the candidate parser, the shadow agreements and the promotion. Click any bar in the invoice strip to see exactly why that invoice cost what it cost: route reasons, checks, AI calls, parser version and the raw text. Click a vendor to see its parser lineage and roll back. The full script is in [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md).

## Code map

| File | What it does |
|---|---|
| `distill/datagen.py` | Synthetic invoice stream: 20 vendors, 5 countries, varied labels, date/number locales, table styles, noise, exact ground truth, six drift mutations |
| `distill/parser.py` | Parser specs as JSON (line patterns, styles, required fields) and the executor; no generated code ever runs |
| `distill/synth.py` | Candidate generation by induction from labelled examples (strict and relaxed generators, optional LLM-proposed specs) and isolated validation |
| `distill/checks.py` | Invariant, format, master-data and history checks |
| `distill/fingerprint.py` | Vendor identification and layout fingerprints |
| `distill/drift.py` | Per-vendor drift monitor |
| `distill/policy.py` | Risk, uncertainty, cost and latency routing with readable reasons; economy / balanced / assurance profiles |
| `distill/audit.py` | Adaptive audit rate, deterministic sampling, exact error bounds |
| `distill/engine.py` | The pipeline: routing, audits, arbitration, quarantine, synthesis, shadow, promotion, rollback |
| `distill/registry.py` | SQLite ledger of parser versions, documents and events |
| `distill/llm.py` | Claude Messages API client (standard library only) and the labelled simulation |
| `distill/bench.py` | Four-system benchmark |
| `distill/server.py`, `web/index.html` | Live dashboard |

## Honest limitations

- Documents are synthetic text, as if after PDF text extraction. Scanned PDFs need an OCR step first (Docling or PaddleOCR would slot in before `Engine.process`).
- One layout per vendor at a time. Vendors with several concurrent templates (invoices and credit notes) would need fingerprint clustering, which the fingerprint module is shaped for but doesn't do yet.
- The simulated LLM's error rates are assumptions; the real-mode benchmark is the number to quote.
- The induced parsers are line-pattern based. Layouts that wrap fields across lines unpredictably need the LLM-proposed generator (`Config(use_llm_spec_proposals=True)`), which is implemented but untested against the live API.
