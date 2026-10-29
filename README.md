# Distill

**Document extraction that compiles itself into code where that is safe, and knows when it isn't.**

Companies send the same kinds of documents through language models over and over, even though most suppliers' layouts barely change. Distill reads a source's first documents with a model, compiles what it learned into a small parser (JSON data, never generated code), and from then on extracts that source's documents without AI, while checking every result, auditing its parsers against the model, and telling scan damage apart from a real change of template.

## What it does

1. **Compile.** From a few verified examples, Distill induces candidate parsers: text-line patterns for text documents, and *registered geometric templates* for scanned pages (each page is aligned to a reference copy of its layout using shared landmark words, then fields are read from learned zones). Candidates are validated in isolation, only against values that are actually legible in each example, then run in **shadow mode** next to live traffic before promotion.
2. **Check everything.** Arithmetic (line math, subtotal, discount, tax, total), formats, master data (vendor tax ID, currency, tax rate, payment terms, PO), duplicates, and the vendor's own price and quantity history.
3. **Tell noise from drift.** Scans damage documents: look-alike characters (0/O, 1/l, 5/S), smudged labels, dropped digits. Distill recognises these: values one look-alike or one slip apart, labels within an edit budget, a parser output that already failed the checks on an unchanged layout. Only a **clean** disagreement confirmed by an arbiter model, or the **same field** disagreeing copy after copy, suspends a parser.
4. **Recover.** On real drift: suspend the parser, quarantine and re-check what it accepted since its last verified result, synthesize a replacement from fresh examples, shadow-test it, promote it. A parser that has simply never seen a field (a vendor's first discount) is kept and extended, not suspended.
5. **Keep proving reliability.** Audits compare parsers with the model at a rate that adapts to evidence (100% for a new parser, halving every 3 clean audits, 2% floor) and spike on risk. Each parser carries an exact 95% upper bound on its error rate.
6. **Route by policy.** Each document's path (code, code + audit, small model, large model, person) is chosen from risk, uncertainty, cost and latency, and the reasons are recorded with the result.

## Results

Three evaluations, each answering a different question. All numbers come from scripts in this repository.

### 1. Synthetic invoice stream with realistic scan damage, and injected template changes

`python -m distill.bench`: 8,000 invoices from 20 vendors (5 countries, 5 table styles, wrapped descriptions, tables continued on a second page, free-text notes). Each vendor is a born-digital PDF, a good scan or a poor scan, with character-level OCR damage. Six template changes are injected mid-stream. **The model tiers are simulated**: ground truth plus injected mistakes, which grow with document length and with damage to printed values; tokens are billed at published Claude Haiku 5.5 / Sonnet 5.5 prices.

| System | Exactly right | AI cost | Model calls per invoice | Cost per 1k, steady state |
|---|---|---|---|---|
| Claude Sonnet on every invoice | 94.1% | $24.69 | 1.00 | $3.08 |
| Claude Haiku on every invoice (Sonnet when checks fail) | 94.8% | $4.18 | 1.11 | $0.53 |
| Compile once, no drift handling | 80.8% | $2.74 | 0.56 | $0.38 |
| **Distill** | **96.2%** | **$3.75** | **0.64** | **$0.44** |

Code handles 74% of steady-state traffic. Template changes were detected within 1–3 documents of the changed vendor and a validated replacement was live within 8–21. The compile-once baseline got 393 of 430 invoices wrong after a silent change (price and quantity swapped, every total still adds up); Distill got 16 wrong, in line with the model-only systems on the same vendor (17 and 32), whose errors come from scan damage.

**What this says, honestly:** against the cheapest model, Distill's saving is modest (about 10% in dollars, 42% fewer model calls), because the documents it cannot compile are the damaged ones, and those are the expensive ones for every system. Its clear wins are accuracy (checks, arbitration and the parser correcting model slips), latency (compiled documents take under a millisecond instead of 1–4 seconds), a large saving against larger models (6.6x versus Sonnet), and the evidence trail on every result.

### 2. DocILE: 5,680 real business documents (Rossum, ICDAR 2023)

`python -m distill.realdata docile`: invoices, orders, credit notes and utility bills with DocTR OCR, 1,063 layout clusters. The model tier is a **perfect teacher** (the human labels) so the run isolates the parser layer: how much real traffic compiled templates can take over, and how often they disagree with the teacher.

| System | Handled by code | Right when handled by code | Wrong but accepted by code | Parsers promoted |
|---|---|---|---|---|
| Compile once, no drift handling | 4.4% | 32.1% | 171 | 37 |
| **Distill** | 0.3% | 100% | **0** | 9 |

This is the most important result in the repository, and it is sobering. DocILE's "layouts" are much looser than real templates: copies of one layout label different fields and place values differently, and the same value is often printed in several places. A compiler that trusts what it learned (compile-once) takes 4.4% of traffic and gets two thirds of it wrong. Distill's validation, shadow testing and noise-versus-drift logic refuse almost everything, so it makes no silent errors but saves nothing on this data.

What this means for the product: compilation pays off on high-volume, consistently templated sources (the situation in result 1, and typical for a company's top suppliers), and the hard engineering problem is knowing *when* that is the case. Raising coverage on DocILE-like data (per-field partial templates with targeted model calls for the remaining fields, more examples per layout) is the main open work.

### 3. SROIE: 619 real scanned receipts (ICDAR 2019), Tesseract OCR

`python -m distill.realdata sroie`: faded thermal receipts from 236 shops, OCR'd here with Tesseract, identified by unsupervised layout clustering. Distill compiled **no** parsers: nothing it learned held up across copies of these scans, so every receipt went to the model and none was accepted wrongly. The compile-once baseline compiled anyway and was wrong on 9 of the 17 receipts it handled. This is the behaviour we want on data that should not be compiled.

## How it relates to prior work

Stanford's Evaporate (Arora et al., VLDB 2024) showed that synthesizing extraction functions instead of reading every document cuts the tokens an LLM processes by about 110x, and Evaporate-Code+ already generates and aggregates candidate functions. Template ("zonal") extraction for known layouts is decades old. Distill's contribution is the operating discipline for a **stream** of real, damaged business documents that **change over time**: learning templates automatically from a model, telling scan noise from drift, versioning with isolated validation and shadow promotion, quarantine of affected results, risk-based routing, and continuously updated evidence that each parser is still right.

## Quick start

Python 3.10+.

```bash
pip install -r requirements.txt
python -m pytest -q
python -m distill.bench                 # synthetic benchmark -> results/benchmark.md
python -m distill.server                # live dashboard on http://127.0.0.1:8000
```

Real datasets:

```bash
# DocILE: get a token at https://docile.rossum.ai, then
curl -O https://docile-dataset-rossum.s3.eu-west-1.amazonaws.com/<TOKEN>/labeled-trainval.zip
python scripts/prepare_docile.py labeled-trainval.zip      # -> data/docile_trainval_words.jsonl.gz
python -m distill.realdata docile

# SROIE: images + keys from the ICDAR 2019 SROIE release, OCR with Tesseract into data/sroie_ocr/*.txt
python -m distill.realdata sroie --sroie-ocr data/sroie_ocr --sroie-keys data/sroie_keys
```

With real Claude (costs money; Haiku 5.5 is $0.10 / $0.50 per million input/output tokens, Sonnet 5.5 $2 / $10):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/smoke_real.py 3
python -m distill.realdata docile --real --limit 400
python -m distill.bench --real --n 400 --systems distill,llm_only_small
python -m distill.server --real
```

## The live demo

Press **Start stream** and let it reach about 1,000 invoices. The **verification panel** shows every recent invoice with how it was processed, Distill's own verdict, and the answer-key verdict, plus the running count of "accepted, but wrong". Click any invoice to see extracted values next to the truth, the checks, the AI calls and the decision trail. Then let a judge change any vendor's template, including the silent changes, and watch the feed: drift, quarantine, candidate, shadow agreements, promotion. Script: [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md).

## Code map

| File | What it does |
|---|---|
| `distill/model.py` | Schemas and records for any document type; field comparison |
| `distill/doctypes.py` | Invoice, receipt (SROIE) and DocILE document types: fields, checks, source identification |
| `distill/parser.py` | Spec interpreter: line rules, constant fields, zone rules, OCR-tolerant values, fuzzy labels, multi-page tables |
| `distill/synth.py` | Candidate generation (strict, relaxed, geometric) and legibility-aware validation |
| `distill/zones.py` | Geometric templates with page registration |
| `distill/ocr.py` | OCR damage model and noise recognition |
| `distill/engine.py` | Routing, audits, arbitration, noise-vs-drift, quarantine, synthesis, shadow, promotion, rollback |
| `distill/checks.py`, `policy.py`, `audit.py`, `drift.py`, `fingerprint.py` | Checks, routing policy, audit statistics, drift monitor, layout fingerprints and clustering |
| `distill/datagen.py` | Synthetic invoice stream with scan quality, layout quirks and drift mutations |
| `distill/realdata.py`, `scripts/prepare_docile.py` | Public-dataset loaders, scoring and reports |
| `distill/bench.py`, `distill/server.py`, `web/index.html` | Benchmark and live dashboard |

## Limitations

- Model-tier accuracy in the synthetic benchmark is simulated, and the public-dataset runs use a perfect teacher. Run with `--real` for numbers against real Claude.
- Line items are extracted for invoices only; DocILE line items (LIR) are not attempted.
- Values spanning several text lines are handled only as constants (vendor name, address) or by geometric zones.
- DocILE's layout clusters are used as the source identity in the main DocILE run, the way an AP system knows the vendor; `--cluster learned` uses unsupervised clustering instead.
- DocILE and SROIE are distributed under their own terms; the converted files are for local experiments.
