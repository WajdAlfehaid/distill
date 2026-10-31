# Distill

**Document extraction that compiles itself into code where that is safe, and knows when it isn't.**

Companies send the same kinds of documents through language models over and over, even though most suppliers' layouts barely change. Distill reads a source's first documents with a model, compiles what it learned into a small parser (JSON data, never generated code), and from then on extracts that source's documents without AI, while checking every result, auditing its parsers against the model, and telling scan damage apart from a real change of template.

## What it does

1. **Compile.** From a few verified examples, Distill induces candidate parsers: text-line patterns for text documents, and *registered geometric templates* for scanned pages (each page is aligned to a reference copy of its layout using shared landmark words, then fields are read from learned zones). Candidates are validated in isolation, only against values that are actually legible in each example, then run in **shadow mode** next to live traffic before promotion.
2. **Repair before paying.** An invoice is an error-correcting code: line math, subtotal, discount, tax rate, total and payment terms all constrain each other. When a scan smudges one value (or the small model slips on one), Distill recomputes it from the others and accepts the repair only if it is the *single* change that makes every check pass; ambiguous damage still goes to the model. In the benchmark this repaired 1,529 of 8,000 invoices (19.1%) instead of paying for a model call, and 96.8% of them came out exactly right, against 94.7% for the model-only systems. When the parser's reading and the model's both fail the checks, Distill also tries combining them field by field and accepts the one combination that passes.
3. **Check everything.** Arithmetic (line math, subtotal, discount, tax, total), formats, master data (vendor tax ID, currency, tax rate, payment terms, PO), duplicates, and the vendor's own price and quantity history.
4. **Tell noise from drift.** Scans damage documents: look-alike characters (0/O, 1/l, 5/S), smudged labels, dropped digits. Distill recognises these: values one look-alike or one slip apart, labels within an edit budget, a parser output that already failed the checks on an unchanged layout. Only a **clean** disagreement confirmed by an arbiter model, or the **same field** disagreeing copy after copy, suspends a parser.
5. **Recover.** On real drift: suspend the parser, quarantine and re-check what it accepted since its last verified result, synthesize a replacement from fresh examples, shadow-test it, promote it. A parser that has simply never seen a field (a vendor's first discount) is kept and extended, not suspended.
6. **Keep proving reliability.** Audits compare parsers with the model at a rate that adapts to evidence (100% for a new parser, halving every 3 clean audits, 2% floor) and spike on risk. Each parser carries an exact 95% upper bound on its error rate.
7. **Route by policy.** Each document's path (code, code + audit, small model, large model, person) is chosen from risk, uncertainty, cost and latency, and the reasons are recorded with the result.

## How it answers Challenge 2 ("Deliver AI with less compute")

| Challenge point | What Distill does | Measured (generated benchmark, 8,000 invoices) |
|---|---|---|
| Model choice per task | Code for compiled layouts, Haiku for new or uncertain ones, Sonnet only when Haiku's answer fails its checks or for arbitration | 79% of steady-state invoices read by code; 0.49 model calls per invoice |
| Context that earns its tokens | Compiled invoices send no tokens at all; escalations ask Sonnet only for the fields or line rows behind the failing checks | 454 tokens per invoice vs 992 for Haiku; Sonnet answers shrink from 248 to 59 tokens |
| Verification where it pays | Free arithmetic and master-data checks on every result; model spot-checks that thin out as a parser proves itself and rise for high-value or risky invoices; Sonnet only for disputes the checks cannot settle | 19.1% of invoices repaired from the checks without a model call, 96.8% of them right |
| Cost per successful task | Repair and checks turn would-be failures into right answers instead of re-runs | $0.18 per 1,000 correct invoices vs $0.62 (Haiku) and $3.76 (Sonnet) |
| Accuracy and latency floors | More accurate than either model alone; compiled invoices answer in under a millisecond | 96.8% vs 94.7% (Haiku) and 94.1% (Sonnet); median 0.3 ms vs 1.7 s |

On FATURA (10,000 third-party invoices) Distill is 3.0x cheaper than Haiku on every invoice. Model tiers in these benchmarks are simulated, calibrated on real Claude calls; on DocILE with the real API Distill cost $0.53 against $2.23 for Sonnet.

## Results

Five evaluations, each answering a different question. All numbers come from scripts in this repository.

### 1. Synthetic invoice stream with realistic scan damage, and injected template changes

`python -m distill.bench`: 8,000 invoices from 20 vendors (5 countries, 5 table styles, wrapped descriptions, tables continued on a second page, free-text notes). Each vendor is a born-digital PDF, a good scan or a poor scan, with character-level OCR damage. Six template changes are injected mid-stream. **The model tiers are simulated**: ground truth plus injected mistakes, which grow with document length and with damage to printed values; tokens are billed at published Claude Haiku 5.5 / Sonnet 5.5 prices.

| System | Exactly right | AI cost | Cost per 1k correct invoices | Tokens per invoice | Model calls per invoice | Typical time per invoice |
|---|---|---|---|---|---|---|
| Claude Sonnet on every invoice | 94.1% | $28.34 | $3.76 | 891 | 1.00 | 2.2 s |
| Claude Haiku on every invoice (Sonnet when checks fail) | 94.7% | $4.72 | $0.62 | 992 | 1.11 | 1.7 s |
| Same, plus Distill's invariant repair (ablation) | 96.1% | $2.72 | $0.35 | 930 | 1.04 | 1.6 s |
| Compile once, no drift handling | 81.2% | $3.07 | $0.47 | 508 | 0.57 | 0.7 ms |
| **Distill** | **96.8%** | **$1.42** | **$0.18** | **454** | **0.49** | **0.3 ms** |

Code handles 79% of steady-state traffic, and 19.1% of invoices had a smudged or misread value repaired for free (96.8% of those exactly right). Disagreements in audits and shadow tests are settled with the checks first; the larger model is asked only when both readings pass every check and still differ. When Haiku's answer fails only some checks, Sonnet is asked only for the fields or line-item rows behind them (59 output tokens on average instead of 248 for a full re-read); 158 of 225 such calls settled every check, and a full re-read follows only when they do not. Template changes were detected within 3–5 invoices of the changed vendor (a single disagreement only raises suspicion; a second copy confirms it) and a validated replacement was live within 9–28. The compile-once baseline got 378 of 430 invoices wrong after a silent change (price and quantity swapped, every total still adds up); Distill got 18 wrong, in line with the model-only systems on the same vendor (17 and 32), whose errors come from scan damage.

**What this says, honestly:** per correct invoice, Distill costs 3.4x less than the Haiku pipeline and 20x less than Sonnet, with 56% fewer model calls, 54% fewer tokens and 40% fewer wrong invoices. The ablation row shows both halves pay: invariant repair alone takes the Haiku pipeline from $4.72 to $2.72, and compiling on top of it takes it to $1.42. The remaining model calls go mostly to damaged documents, which are the expensive ones for every system. Times are the median per invoice; model times are simulated from real calls.

### 2. FATURA: 10,000 invoices from 50 layouts, made by other researchers

`python scripts/prepare_fatura.py FATURA2.zip && python -m distill.realdata fatura`. FATURA (Limam et al. 2023, [Zenodo](https://zenodo.org/records/10371464), CC BY 4.0) has 200 invoices for each of 50 real invoice layouts, with the page's text layer and labelled fields. Its content is generated from the templates (by its authors, not us), and its text layer carries OCR-like damage. Label wordings vary from invoice to invoice ('INVOICE #', 'INVOICE ID', 'Invoice number'), and boxes share printed rows. Teacher: the labels (a model that never errs, the cheapest possible baseline).

| System | Handled by code | Wrong but accepted by code | Model calls / invoice | AI cost |
|---|---|---|---|---|
| Haiku on every invoice | 0% | 0 | 1.00 | $1.07 |
| Compile once, no drift handling | 54.8% | 117 | 0.45 | $0.49 |
| **Distill** | **82.0%** | **113** | **0.29** | **$0.36** |

Distill is **3.0x cheaper** than reading every invoice with Haiku, cheaper than compile-once as well, and handles 86% of a layout's invoices after its first ten. Of the 113 compiled answers that differ from the labels, 109 are values the text layer itself prints damaged (a model is given the same text); on values printed legibly, 4 of 8,201 compiled answers are wrong (0.05%).

What it took: labels learned as a set of wordings, labels found after other text on the same printed row, a missing value treated as "learn the new wording" rather than drift, new content lines that keep proving harmless (a new buyer's address) no longer sending a vendor's invoices to the model, and a disagreement where the parser's value is printed on the page but the model's is not counted as damaged print, not as a broken parser.

### 3. DocILE: 5,680 real business documents (Rossum, ICDAR 2023)

`python -m distill.realdata docile`: invoices, orders, credit notes and utility bills with DocTR OCR, 1,063 layout clusters. The model tier is a **perfect teacher** (the human labels) so the run isolates the parser layer: how much real traffic compiled templates can take over, and how often they disagree with the teacher.

| System | Handled by code | Right when handled by code | Wrong but accepted by code | Parsers promoted |
|---|---|---|---|---|
| Compile once, no drift handling | 4.4% | 32.1% | 171 | 37 |
| **Distill** | 0.3% | 100% | **0** | 9 |

This is the most important result in the repository, and it is sobering. DocILE's "layouts" are much looser than real templates: copies of one layout label different fields and place values differently, and the same value is often printed in several places. A compiler that trusts what it learned (compile-once) takes 4.4% of traffic and gets two thirds of it wrong. Distill's validation, shadow testing and noise-versus-drift logic refuse almost everything, so it makes no silent errors but saves nothing on this data.

What this means for the product: compilation pays off on high-volume, consistently templated sources (the situation in result 1, and typical for a company's top suppliers), and the hard engineering problem is knowing *when* that is the case. Raising coverage on DocILE-like data (per-field partial templates with targeted model calls for the remaining fields, more examples per layout) is the main open work.

### 4. SROIE: 619 real scanned receipts (ICDAR 2019), Tesseract OCR

`python -m distill.realdata sroie`: faded thermal receipts from 236 shops, OCR'd here with Tesseract, identified by unsupervised layout clustering. Distill compiled **no** parsers: nothing it learned held up across copies of these scans, so every receipt went to the model and none was accepted wrongly. The compile-once baseline compiled anyway and was wrong on 9 of the 17 receipts it handled. This is the behaviour we want on data that should not be compiled.

### 5. Real Claude on DocILE, and the cheapest correct way to call it

`python -m distill.realdata docile --real --clusters 10 --limit 300` (300 documents from the 10 largest layouts; answers cached in `results/llm_cache_docile.jsonl`, so the run replays for free). Haiku 5.5 reads every document; documents whose answer fails the checks are re-read by Sonnet 5.5 (17%).

| | Answerable labelled fields right | Labelled fields right | LLM cost (300 docs) | Haiku on all / Sonnet on all |
|---|---|---|---|---|
| Claude (Haiku, Sonnet when checks fail) | 72.2% | 68.1% | $0.53 | $0.15 / $2.23 (measured) |
| Distill | 72.2% | 68.1% | $0.54 | |

Two findings. First, a fair score on DocILE needs care: 20% of labelled values do not appear in DocILE's own OCR text (cut or split numbers such as `$11,3` for `$11,317.00`), and many "errors" are labelling conventions (the address block includes the company name; `$` rather than `USD`; the agency, not the advertiser, as the billed customer). Second, with a real model as teacher **neither Distill nor the compile-once baseline compiled a single parser** here (with perfect labels, compile-once had compiled 37): the model's answers vary in form from copy to copy (which name line, which of two totals), so no rule is consistent across examples. Distill therefore routed everything to the model, at the model's accuracy, and accepted nothing wrong.

Thinking settings, measured on the same 10 invoices ([results/thinking_modes.md](results/thinking_modes.md)): with thinking off Haiku got 7/10 right at $0.21 per 1k; with low effort 9/10 at $0.37; with defaults 9/10 at $0.42. A miss costs a Sonnet call (~$4-6 per 1k), so low effort is the cheapest correct setting and is Distill's default.

## How it relates to prior work

Stanford's Evaporate (Arora et al., VLDB 2024) showed that synthesizing extraction functions instead of reading every document cuts the tokens an LLM processes by about 110x, and Evaporate-Code+ already generates and aggregates candidate functions. Template ("zonal") extraction for known layouts is decades old. Distill's contribution is the operating discipline for a **stream** of real, damaged business documents that **change over time**: learning templates automatically from a model, telling scan noise from drift, versioning with isolated validation and shadow promotion, quarantine of affected results, risk-based routing, and continuously updated evidence that each parser is still right.

## Quick start

Python 3.10+.

```bash
pip install -r requirements.txt
python -m pytest -q
python -m distill.bench                 # synthetic benchmark -> results/benchmark.md
python -m distill.server --warm 1500    # live dashboard on http://127.0.0.1:8000, starting with parsers in place
python -m distill.server --dataset fatura --warm 5000   # starts on FATURA; both datasets warmed, switchable in the page
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
python -m distill.realdata docile --real --clusters 10 --limit 300 --systems llm_only,distill
python -m distill.bench --real --n 400 --systems distill,llm_only_small
python -m distill.server --real
```

Real runs read documents 8 at a time (`--workers`), cache every answer in `results/llm_cache_<dataset>.jsonl` (an interrupted run resumes; re-runs and extra systems cost nothing), print progress, save results after each system, and write each document's prediction and annotations to `results/<run>_<system>_predictions.jsonl`. Extraction calls switch extended thinking off and use low effort (`DISTILL_THINKING=on` restores the defaults).

## The live demo

`python -m distill.server --dataset fatura --warm 5000` (about 90 seconds: it processes 5,000 invoices of each dataset first), then open http://127.0.0.1:8000. The page shows one bar per invoice (click one to see how it was handled and why), the cost per 1,000 invoices for Distill, Haiku and Sonnet, the total spend of each, and what Distill did, in light or dark mode. Switch the data to *Generated invoices* (each dataset keeps its own progress) to change a supplier's template live; Distill is not told which one.

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

- Model-tier accuracy in the synthetic benchmark is simulated (its tokens and latency are calibrated on real calls); result 4 uses real Claude.
- On DocILE Distill compiles almost nothing: its layout clusters are shared billing software used by many suppliers, and its OCR cuts values. Per-field partial templates are the main open work.
- Multi-line values (address blocks) are not compiled; FATURA's buyer and seller-address fields are left out for that reason, and its seller name is not in the text layer.
- Line items are extracted for invoices only; DocILE line items (LIR) are not attempted.
- Values spanning several text lines are handled only as constants (vendor name, address) or by geometric zones.
- DocILE's layout clusters are used as the source identity in the main DocILE run, the way an AP system knows the vendor; `--cluster learned` uses unsupervised clustering instead.
- DocILE and SROIE are distributed under their own terms; the converted files are for local experiments.
