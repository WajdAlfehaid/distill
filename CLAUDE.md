# Distill: notes for Claude Code

Hackathon project (Challenge 2, "Deliver AI with less compute"). Document extraction that compiles LLM work into
JSON parser specs (text-line patterns or registered geometric templates), then detects drift, tells OCR noise from
real drift, re-synthesizes, shadow-tests, promotes and keeps auditing. See README.md.

## Commands
- `python -m pytest -q` — must stay green
- `python -m distill.bench` — synthetic invoices with realistic scan noise, 4 systems x 8,000 docs; writes results/
- add `--real` to use real Claude (ANTHROPIC_API_KEY); without it the model tier is simulated or a perfect teacher
- `python -m distill.server [--real] [--port 8000]` — dashboard (web/index.html), SSE at /api/stream
- `python -m distill.server --dataset fatura --warm 9000` — dashboard on FATURA (data/fatura_invoices.jsonl.gz, see scripts/prepare_fatura.py)
- `python -m distill.realdata fatura` — FATURA benchmark (third-party invoices, 50 layouts)
- `python scripts/smoke_real.py 3` — real-API sanity check

## Rules
- Dependencies: standard library + starlette + uvicorn only. The Claude client is urllib-based on purpose.
- Parsers are data (JSON specs). Never execute model-written code.
- Every number shown to judges must be reproducible from a script; label simulated numbers as simulated.
- Ground truth is only for scoring and the simulated LLM oracle. The engine must never read it.
- `Engine.process(doc_id, text, received_at, meta)` is the single entry point; `meta["words"]` carries OCR word boxes.

## Code map
- model.py: Schema/FieldSpec/Record (generic records; `Invoice(...)` builds an invoice Record), comparison
- doctypes.py: invoice and FATURA schemas, their checks and source identification
- parser.py: spec executor (line rules, const rules, zone rules, OCR-tolerant numbers, fuzzy labels, multi-page items)
- synth.py: induction of candidate specs (strict, relaxed, zones), legibility-aware validation
- zones.py: geometric templates with page registration (align scans to a reference page)
- ocr.py: OCR corruption model + noise recognition (ocr_equivalent, fuzzy_label)
- engine.py: routing, audits, arbitration, noise-vs-drift classification, quarantine, synthesis, shadow, promotion, scoped escalation
- repair.py: invariant repair of a damaged value, accepted only when the fix is unique
- datagen.py: synthetic invoice stream with per-vendor scan quality, wrapped rows, page breaks, drift mutations
- realdata.py: FATURA loader, scoring against any annotated value, report
- bench.py, server.py, web/index.html: benchmark and dashboard

## Open work, in priority order
1. Real-API runs: `python -m distill.realdata fatura --real --limit 400` and `python -m distill.bench --real --n 400`.
2. Batch API (50% cheaper) for audits and quarantine re-checks.
3. Multi-line values in line rules.
