# Distill: notes for Claude Code

Hackathon project (Challenge 2, "Deliver AI with less compute"). Invoice extraction that compiles LLM work into
JSON parser specs, then detects drift, re-synthesizes, shadow-tests, promotes and keeps auditing. See README.md.

## Commands
- `python -m pytest -q` — must stay green (21 tests, ~4 s)
- `python -m distill.bench` — 4 systems x 8,000 docs, simulated LLM, ~15 s; writes results/
- `python -m distill.bench --seed 5` — other seeds pick drift vendors automatically (`auto_schedule`)
- `python -m distill.server [--real] [--port 8000]` — dashboard at web/index.html, SSE at /api/stream
- `python scripts/smoke_real.py 3` — real-API sanity check (needs ANTHROPIC_API_KEY)

## Rules
- Dependencies: standard library + starlette + uvicorn only. The Claude client is urllib-based on purpose.
- Parsers are data (JSON specs). Never execute model-written code.
- Every number shown to judges must be reproducible from `distill.bench`; label simulated numbers as simulated.
- Ground truth (`Doc.truth`) is only for scoring and the simulated LLM oracle. The engine must never read it.
- Keep `Engine.process` the single entry point; the benchmark and server both drive it.

## Architecture in one line per stage
fingerprint.identify_vendor -> parser.execute (active spec) -> checks.run_checks -> drift.DriftMonitor.observe
-> policy.decide -> code | _audit (small model, large arbiter) | _llm_extract_with_escalation
-> learn (examples, history, layout) -> _shadow_step (arbitrated) -> _maybe_synthesize (synth.generate_candidates,
best_candidate; fallbacks: fresh+older, fresh, leave-one-out, latest-k) -> promotion; _on_drift -> _quarantine.

## Open work, in priority order
1. Run the real-API benchmark (`--real --n 400 --systems distill,llm_only_small`) and put real numbers in README
   and the pitch. Check `llm.SYSTEM_PROMPT` against real failure modes first with `scripts/smoke_real.py`.
2. Record a backup video of the dashboard demo (silent swap on V14, relabel on V20).
3. Pitch slides from results/benchmark.md (cost chart, drift table, the silent-swap story).
4. Stretch: PDF input (render invoices to PDF with reportlab, extract with pypdf/pdfplumber) so judges see real files.
5. Stretch: Batch API (50% cheaper) for audits and quarantine re-checks, which are not latency-critical.
6. Stretch: multiple concurrent templates per vendor (cluster fingerprints, one parser per cluster).
