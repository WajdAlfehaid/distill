# Distill benchmark (8,000 documents, SIMULATED LLM tiers)

| System | Doc accuracy | Wrong docs | LLM cost | Cost / 1k correct docs | Tokens / doc | Cost / 1k docs (last 1k) | LLM calls / doc | Code share (last 1k) | p50 latency | Human review |
|---|---|---|---|---|---|---|---|---|---|---|
| llm_only_large | 95.25% | 380 | $29.3142 | $3.8470 | 920 | $3.6532 | 1.001 | 0% | 2167.5 ms | 347 |
| llm_only_small | 95.73% | 342 | $4.0567 | $0.5297 | 1,001 | $0.5252 | 1.091 | 0% | 1754.0 ms | 186 |
| static | 81.40% | 1488 | $2.3998 | $0.3685 | 471 | $0.3546 | 0.523 | 46% | 0.3 ms | 26 |
| distill | 98.00% | 160 | $0.8749 | $0.1116 | 263 | $0.1266 | 0.302 | 86% | 0.2 ms | 65 |

## Drift events (wrong documents from the drifted vendor after the change)

| Vendor | Drift | Class | llm_only_large | llm_only_small | static | distill | Distill: detected after | Distill: recovered after |
|---|---|---|---|---|---|---|---|---|
| V01 | inline_swap | silent | 3/318 | 3/318 | 318/318 | 0/318 | 1 docs | 6 docs |
| V02 | date_locale | silent | 14/1298 | 29/1298 | 288/1298 | 4/1298 | 4 docs | 9 docs |
| V14 | relabel | loud | 9/705 | 17/705 | 17/705 | 0/705 | 1 docs | 6 docs |
| V03 | table_reflow | loud | 4/475 | 14/475 | 14/475 | 3/475 | 1 docs | 7 docs |
| V08 | number_style | loud | 2/357 | 5/357 | 5/357 | 1/357 | 4 docs | 9 docs |
| V19 | new_fields | structural | 86/286 | 60/286 | 60/286 | 38/286 | 97 docs | 90 docs |

'Not needed': the change added lines the parser did not use; Distill checked those documents with the model, found the parser still right, and learned the new layout without replacing the parser.

Repaired from the document's own invariants instead of a (further) model call: 1,267 of 8,000 documents (15.8%); 1,237 of them exactly right (97.6%).

Scoped escalation: when the small model's answer fails only some checks, the large model is asked only for the fields (or the line-item rows) behind them, not the whole record. 163 such calls, 116 of them settled every check; answers averaged 57 output tokens against 225 for a full re-read (47 full re-reads remained).

_LLM tiers are simulated (ground truth + injected errors: small 5%, large 1.2% of documents; tokens billed at published Haiku 5.5 / Sonnet 5.5 prices). Parsers, checks, drift detection, routing and audits are real._