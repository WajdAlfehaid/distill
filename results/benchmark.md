# Distill benchmark (8,000 documents, SIMULATED LLM tiers)

| System | Doc accuracy | Wrong docs | LLM cost | Cost / 1k docs (last 1k) | LLM calls / doc | Code share (last 1k) | p50 latency | Human review |
|---|---|---|---|---|---|---|---|---|
| llm_only_large | 94.14% | 469 | $24.6894 | $3.0848 | 1.000 | 0% | 4512.0 ms | 444 |
| llm_only_small | 94.77% | 418 | $4.1817 | $0.5256 | 1.108 | 0% | 1604.0 ms | 245 |
| static | 80.80% | 1536 | $2.7351 | $0.3750 | 0.557 | 46% | 0.7 ms | 43 |
| distill | 96.19% | 305 | $3.7470 | $0.4401 | 0.644 | 74% | 1348.1 ms | 238 |

## Drift events (wrong documents from the drifted vendor after the change)

| Vendor | Drift | Class | llm_only_large | llm_only_small | static | distill | Distill: detected after | Distill: recovered after |
|---|---|---|---|---|---|---|---|---|
| V01 | inline_swap | silent | 32/430 | 17/430 | 393/430 | 16/430 | 2 docs | 9 docs |
| V10 | date_locale | silent | 19/1316 | 27/1316 | 313/1316 | 0/1316 | 3 docs | 8 docs |
| V19 | relabel | loud | 64/724 | 42/724 | 42/724 | 38/724 | 3 docs | 19 docs |
| V07 | table_reflow | loud | 10/516 | 10/516 | 10/516 | 0/516 | 3 docs | 8 docs |
| V12 | number_style | loud | 31/328 | 30/328 | 30/328 | 29/328 | 1 docs | 21 docs |
| V13 | new_fields | structural | 3/271 | 8/271 | 169/271 | 0/271 | not needed | not needed |

'Not needed': the change added lines the parser did not use; Distill checked those documents with the model, found the parser still right, and learned the new layout without replacing the parser.

_LLM tiers are simulated (ground truth + injected errors: small 5%, large 1.2% of documents; tokens billed at published Haiku 5.5 / Sonnet 5.5 prices). Parsers, checks, drift detection, routing and audits are real._