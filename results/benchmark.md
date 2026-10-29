# Distill benchmark (8,000 documents, SIMULATED LLM tiers)

| System | Doc accuracy | Wrong docs | LLM cost | Cost / 1k docs (last 1k) | LLM calls / doc | Code share (last 1k) | p50 latency | Human review |
|---|---|---|---|---|---|---|---|---|
| llm_only_large | 98.81% | 95 | $19.0925 | $2.3854 | 1.000 | 0% | 3792.0 ms | 71 |
| llm_only_small | 98.15% | 148 | $1.5394 | $0.1948 | 1.030 | 0% | 1364.0 ms | 8 |
| static | 87.89% | 969 | $0.4820 | $0.0846 | 0.316 | 60% | 0.1 ms | 0 |
| distill | 99.95% | 4 | $0.2609 | $0.0243 | 0.143 | 99% | 0.1 ms | 1 |

## Drift events (wrong documents from the drifted vendor after the change)

| Vendor | Drift | Class | llm_only_large | llm_only_small | static | distill | Distill: detected after | Distill: recovered after |
|---|---|---|---|---|---|---|---|---|
| V14 | inline_swap | silent | 10/788 | 12/788 | 788/788 | 0/788 | 1 docs | 6 docs |
| V18 | date_locale | silent | 5/543 | 10/543 | 112/543 | 0/543 | 1 docs | 6 docs |
| V20 | relabel | loud | 14/1222 | 17/1222 | 17/1222 | 0/1222 | 1 docs | 7 docs |
| V11 | table_reflow | loud | 7/412 | 5/412 | 5/412 | 0/412 | 1 docs | 6 docs |
| V10 | number_style | loud | 4/300 | 7/300 | 7/300 | 0/300 | 1 docs | 8 docs |
| V12 | new_fields | structural | 3/250 | 8/250 | 0/250 | 0/250 | not needed | not needed |

'Not needed': the change added lines the parser did not use; Distill checked those documents with the model, found the parser still right, and learned the new layout without replacing the parser.

_LLM tiers are simulated (ground truth + injected errors: small 5%, large 1.2% of documents; tokens billed at published Haiku 5.5 / Sonnet 5.5 prices). Parsers, checks, drift detection, routing and audits are real._