# Distill benchmark (4,000 documents, SIMULATED LLM tiers)

| System | Doc accuracy | Wrong docs | LLM cost | Cost / 1k docs (last 1k) | LLM calls / doc | Code share (last 1k) | p50 latency | Human review |
|---|---|---|---|---|---|---|---|---|
| llm_only_large | 99.02% | 39 | $9.5087 | $2.3668 | 1.000 | 0% | 3792.0 ms | 29 |
| llm_only_small | 98.10% | 76 | $0.7842 | $0.1823 | 1.031 | 0% | 1364.0 ms | 4 |
| static | 89.12% | 435 | $0.1613 | $0.0820 | 0.202 | 60% | 0.1 ms | 0 |
| distill | 99.90% | 4 | $0.3089 | $0.0559 | 0.316 | 96% | 0.1 ms | 0 |

## Drift events (wrong documents from the drifted vendor after the change)

| Vendor | Drift | Class | llm_only_large | llm_only_small | static | distill | Distill: detected after | Distill: recovered after |
|---|---|---|---|---|---|---|---|---|
| V14 | inline_swap | silent | 0/308 | 4/308 | 308/308 | 0/308 | 1 docs | 6 docs |
| V18 | date_locale | silent | 1/198 | 4/198 | 82/198 | 0/198 | 1 docs | 6 docs |
| V20 | relabel | loud | 2/378 | 3/378 | 3/378 | 0/378 | 1 docs | 7 docs |
| V11 | table_reflow | loud | 1/126 | 2/126 | 2/126 | 0/126 | 1 docs | 6 docs |
| V10 | number_style | loud | 1/76 | 0/76 | 0/76 | 0/76 | 1 docs | 8 docs |
| V12 | new_fields | structural | 0/46 | 1/46 | 0/46 | 0/46 | - docs | - docs |

_LLM tiers are simulated (ground truth + injected errors: small 5%, large 1.2% of documents; tokens billed at published Haiku 5.5 / Sonnet 5.5 prices). Parsers, checks, drift detection, routing and audits are real._