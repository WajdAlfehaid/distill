# Distill benchmark (8,000 documents, SIMULATED LLM tiers)

| System | Doc accuracy | Wrong docs | LLM cost | Cost / 1k correct docs | Tokens / doc | Cost / 1k docs (last 1k) | LLM calls / doc | Code share (last 1k) | p50 latency | Human review |
|---|---|---|---|---|---|---|---|---|---|---|
| llm_only_large | 94.05% | 476 | $30.2248 | $4.0171 | 918 | $3.8086 | 1.000 | 0% | 2311.5 ms | 423 |
| llm_only_small | 95.30% | 376 | $4.9569 | $0.6502 | 1,023 | $0.6258 | 1.114 | 0% | 1880.0 ms | 200 |
| static | 80.10% | 1592 | $2.8478 | $0.4444 | 491 | $0.3945 | 0.538 | 45% | 0.5 ms | 24 |
| distill | 97.24% | 221 | $1.2905 | $0.1659 | 453 | $0.1581 | 0.489 | 80% | 0.4 ms | 63 |

## Drift events (wrong documents from the drifted vendor after the change)

| Vendor | Drift | Class | llm_only_large | llm_only_small | static | distill | Distill: detected after | Distill: recovered after |
|---|---|---|---|---|---|---|---|---|
| V01 | inline_swap | silent | 7/589 | 16/589 | 589/589 | 1/589 | 1 docs | 6 docs |
| V04 | date_locale | silent | 119/1327 | 82/1327 | 82/1327 | 76/1327 | 4 docs | 26 docs |
| V19 | relabel | loud | 8/726 | 17/726 | 17/726 | 1/726 | 1 docs | 6 docs |
| V15 | table_reflow | loud | 10/378 | 5/378 | 5/378 | 2/378 | 4 docs | 9 docs |
| V14 | number_style | loud | 8/329 | 7/329 | 7/329 | 0/329 | 4 docs | 9 docs |
| V12 | new_fields | structural | 6/249 | 2/249 | 0/249 | 0/249 | not needed | not needed |

'Not needed': the change added lines the parser did not use; Distill checked those documents with the model, found the parser still right, and learned the new layout without replacing the parser.

Repaired from the document's own invariants instead of a (further) model call: 1,339 of 8,000 documents (16.7%); 1,279 of them exactly right (95.5%).

Scoped escalation: when the small model's answer fails only some checks, the large model is asked only for the fields (or the line-item rows) behind them, not the whole record. 189 such calls, 142 of them settled every check; answers averaged 55 output tokens against 251 for a full re-read (47 full re-reads remained).

_LLM tiers are simulated (ground truth + injected errors: small 5%, large 1.2% of documents; tokens billed at published Haiku 5.5 / Sonnet 5.5 prices). Parsers, checks, drift detection, routing and audits are real._