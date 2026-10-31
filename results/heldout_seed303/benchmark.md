# Distill benchmark (8,000 documents, SIMULATED LLM tiers)

| System | Doc accuracy | Wrong docs | LLM cost | Cost / 1k correct docs | Tokens / doc | Cost / 1k docs (last 1k) | LLM calls / doc | Code share (last 1k) | p50 latency | Human review |
|---|---|---|---|---|---|---|---|---|---|---|
| llm_only_large | 89.54% | 837 | $29.7159 | $4.1485 | 922 | $3.7041 | 1.002 | 0% | 2199.0 ms | 778 |
| llm_only_small | 91.67% | 666 | $7.2116 | $0.9833 | 1,095 | $0.9400 | 1.189 | 0% | 1873.0 ms | 456 |
| static | 72.74% | 2181 | $5.0212 | $0.8629 | 604 | $0.7151 | 0.647 | 45% | 1390.0 ms | 66 |
| distill | 94.90% | 408 | $2.3626 | $0.3112 | 679 | $0.2979 | 0.750 | 64% | 1530.2 ms | 154 |

## Drift events (wrong documents from the drifted vendor after the change)

| Vendor | Drift | Class | llm_only_large | llm_only_small | static | distill | Distill: detected after | Distill: recovered after |
|---|---|---|---|---|---|---|---|---|
| V13 | inline_swap | silent | 132/1402 | 94/1402 | 1225/1402 | 58/1402 | 3 docs | 11 docs |
| V02 | date_locale | silent | 59/775 | 48/775 | 48/775 | 28/775 | 4 docs | 11 docs |
| V19 | relabel | loud | 5/491 | 7/491 | 7/491 | 1/491 | 1 docs | 6 docs |
| V10 | table_reflow | loud | 35/379 | 29/379 | 29/379 | 29/379 | 1 docs | 8 docs |
| V06 | number_style | loud | 71/292 | 60/292 | 60/292 | 44/292 | - | 246 docs |
| V07 | new_fields | structural | 5/284 | 7/284 | 0/284 | 0/284 | not needed | not needed |

'Not needed': the change added lines the parser did not use; Distill checked those documents with the model, found the parser still right, and learned the new layout without replacing the parser.

Repaired from the document's own invariants instead of a (further) model call: 1,988 of 8,000 documents (24.9%); 1,898 of them exactly right (95.5%).

Scoped escalation: when the small model's answer fails only some checks, the large model is asked only for the fields (or the line-item rows) behind them, not the whole record. 437 such calls, 320 of them settled every check; answers averaged 58 output tokens against 259 for a full re-read (117 full re-reads remained).

_LLM tiers are simulated (ground truth + injected errors: small 5%, large 1.2% of documents; tokens billed at published Haiku 5.5 / Sonnet 5.5 prices). Parsers, checks, drift detection, routing and audits are real._