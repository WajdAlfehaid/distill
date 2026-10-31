# Distill benchmark (8,000 documents, SIMULATED LLM tiers)

| System | Doc accuracy | Wrong docs | LLM cost | Cost / 1k correct docs | Tokens / doc | Cost / 1k docs (last 1k) | LLM calls / doc | Code share (last 1k) | p50 latency | Human review |
|---|---|---|---|---|---|---|---|---|---|---|
| distill | 96.84% | 253 | $1.4220 | $0.1835 | 454 | $0.1625 | 0.492 | 79% | 0.4 ms | 102 |
| distill_no_repair | 96.29% | 297 | $3.2612 | $0.4234 | 599 | $0.4061 | 0.665 | 72% | 1400.8 ms | 197 |
| distill_no_scoped | 96.84% | 253 | $1.6013 | $0.2067 | 452 | $0.1836 | 0.484 | 79% | 0.3 ms | 111 |

## Drift events (wrong documents from the drifted vendor after the change)

| Vendor | Drift | Class | distill | distill_no_repair | distill_no_scoped | Distill: detected after | Distill: recovered after |
|---|---|---|---|---|---|---|---|
| V01 | inline_swap | silent | 18/430 | 20/430 | 17/430 | 3 docs | 10 docs |
| V10 | date_locale | silent | 1/1316 | 1/1316 | 1/1316 | 4 docs | 10 docs |
| V19 | relabel | loud | 34/724 | 41/724 | 34/724 | 5 docs | 14 docs |
| V07 | table_reflow | loud | 4/516 | 0/516 | 4/516 | 4 docs | 9 docs |
| V12 | number_style | loud | 19/328 | 27/328 | 18/328 | 4 docs | 28 docs |
| V13 | new_fields | structural | 5/271 | 0/271 | 5/271 | - | - |

'Not needed': the change added lines the parser did not use; Distill checked those documents with the model, found the parser still right, and learned the new layout without replacing the parser.

Repaired from the document's own invariants instead of a (further) model call: 1,527 of 8,000 documents (19.1%); 1,478 of them exactly right (96.8%).

Scoped escalation: when the small model's answer fails only some checks, the large model is asked only for the fields (or the line-item rows) behind them, not the whole record. 225 such calls, 158 of them settled every check; answers averaged 59 output tokens against 248 for a full re-read (67 full re-reads remained).

_LLM tiers are simulated (ground truth + injected errors: small 5%, large 1.2% of documents; tokens billed at published Haiku 5.5 / Sonnet 5.5 prices). Parsers, checks, drift detection, routing and audits are real._