# Distill benchmark (400 documents, real Claude calls)

| System | Doc accuracy | Wrong docs | LLM cost | Cost / 1k correct docs | Tokens / doc | Cost / 1k docs (last 1k) | LLM calls / doc | Code share (last 1k) | p50 latency | Human review |
|---|---|---|---|---|---|---|---|---|---|---|
| llm_only_small | 79.00% | 84 | $0.6267 | $1.9831 | 1,524 | $0.6267 | 1.198 | 0% | 3136.9 ms | 74 |
| distill | 91.25% | 35 | $0.3551 | $0.9730 | 1,193 | $0.3552 | 0.948 | 32% | 2821.7 ms | 23 |

## Drift events (wrong documents from the drifted vendor after the change)

| Vendor | Drift | Class | llm_only_small | distill | Distill: detected after | Distill: recovered after |
|---|---|---|---|---|---|---|
| V01 | inline_swap | silent | 4/31 | 4/31 | - | 13 docs |
| V10 | date_locale | silent | 45/67 | 7/67 | 7 docs | 17 docs |
| V19 | relabel | loud | 7/33 | 4/33 | - | - |
| V07 | table_reflow | loud | 0/17 | 0/17 | 4 docs | 9 docs |
| V12 | number_style | loud | 1/9 | 1/9 | - | - |
| V13 | new_fields | structural | 0/10 | 0/10 | not needed | not needed |

'Not needed': the change added lines the parser did not use; Distill checked those documents with the model, found the parser still right, and learned the new layout without replacing the parser.

Repaired from the document's own invariants instead of a (further) model call: 42 of 400 documents (10.5%); 39 of them exactly right (92.9%).

Scoped escalation: when the small model's answer fails only some checks, the large model is asked only for the fields (or the line-item rows) behind them, not the whole record. 26 such calls, 3 of them settled every check; answers averaged 185 output tokens against 484 for a full re-read (23 full re-reads remained).