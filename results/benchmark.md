# Distill benchmark (8,000 documents, SIMULATED LLM tiers)

| System | Doc accuracy | Wrong docs | LLM cost | Cost / 1k correct docs | Tokens / doc | Cost / 1k docs (last 1k) | LLM calls / doc | Code share (last 1k) | p50 latency | Human review |
|---|---|---|---|---|---|---|---|---|---|---|
| llm_only_large | 94.10% | 472 | $28.3361 | $3.7641 | 891 | $3.5416 | 1.001 | 0% | 2158.5 ms | 444 |
| llm_only_small | 94.74% | 421 | $4.7207 | $0.6229 | 992 | $0.5916 | 1.109 | 0% | 1656.0 ms | 245 |
| llm_only_small_repair | 96.09% | 313 | $2.7157 | $0.3533 | 930 | $0.3278 | 1.043 | 0% | 1649.0 ms | 118 |
| static | 81.17% | 1506 | $3.0680 | $0.4724 | 508 | $0.4285 | 0.565 | 43% | 0.7 ms | 43 |
| distill | 96.84% | 253 | $1.4220 | $0.1835 | 454 | $0.1625 | 0.492 | 79% | 0.3 ms | 102 |

## Drift events (wrong documents from the drifted vendor after the change)

| Vendor | Drift | Class | llm_only_large | llm_only_small | llm_only_small_repair | static | distill | Distill: detected after | Distill: recovered after |
|---|---|---|---|---|---|---|---|---|---|
| V01 | inline_swap | silent | 32/430 | 17/430 | 14/430 | 378/430 | 18/430 | 3 docs | 10 docs |
| V10 | date_locale | silent | 19/1316 | 27/1316 | 27/1316 | 313/1316 | 1/1316 | 4 docs | 10 docs |
| V19 | relabel | loud | 66/724 | 44/724 | 28/724 | 44/724 | 34/724 | 5 docs | 14 docs |
| V07 | table_reflow | loud | 10/516 | 10/516 | 10/516 | 10/516 | 4/516 | 4 docs | 9 docs |
| V12 | number_style | loud | 31/328 | 30/328 | 16/328 | 30/328 | 19/328 | 4 docs | 28 docs |
| V13 | new_fields | structural | 3/271 | 8/271 | 8/271 | 169/271 | 5/271 | - | - |

'Not needed': the change added lines the parser did not use; Distill checked those documents with the model, found the parser still right, and learned the new layout without replacing the parser.

Repaired from the document's own invariants instead of a (further) model call: 1,528 of 8,000 documents (19.1%); 1,479 of them exactly right (96.8%).

Scoped escalation: when the small model's answer fails only some checks, the large model is asked only for the fields (or the line-item rows) behind them, not the whole record. 225 such calls, 158 of them settled every check; answers averaged 59 output tokens against 248 for a full re-read (67 full re-reads remained).

_LLM tiers are simulated (ground truth + injected errors: small 5%, large 1.2% of documents; tokens billed at published Haiku 5.5 / Sonnet 5.5 prices). Parsers, checks, drift detection, routing and audits are real._