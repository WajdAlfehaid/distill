# DOCILE: 300 real documents, 10 sources
Teacher: real Claude.

| System | Docs fully right (answerable fields) | Fields right | Labelled fields right | Answerable labelled fields right | Handled by code | Right when handled by code | Wrong but accepted by code | LLM calls / doc | LLM cost | Cost if every doc went to Haiku / Sonnet |
|---|---|---|---|---|---|---|---|---|---|---|
| llm_only | 11.0% | 72.1% | 68.1% | 72.2% | 0.0% | - | 0 | 1.17 | $0.5307 | $0.1499 / $2.23 |
| static | 11.0% | 72.1% | 68.1% | 72.2% | 0.0% | - | 0 | 1.17 | $0.5307 | $0.1499 / $2.23 |
| distill | 11.0% | 72.1% | 68.1% | 72.2% | 0.0% | - | 0 | 1.17 | $0.5365 | $0.1499 / $2.23 |

## Distill by how many documents it had already seen from that source

| Position | Docs | Handled by code | Fully right |
|---|---|---|---|
| first 3 from a source | 30 | 0.0% | 0.0% |
| 4th-10th | 70 | 0.0% | 0.0% |
| 11th onward | 200 | 0.0% | 1.5% |

Parsers promoted: 0. Disagreements recognised as OCR noise: 0. Drifts confirmed: 0.

_Fields right_ counts a value the annotators did not label as wrong; _labelled fields right_ scores only the fields DocILE labelled on that document (DocILE does not label every printed value); _answerable_ further drops labelled values that do not appear in the OCR text the model is given (20% of labelled fields here: DocILE's OCR cuts or splits them). Amounts, dates and currencies are compared by value ($ = USD). Per-document predictions are in the matching `_predictions.jsonl` file._

Fields Distill got wrong most often: customer_billing_name (235), vendor_address (192), amount_total_net (162), amount_total_gross (129), order_id (124), customer_id (102)