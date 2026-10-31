# FATURA: 10,000 third-party template invoices (FATURA, Limam et al. 2023), text layer as provided, 50 sources
Teacher: human labels (perfect teacher).

| System | Docs fully right (answerable fields) | Fields right | Labelled fields right | Answerable labelled fields right | Handled by code | Right when handled by code | Wrong but accepted by code | LLM calls / doc | LLM cost | Cost if every doc went to Haiku / Sonnet |
|---|---|---|---|---|---|---|---|---|---|---|
| llm_only | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | - | 0 | 1.00 | $1.0697 | $1.0697 / $21.39 |
| static | 100.0% | 99.9% | 99.8% | 100.0% | 54.8% | 97.9% | 117 | 0.45 | $0.4859 | $1.0697 / $21.39 |
| distill | 100.0% | 99.9% | 99.8% | 100.0% | 82.0% | 98.6% | 113 | 0.29 | $0.3616 | $1.0697 / $21.39 |

## Distill by how many documents it had already seen from that source

| Position | Docs | Handled by code | Fully right |
|---|---|---|---|
| first 3 from a source | 150 | 0.0% | 100.0% |
| 4th-10th | 350 | 1.1% | 100.0% |
| 11th onward | 9,500 | 86.3% | 98.8% |

Parsers promoted: 138. Disagreements recognised as OCR noise: 39. Drifts confirmed: 14.

_FATURA's text layer has its own OCR-like damage (e.g. 'INVO1036008' for 'INV01036008'); the model is given the same text. _Fields right_ counts a value the annotators did not label as wrong; _labelled fields right_ scores only the fields DocILE labelled on that document (DocILE does not label every printed value); _answerable_ further drops labelled values that do not appear in the text the model is given (1% of labelled fields here). Amounts, dates and currencies are compared by value ($ = USD). Per-document predictions are in the matching `_predictions.jsonl` file._

Fields Distill got wrong most often: discount (69), invoice_number (29), total (4), due_date (4), subtotal (3), invoice_date (3)

_With a perfect teacher, reading every document with the model is 100% right by construction; the question is how much of that Distill keeps while replacing the model with compiled code. Run with `--real` to measure accuracy against real Claude._