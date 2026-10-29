# DOCILE: 600 real documents, 280 sources
Teacher: human labels (perfect teacher).

| System | Docs fully right | Fields right | Handled by code | Right when handled by code | Wrong but accepted by code | LLM calls / doc | LLM cost | Cost if every doc went to Haiku / Sonnet |
|---|---|---|---|---|---|---|---|---|
| llm_only | 99.8% | 100.0% | 0.0% | - | 0 | 1.02 | $0.1358 | $0.0958 / $1.92 |
| static | 99.2% | 99.9% | 0.8% | 20.0% | 4 | 1.01 | $0.1350 | $0.0958 / $1.92 |
| distill | 99.8% | 100.0% | 0.0% | - | 0 | 1.02 | $0.1412 | $0.0958 / $1.92 |

## Distill by how many documents it had already seen from that source

| Position | Docs | Handled by code | Fully right |
|---|---|---|---|
| first 3 from a source | 457 | 0.0% | 99.8% |
| 4th-10th | 141 | 0.0% | 100.0% |
| 11th onward | 2 | 0.0% | 100.0% |

Parsers promoted: 0. Disagreements recognised as OCR noise: 0. Drifts confirmed: 0.

Fields Distill got wrong most often: order_id (1)

_With a perfect teacher, reading every document with the model is 100% right by construction; the question is how much of that Distill keeps while replacing the model with compiled code. Run with `--real` to measure accuracy against real Claude._