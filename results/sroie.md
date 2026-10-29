# SROIE: 619 real documents, 306 sources
Teacher: human labels (perfect teacher).

| System | Docs fully right | Fields right | Handled by code | Right when handled by code | Wrong but accepted by code | LLM calls / doc | LLM cost | Cost if every doc went to Haiku / Sonnet |
|---|---|---|---|---|---|---|---|---|
| distill | 100.0% | 100.0% | 0.0% | - | 0 | 1.03 | $0.0700 | $0.0462 / $0.92 |
| static | 98.5% | 99.6% | 2.7% | 47.1% | 9 | 1.00 | $0.0688 | $0.0462 / $0.92 |

## Distill by how many documents it had already seen from that source

| Position | Docs | Handled by code | Fully right |
|---|---|---|---|
| first 3 from a source | 394 | 0.0% | 100.0% |
| 4th-10th | 81 | 0.0% | 100.0% |
| 11th onward | 144 | 0.0% | 100.0% |

Parsers promoted: 0. Disagreements recognised as OCR noise: 0. Drifts confirmed: 0.

Fields Distill got wrong most often: 

_With a perfect teacher, reading every document with the model is 100% right by construction; the question is how much of that Distill keeps while replacing the model with compiled code. Run with `--real` to measure accuracy against real Claude._