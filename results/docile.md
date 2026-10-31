# DOCILE: 50 real documents, 10 sources
Teacher: human labels (perfect teacher).

| System | Docs fully right (answerable fields) | Fields right | Labelled fields right | Answerable labelled fields right | Handled by code | Right when handled by code | Wrong but accepted by code | LLM calls / doc | LLM cost | Cost if every doc went to Haiku / Sonnet |
|---|---|---|---|---|---|---|---|---|---|---|
| llm_only | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | - | 0 | 1.00 | $0.0138 | $0.0138 / $0.28 |

_With a perfect teacher, reading every document with the model is 100% right by construction; the question is how much of that Distill keeps while replacing the model with compiled code. Run with `--real` to measure accuracy against real Claude._