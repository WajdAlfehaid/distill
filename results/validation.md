# Validation: held-out streams and ablations

`python scripts/validate.py`. Model tiers are simulated as in the main benchmark (tokens billed at Claude Haiku 5.5 / Sonnet 5.5 prices); parsers, checks, drift detection, routing and audits are real.

## 1. Held-out streams (never seen during development)

Seeds 101, 202 and 303 each generate 8,000 invoices from new suppliers, layouts, scan damage and template changes. Development used seed 11 only.

| Stream | System | Exactly right | AI cost | Cost per 1k correct | Wrong after template changes | Changes caught (median invoices) |
|---|---|---|---|---|---|---|
| seed 101 | Sonnet on every invoice | 94.0% | $30.22 | $4.017 | 158 / 3598 |  |
| seed 101 | Haiku (Sonnet when checks fail) | 95.3% | $4.96 | $0.650 | 129 / 3598 |  |
| seed 101 | Compile once | 80.1% | $2.85 | $0.444 | 700 / 3598 |  |
| seed 101 | Distill | 97.2% | $1.29 | $0.166 | 80 / 3598 | 4 |
| seed 202 | Sonnet on every invoice | 95.2% | $29.31 | $3.847 | 118 / 3439 |  |
| seed 202 | Haiku (Sonnet when checks fail) | 95.7% | $4.06 | $0.530 | 128 / 3439 |  |
| seed 202 | Compile once | 81.4% | $2.40 | $0.369 | 702 / 3439 |  |
| seed 202 | Distill | 98.0% | $0.87 | $0.112 | 46 / 3439 | 4 |
| seed 303 | Sonnet on every invoice | 89.5% | $29.72 | $4.149 | 307 / 3623 |  |
| seed 303 | Haiku (Sonnet when checks fail) | 91.7% | $7.21 | $0.983 | 245 / 3623 |  |
| seed 303 | Compile once | 72.7% | $5.02 | $0.863 | 1369 / 3623 |  |
| seed 303 | Distill | 94.9% | $2.36 | $0.311 | 160 / 3623 | 3 |

Across the three held-out streams Distill costs 3.2–4.7x less than Haiku and 13–34x less than Sonnet per correct invoice, and is more accurate than Haiku on every stream (97.2% vs 95.3%, 98.0% vs 95.7%, 94.9% vs 91.7%).

## 2. Ablations (main stream, seed 11)

| System | Exactly right | Wrong invoices | AI cost | Cost per 1k correct | Model calls / invoice |
|---|---|---|---|---|---|
| Distill | 96.84% | 253 | $1.42 | $0.184 | 0.49 |
| Distill without invariant repair | 96.29% | 297 | $3.26 | $0.423 | 0.67 |
| Distill without scoped escalation | 96.84% | 253 | $1.60 | $0.207 | 0.48 |

- Switching off invariant repair: cost 2.29x, +44 wrong invoices.

- Switching off scoped escalation: cost 1.13x, +0 wrong invoices.

The main benchmark (`results/benchmark.md`) also includes the reverse ablation: Haiku with Distill's invariant repair but no compiling, which separates what repair adds from what compiling adds.
