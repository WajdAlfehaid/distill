# Demo script (5 minutes)

Before going on stage: start `python -m distill.server` (add `--real` if the venue network and API key are reliable), press **Start stream** at speed 40, and let it reach about 1,500 invoices so every vendor has a stable parser. Then pause. Keep the recorded backup video one click away.

## 0:00 The problem (40 s)

"Accounts-payable teams receive thousands of invoices from the same few hundred suppliers. Today the modern answer is to send every one of them to a large language model. That works, but you pay full price to read the same layout for the thousandth time, and you wait a second or more per page.

The obvious fix is to have the AI write a parser once and reuse it. Stanford showed in 2023 that this can cut AI usage by about 110x. The reason nobody runs that in production is simple: suppliers change their templates, and a parser that is silently wrong is worse than paying for the AI."

## 0:40 What you're looking at (40 s)

Point at the invoice strip. "Each bar is an invoice. Green means compiled code handled it with no AI at all. Blue means a parser result that was double-checked by the model. Amber means the model read it. Bar height is what that invoice cost."

Point at the chart (cost per 1,000 view). "Grey is what it costs to send every invoice to Sonnet, amber to Haiku. Green is Distill." Point at the numbers panel: share handled by code, accuracy, cost per thousand.

Click one green bar. "Every result is traceable: which parser version produced it, which 21 checks it passed, and why it wasn't sent to the AI."

## 1:20 Hand over the controls (90 s)

Press **Start stream**. Hand the judge the **Change a vendor's template** panel. Suggest they pick **Swap price and quantity (silent)** on Lone Star Fasteners, and explain while they do:

"This one is nasty. `3 @ 12.50 = 37.50` becomes `12.50 @ 3 = 37.50`. The old parser still runs, every total still adds up. A compile-once system gets every one of these invoices wrong, forever."

Narrate the feed as it updates:
1. History check: this vendor never bills fractional quantities or sells at these prices, so the policy orders a targeted audit.
2. The audit disagrees and the larger model confirms: drift. The parser is suspended.
3. Quarantine: invoices the parser accepted since its last verified result are re-checked.
4. A new parser is synthesized from fresh examples, validated in isolation, then runs in shadow next to live traffic.
5. After three shadow agreements it's promoted. Click the vendor to show the lineage and the rollback button.

Let a second judge try a loud change (rename labels) on another vendor if time allows.

## 2:50 The numbers (60 s)

Show results/benchmark.md and the DocILE / SROIE tables (or slides built from them):
- Synthetic stream with realistic scan damage, 8,000 invoices, six template changes:
  Sonnet on everything $24.69 at 94.1% right; Haiku on everything $4.18 at 94.8%; Distill $3.75 at 96.2%,
  with 42% fewer model calls and 74% of steady-state traffic handled by code in under a millisecond.
- Compile-once: 393 of 430 invoices wrong after the silent price/quantity swap. Distill: caught within 2 invoices.
- Real data: DocILE (5,680 business documents) and SROIE (619 scanned receipts). On SROIE Distill compiled nothing
  and made no silent errors; compile-once compiled anyway and was wrong on 9 of 17.

Say plainly which numbers are simulated (model tiers), which use a perfect teacher (public datasets), and which come
from real Claude if you ran `--real`. Be upfront that against the cheapest model the dollar saving is modest; the
case is accuracy, latency, a 6.6x saving against larger models, and evidence on every result.

## 3:50 Why it matters (40 s)

"The cost curve is the obvious win. The real product is the evidence: every parser carries a statistical bound on its error rate from continuous audits, and when that evidence weakens, Distill spends more AI exactly where it's needed. That's how you make AI cheaper without making it less trustworthy. And nothing here is invoice-specific: the same loop works for purchase orders, shipping notices, bank statements and log lines."

## 4:30 Questions

Likely questions and short answers:
- *Why not just use Haiku on everything?* You can, and it is cheap. Distill is more accurate than that (checks,
  arbitration, parsers correcting model slips), answers in milliseconds for most traffic, and tells you when a
  result needs a person. The dollar saving against the cheapest model is modest; against larger models it is large.
- *What about badly scanned documents?* Distill refuses to compile what does not hold up (SROIE), and tells scan
  noise apart from template changes so one smudged copy does not throw away a good parser.
- *What if the LLM labels are wrong?* Labels must pass the business checks; synthesis tolerates a mislabelled example (leave-one-out); shadow disagreements are arbitrated by a larger model, and the parser can correct the LLM.
- *Is generated code safe to run?* Nothing generated is executed. Parsers are JSON specs of line patterns run by a fixed interpreter.
- *How is this different from Evaporate?* Evaporate already validates and aggregates candidate functions on a fixed batch. Distill adds what a changing stream needs: drift detection, versioning, shadow promotion, quarantine, adaptive audits and error bounds.
- *Real PDFs?* Text extraction or OCR goes in front of the engine; the engine works on text.
