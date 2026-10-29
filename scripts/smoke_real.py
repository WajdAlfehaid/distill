"""Quick check that real Claude extraction works before a live demo.

    ANTHROPIC_API_KEY=sk-... python scripts/smoke_real.py [n_docs]

Extracts a few synthetic invoices with the small and large models and prints
field differences against ground truth, token usage and cost.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distill.datagen import DEMO_OVERRIDES, InvoiceStream  # noqa: E402
from distill.llm import AnthropicLLM  # noqa: E402
from distill.model import diff  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    llm = AnthropicLLM()
    s = InvoiceStream(seed=11, vendor_overrides=DEMO_OVERRIDES)
    total = 0.0
    for _ in range(n):
        d = s.next_doc()
        for tier in ("small", "large"):
            r = llm.extract(d.doc_id, d.text, tier)
            total += r.cost_usd
            if r.invoice is None:
                print(f"{d.doc_id} {tier:5s} {r.model}: ERROR {r.error}")
                continue
            bad = diff(r.invoice, d.truth)
            print(f"{d.doc_id} {tier:5s} {r.model}: {'exact' if not bad else 'wrong ' + str(bad)} "
                  f"({r.input_tokens}+{r.output_tokens} tokens, ${r.cost_usd:.5f}, {r.latency_ms:.0f} ms)")
    print(f"total cost ${total:.4f}")


if __name__ == "__main__":
    main()
