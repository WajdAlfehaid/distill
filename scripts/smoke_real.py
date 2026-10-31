"""Quick check that real Claude extraction works before a live demo.

    ANTHROPIC_API_KEY=sk-... python scripts/smoke_real.py [n_docs]

    python scripts/smoke_real.py 10 --compare   # thinking off / low / on, side by side

Extracts a few synthetic invoices with the small and large models and prints
field differences against ground truth, token usage and cost.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distill.datagen import DEMO_OVERRIDES, InvoiceStream  # noqa: E402
from distill.llm import AnthropicLLM  # noqa: E402
from distill.model import diff  # noqa: E402


def compare(n: int):
    """Same n invoices with each thinking mode; prints accuracy, output tokens, cost and time per mode."""
    docs = []
    s = InvoiceStream(seed=11, vendor_overrides=DEMO_OVERRIDES)
    for _ in range(n):
        docs.append(s.next_doc())
    print(f"{'mode':5s} {'tier':5s} {'right':>7s} {'out tok/doc':>11s} {'$/1k docs':>10s} {'s/doc':>6s}")
    for mode in ("off", "low", "on"):
        llm = AnthropicLLM(thinking=mode)
        for tier in ("small", "large"):
            ok = tout = 0
            cost = ms = 0.0
            for d in docs:
                r = llm.extract(d.doc_id, d.text, tier)
                ok += r.invoice is not None and not diff(r.invoice, d.truth)
                tout += r.output_tokens
                cost += r.cost_usd
                ms += r.latency_ms
            print(f"{mode:5s} {tier:5s} {ok:>3d}/{n:<3d} {tout / n:>11.0f} {cost / n * 1000:>10.2f} {ms / n / 1000:>6.1f}", flush=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show = "--show" in sys.argv
    n = int(args[0]) if args else 3
    if "--compare" in sys.argv:
        return compare(n)
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
            if show:
                print(f"--- raw {tier} output ---\n{r.raw}\n")
            print(f"{d.doc_id} {tier:5s} {r.model}: {'exact' if not bad else 'wrong ' + str(bad)} "
                  f"({r.input_tokens}+{r.output_tokens} tokens, ${r.cost_usd:.5f}, {r.latency_ms:.0f} ms)")
    print(f"total cost ${total:.4f}")


if __name__ == "__main__":
    main()
