# Thinking modes for extraction (real Claude API)

`python scripts/smoke_real.py 10 --compare`: the same 10 synthetic invoices extracted with each setting.
Measured 30 October 2026 with claude-haiku-5-5 (small) and claude-sonnet-5-5 (large).

| Thinking | Model | Exactly right | Output tokens / doc | $ per 1k docs | Seconds / doc |
|---|---|---|---|---|---|
| off (disabled, low effort) | Haiku | 7/10 | 287 | 0.21 | 2.1 |
| off (disabled, low effort) | Sonnet | 9/10 | 264 | 4.00 | 3.3 |
| **low effort (default)** | **Haiku** | **9/10** | **599** | **0.37** | **3.4** |
| low effort | Sonnet | 9/10 | 420 | 5.56 | 3.5 |
| on (API defaults) | Haiku | 9/10 | 713 | 0.42 | 4.0 |
| on (API defaults) | Sonnet | 9/10 | 450 | 5.86 | 4.1 |

Switching thinking off halves Haiku's cost per call but costs accuracy (both misses were misread dates), and a
miss sends the document to Sonnet at ~$4-6 per 1k, so it is more expensive overall. Low effort keeps the accuracy
of the default for ~12% less. Distill uses low effort (`DISTILL_THINKING=off|low|on`).
