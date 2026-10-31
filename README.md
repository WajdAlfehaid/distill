# Distill: browser version

This branch holds only the browser version of the Distill dashboard, published with GitHub Pages at
https://wajdalfehaid.github.io/distill/. The project itself (the engine, the live dashboard, tests and benchmarks)
is on the `master` branch.

GitHub Pages serves static files, so it cannot run the Python engine. This page plays back recorded runs of the
real engine: it was warmed up with 9,000 invoices per dataset, then recorded invoice by invoice (the strip, the totals,
the cost curve, the event log and every invoice's details). On the generated invoices the silent template change was
recorded at regular points along the run, so the change button works whenever it is pressed. AI calls are simulated
at Claude's prices and timings.

- `index.html`, `data/`: the site.
- `tools/build_static.py`, `tools/static_shim.js`: how it was built. To rebuild, copy them into a `master` checkout
  as `scripts/build_static.py` and `web/static_shim.js`, run `python scripts/build_static.py`, and copy `docs/` here.
