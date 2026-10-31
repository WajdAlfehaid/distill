"""Build the static dashboard for GitHub Pages: recorded runs of the real Distill engine, played back in the browser.

    python scripts/build_static.py            # writes docs/index.html and docs/data/*.json

GitHub Pages serves static files only, so the page cannot run the Python engine. Instead this script warms the
engine up (9,000 invoices per dataset, like the live demo), then records what it does next, invoice by invoice:
the strip mark, the running totals, the cost curve, the event log and every invoice's details. On the generated
invoices it also records the silent template change made at regular points along the run, so the change button
works whenever it is pressed. Every number on the page comes from these recorded runs.
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":  # identical runs for the main track and every fork
    os.environ["PYTHONHASHSEED"] = "0"
    sys.exit(subprocess.call([sys.executable] + sys.argv))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from distill.server import Simulation  # noqa: E402

WARM = 9000
FATURA_STEPS = 1000          # 9,000 + 1,000 = the whole FATURA set, no invoice shown twice
SYN_STEPS = 960              # two minutes at 8 invoices per second
FORK_EVERY = 80              # a recorded template change every 10 seconds of playback
BRANCH_STEPS = 240           # 30 seconds after each change
SCALARS = ("docs", "cost", "small", "large", "acc", "base_acc", "lat", "base_lat")


def r6(x):
    return None if x is None else round(float(x), 6)


def scalars(s: dict) -> list:
    return [s["totals"]["docs"], r6(s["totals"]["cost_usd"]), r6(s["counterfactual"]["small"]), r6(s["counterfactual"]["large"]),
            r6(s["accuracy"]), r6(s["baseline_accuracy"]), r6(s["latency_ms_p50"]), r6(s["baseline_latency_ms_p50"])]


def doc_detail(sim: Simulation, doc_id: str) -> dict:
    d = sim.doc(doc_id) or {}
    final, truth = d.get("final") or {}, d.get("truth") or {}
    slim = lambda r: {k: (["" for _ in v] if k == "line_items" else v) for k, v in r.items()}
    return {"seq": d.get("seq"), "vendor_id": d.get("vendor_id"), "parser_id": d.get("parser_id"),
            "route": {"path": (d.get("route") or {}).get("path"), "reasons": (d.get("route") or {}).get("reasons", [])},
            "cost_usd": r6(d.get("cost_usd")), "correct": d.get("correct"), "wrong_fields": d.get("wrong_fields") or [],
            "checks": [{"name": c["name"], "ok": c["ok"], "severity": c.get("severity", "fail")} for c in d.get("checks", [])],
            "final": slim(final), "truth": slim(truth), "trace": d.get("trace", [])}


EVENT_KEYS = ("seq", "kind", "vendor_id", "message", "n", "corrected")


def event(e: dict) -> dict:
    return {k: e[k] for k in EVENT_KEYS if k in e}


class Recorder:
    """Steps a simulation and records one compact frame per invoice."""

    def __init__(self, sim: Simulation):
        self.sim = sim
        self.n_events = len(sim.engine.events)
        self.n_curve = len(sim.curve)
        self.curve_last = sim.curve[-1]["n"] if sim.curve else 0

    def step(self) -> dict:
        sim = self.sim
        rec = sim.step()
        s = sim.snapshot()
        frame = {"t": s["ticks"][-1], "s": scalars(s), "d": doc_detail(sim, rec["doc_id"])}
        new_curve = [p for p in sim.curve if p["n"] > self.curve_last]
        if new_curve:
            frame["c"] = new_curve
            self.curve_last = new_curve[-1]["n"]
        evs = sim.engine.events
        if len(evs) > self.n_events:
            frame["e"] = [event(e) for e in evs[self.n_events:]]
            self.n_events = len(evs)
        return frame


def initial(sim: Simulation) -> dict:
    s = sim.snapshot()
    docs = {t["doc_id"]: doc_detail(sim, t["doc_id"]) for t in s["ticks"]}
    return {"dataset": s["dataset"], "models": s["models"], "scalars": scalars(s), "ticks": s["ticks"],
            "curve": [dict(p) for p in sim.curve], "events": [event(e) for e in sim.engine.events[-60:]],
            "vendors": [{"vendor_id": v["vendor_id"], "vendor": v["vendor"], "docs": v.get("docs", 0)} for v in s["vendors"]],
            "docs": docs}


def warmed(dataset: str) -> Simulation:
    sim = Simulation(dataset=dataset, db=":memory:", fatura_path=str(ROOT / "data" / "fatura_invoices.jsonl.gz"))
    for i in range(WARM):
        sim.step()
    return sim


def drift_vendor(sim: Simulation) -> str:
    """The supplier the dashboard preselects for the silent swap: the busiest one that allows it."""
    for v in sorted(sim.snapshot()["vendors"], key=lambda v: -v.get("docs", 0)):
        if "inline_swap" in sim.stream.eligible_drifts(v["vendor_id"]):
            return v["vendor_id"]
    raise RuntimeError("no supplier allows the silent swap")


def build_fatura() -> dict:
    print("FATURA: warming up", WARM, "invoices ...", flush=True)
    sim = warmed("fatura")
    out = initial(sim)
    rec = Recorder(sim)
    out["frames"] = [rec.step() for _ in range(FATURA_STEPS)]
    return out


def build_synthetic() -> dict:
    print("Generated invoices: warming up", WARM, "invoices ...", flush=True)
    sim = warmed("synthetic")
    vid = drift_vendor(sim)
    out = initial(sim)
    out["drift"] = {"vendor_id": vid, "kind": "inline_swap", "vendor": sim.stream.vendor(vid).name}
    rec = Recorder(sim)
    out["frames"] = [rec.step() for _ in range(SYN_STEPS)]
    forks = []
    for at in range(0, SYN_STEPS, FORK_EVERY):
        print(f"  template change after {at} invoices of playback ...", flush=True)
        f = warmed("synthetic")
        r = Recorder(f)
        for i in range(at):  # replay the main track up to the fork, and check it is the same run
            fr = r.step()
            main = out["frames"][i]  # timings are measured by the clock, so compare everything else
            if fr["s"][:6] != main["s"][:6] or fr["t"]["doc_id"] != main["t"]["doc_id"]:
                raise RuntimeError(f"fork at {at} diverged from the main track at step {i}")
        desc = f.apply_drift(vid, "inline_swap")
        f.pending.extend([vid] * 15)  # the dashboard sends 15 of that supplier's invoices with the change
        evs = f.engine.events
        changed = [event(e) for e in evs[r.n_events:]]
        r.n_events = len(evs)
        frames = [r.step() for _ in range(BRANCH_STEPS)]
        forks.append({"at": at, "desc": desc, "events": changed, "frames": frames})
    out["forks"] = forks
    return out


def main() -> None:
    data_dir = ROOT / "docs" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in (("fatura", build_fatura), ("synthetic", build_synthetic)):
        blob = json.dumps(fn(), separators=(",", ":"), default=str)
        (data_dir / f"{name}.json").write_text(blob)
        print(f"  docs/data/{name}.json: {len(blob) / 1e6:.1f} MB ({len(gzip.compress(blob.encode())) / 1e6:.1f} MB compressed)", flush=True)
    html = (ROOT / "web" / "index.html").read_text()
    shim = (ROOT / "web" / "static_shim.js").read_text()
    html = html.replace('$("#reset").hidden = !!S.public;', '$("#reset").hidden = true;')
    html = html.replace("const vs = S.vendors.slice().sort((a, b) => b.docs - a.docs);",
                        "const vs = S.vendors.slice().sort((a, b) => b.docs - a.docs).filter(v => !S.drift || v.vendor_id === S.drift.vendor_id);")
    html = html.replace('const es = new EventSource("/api/stream");', "const es = new ReplayStream();")
    html = html.replace("<script>", "<script>\n" + shim + "\n", 1)
    (ROOT / "docs" / "index.html").write_text(html)
    (ROOT / "docs" / ".nojekyll").write_text("")
    print("wrote docs/index.html", flush=True)


if __name__ == "__main__":
    main()
