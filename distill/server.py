"""Dashboard server: streams invoices through Distill and serves the dashboard.

    python -m distill.server                 # simulated LLM tiers, http://localhost:8000
    ANTHROPIC_API_KEY=... python -m distill.server --real

The judge can break any vendor's template from the dashboard and watch Distill
detect, quarantine, re-synthesize, shadow-test and promote a new parser.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import threading
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from .datagen import DEMO_OVERRIDES, DRIFT_KINDS, InvoiceStream
from .doctypes import FaturaType, InvoiceType
from .engine import Config, Engine
from .llm import LARGE_MODEL, SMALL_MODEL, AnthropicLLM, SimulatedLLM, cost_of, SYSTEM_PROMPT
from .model import Invoice, Record, diff
from .registry import Registry

WEB = Path(__file__).resolve().parent.parent / "web"
DATASETS = {"synthetic": "Generated invoices, with template changes", "fatura": "FATURA: third-party invoices, 50 layouts"}


class _Doc:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Layout:
    def __init__(self, vid: str):
        self.vendor_id, self.name, self.scan, self.template_rev = vid, f"Layout {vid[1:]}", "text layer", 0


class FaturaSource:
    """Streams FATURA invoices (Limam et al. 2023) in a shuffled order, like a mailbox receiving them.
    The source of an invoice is its layout, the way a supplier would be in accounts payable."""

    def __init__(self, path: str, seed: int):
        import random
        from .realdata import load_fatura
        self.items = load_fatura(path)
        random.Random(seed).shuffle(self.items)
        self.pos = 0
        self.by_vid: dict[str, list] = {}
        for it in self.items:
            self.by_vid.setdefault(f"K{it['meta']['cluster_id']}", []).append(it)

    def next_doc(self, vid: Optional[str] = None) -> _Doc:
        if vid and self.by_vid.get(vid):
            it = self.by_vid[vid][self.pos % len(self.by_vid[vid])]
        else:
            it = self.items[self.pos % len(self.items)]
        self.pos += 1
        return _Doc(doc_id=f"{it['doc_id']}-{self.pos}", text=it["text"], truth=it["truth"], damage={},
                    vendor_id=f"K{it['meta']['cluster_id']}", received_at="2019-01-01T00:00:00", meta=it["meta"])

    def vendor(self, vid: str) -> _Layout:
        return _Layout(vid)

    def eligible_drifts(self, vid: str) -> list:
        return []  # FATURA has no template changes; use the generated stream for that part of the demo

    def apply_drift(self, vid: str, kind: str) -> str:
        raise ValueError("template changes are demonstrated on the generated invoice stream")


class Simulation:
    def __init__(self, real: bool = False, seed: int = 11, db: str = ":memory:", profile: str = "balanced",
                 dataset: str = "synthetic", fatura_path: str = "data/fatura_invoices.jsonl.gz"):
        self.real, self.seed, self.db, self.profile = real, seed, db, profile
        self.dataset, self.fatura_path = dataset, fatura_path
        self.warming = 0
        self.lock = threading.RLock()
        self.running = False
        self.speed = 8.0  # documents per second
        self.started_at = 0.0
        self.pending: deque = deque()  # vendor ids forced next (judge "send invoices")
        self.reset()
        threading.Thread(target=self._loop, daemon=True).start()

    def reset(self) -> None:
        with getattr(self, "lock", threading.RLock()):
            if hasattr(self, "engine"):
                self.engine.reg.db.close()
            if self.db != ":memory:" and Path(self.db).exists():
                try:
                    Path(self.db).unlink()
                except OSError:
                    pass
            self.truth: dict = {}
            self.damage: dict[str, dict] = {}
            self.vendor_of: dict[str, str] = {}
            self.texts: dict[str, str] = {}
            self.meta: dict[str, dict] = {}
            if self.dataset == "fatura":
                from .realdata import PERFECT
                self.stream = FaturaSource(self.fatura_path, self.seed)
                mk_type = lambda: FaturaType(use_given_clusters=True)
                # FATURA's labels act as a model that never errs: the cheapest possible model-only baseline
                mk_llm = lambda: SimulatedLLM(lambda d: self.truth[d], seed=self.seed, profiles=PERFECT)
            else:
                self.stream = InvoiceStream(seed=self.seed, vendor_overrides=DEMO_OVERRIDES)
                mk_type = lambda: InvoiceType(self.stream.vendor_master(), self.stream.po_master)
                mk_llm = lambda: SimulatedLLM(lambda d: self.truth[d], seed=self.seed, damage=lambda d: self.damage.get(d))
            doctype = mk_type()
            self.schema = doctype.schema
            llm = AnthropicLLM() if self.real else mk_llm()
            self.engine = Engine(doctype, llm, Config(mode="distill", profile=self.profile), Registry(self.db))
            self.cf = {"large": 0.0, "small": 0.0}  # counterfactual: every document through one model
            # Baseline run side by side on the same invoices (simulated mode only, so it costs nothing):
            # the small model on every invoice, the large model when its answer fails the checks.
            self.base = None if self.real else Engine(mk_type(), mk_llm(), Config(mode="llm_only", llm_only_tier="small"))
            self.base_wrong = 0
            self.base_recent: deque = deque(maxlen=200)  # baseline cost of the last 200 invoices
            self.lat: deque = deque(maxlen=1000)  # time to an answer per invoice (model time simulated from real calls)
            self.base_lat: deque = deque(maxlen=1000)
            self.curve: list[dict] = []
            self.ticks: deque = deque(maxlen=400)
            self.drifts: list[dict] = []
            self.wrong = 0
            self.verdicts: dict[str, dict] = {}
            self.revised_seen = 0
            self.pending.clear()

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        while True:
            # Public deployment: never run unattended (nobody watching, or one Start left on for more than 5 minutes).
            if PUBLIC and self.running and (VIEWERS == 0 or time.time() - self.started_at > PUBLIC_RUN_SECONDS):
                self.running = False
            if self.running or self.pending:
                try:
                    self.step()
                except Exception as e:  # keep the demo alive; surface the error in the feed
                    self.engine._event("error", None, f"processing error: {type(e).__name__}: {e}")
                time.sleep(1.0 / max(self.speed, 0.2))
            else:
                time.sleep(0.05)

    def step(self) -> dict:
        with self.lock:
            vid = self.pending.popleft() if self.pending else None
            d = self.stream.next_doc(vid)
            self.truth[d.doc_id] = d.truth
            self.damage[d.doc_id] = d.damage
            self.vendor_of[d.doc_id] = d.vendor_id
            self.texts[d.doc_id] = d.text
            meta = getattr(d, "meta", None)
            rec = self.engine.process(d.doc_id, d.text, d.received_at, meta)
            self.lat.append(rec["latency_ms"])
            tin, tout = SimulatedLLM.tokens_for(d.text, d.truth)
            self.cf["large"] += cost_of(LARGE_MODEL, tin, tout)
            if self.base is not None:
                b = self.base.process(d.doc_id, d.text, d.received_at, meta)
                self.cf["small"] += b["cost_usd"]
                self.base_recent.append(b["cost_usd"])
                self.base_lat.append(b["latency_ms"])
                self.base_wrong += bool(self._wrong(self._rec(b["final"]), d.doc_id))
            else:
                self.cf["small"] += cost_of(SMALL_MODEL, tin, tout)
            self._verify(rec)
            # quarantine may have revised earlier results: re-verify those too
            for qid in self.engine.revised[self.revised_seen:]:
                self._verify(self.engine.records[qid])
            self.revised_seen = len(self.engine.revised)
            bad = rec["_wrong_fields"]
            kind = "drift" if "drift_confirmed" in rec["flags"] else (
                "code" if rec["route"] and rec["route"]["path"] == "code" else
                "audit" if rec["route"] and rec["route"]["path"] == "audit" else "llm")
            self.ticks.append({"doc_id": d.doc_id, "kind": kind, "vendor": d.vendor_id, "ok": not bad,
                               "cost": round(rec["cost_usd"], 7)})
            n = self.engine.totals["docs"]
            if n % 5 == 0 or n < 5:
                self.curve.append({"n": n, "distill": round(self.engine.totals["cost_usd"], 6),
                                   "large": round(self.cf["large"], 6), "small": round(self.cf["small"], 6)})
                if len(self.curve) > 600:
                    self.curve = self.curve[::2]
            return rec

    def _rec(self, final: Optional[dict]):
        if self.schema.name == "invoice":
            return Invoice.from_dict(final) if final else Invoice()
        return Record.from_dict(final, self.schema) if final else Record(self.schema)

    def _wrong(self, pred, doc_id: str) -> list[str]:
        truth = self.truth[doc_id]
        if self.dataset != "fatura":
            return diff(pred, truth)
        from .realdata import score  # FATURA: compare on values its text layer prints legibly, by value
        item = {"truth": truth, "text": self.texts[doc_id], "alts": {k: [v] for k, v in truth.values.items() if v}}
        return score(pred, item, answerable_only=True)

    def _verify(self, rec: dict) -> None:
        """Compare a result with the answer key (available because the demo data is generated)."""
        truth = self.truth[rec["doc_id"]]
        pred = self._rec(rec["final"])
        bad = self._wrong(pred, rec["doc_id"])
        prev = self.verdicts.get(rec["doc_id"])
        if prev is not None and prev["wrong"]:
            self.wrong -= 1
        rec["_correct"] = not bad
        rec["_wrong_fields"] = bad
        self.wrong += 1 if bad else 0
        path = (rec.get("route") or {}).get("path") or "llm"
        self.verdicts[rec["doc_id"]] = {"path": path, "wrong": bool(bad), "flagged": "human_review" in rec["flags"]}

    def verification(self) -> dict:
        by = {p: {"docs": 0, "right": 0} for p in ("code", "audit", "llm")}
        cm = {"passed_right": 0, "passed_wrong": 0, "flagged_wrong": 0, "flagged_right": 0}
        for v in self.verdicts.values():
            b = by.setdefault(v["path"], {"docs": 0, "right": 0})
            b["docs"] += 1
            b["right"] += 0 if v["wrong"] else 1
            key = ("flagged_" if v["flagged"] else "passed_") + ("wrong" if v["wrong"] else "right")
            cm[key] += 1
        return {"by_route": by, "confusion": cm}

    def recent_rows(self, n: int = 30) -> list[dict]:
        out = []
        for did in reversed(self.engine.order[-n:]):
            r = self.engine.records[did]
            v = self.stream.vendor(self.vendor_of[did])
            out.append({"doc_id": did, "seq": r["seq"], "vendor": v.name, "scan": v.scan,
                        "path": (r.get("route") or {}).get("path"), "source": r["source"], "parser": r.get("parser_id"),
                        "checks_passed": r.get("checks_passed"), "flags": r["flags"], "correct": r.get("_correct"),
                        "wrong_fields": r.get("_wrong_fields") or [], "cost": r["cost_usd"],
                        "damaged": sorted(self.damage.get(did, {}).keys())})
        return out

    # ------------------------------------------------------------------ views
    def snapshot(self) -> dict:
        with self.lock:
            e = self.engine
            T = dict(e.totals)
            n = max(T["docs"], 1)
            recent = [e.records[i] for i in e.order[-200:]]
            last_code = sum(1 for r in recent if r["source"] == "code") / len(recent) if recent else 0
            return {
                "mode": "real Claude" if self.real else ("FATURA invoices, labels as a perfect model" if self.dataset == "fatura"
                                                         else "simulated LLM tiers"),
                "dataset": self.dataset, "datasets": DATASETS, "warming": self.warming, "public": PUBLIC,
                "available": [d for d in DATASETS if d in SIMS or d == "synthetic" or Path(self.fatura_path).exists()],
                "running": self.running, "speed": self.speed, "profile": self.profile,
                "models": {"small": SMALL_MODEL, "large": LARGE_MODEL},
                "totals": T, "wrong": self.wrong, "accuracy": 1 - self.wrong / n if T["docs"] else None,
                "cost_per_1k": 1000 * T["cost_usd"] / n,
                "cost_per_1k_recent": 1000 * sum(r["cost_usd"] for r in recent) / max(len(recent), 1),
                "code_share_recent": last_code,
                "counterfactual": self.cf,
                "baseline_per_1k_recent": (1000 * sum(self.base_recent) / len(self.base_recent)) if self.base and self.base_recent else None,
                "baseline_accuracy": (1 - self.base_wrong / max(1, self.engine.totals["docs"])) if self.base else None,
                "latency_ms_p50": statistics.median(self.lat) if self.lat else None,
                "baseline_latency_ms_p50": statistics.median(self.base_lat) if self.base_lat else None,
                "curve": self.curve[-400:],
                "ticks": list(self.ticks),
                "vendors": e.parser_table(),
                "events": e.events[-60:],
                "drifts": self.drifts[-20:],
                "verification": self.verification(),
                "recent": self.recent_rows(),
            }

    def doc(self, doc_id: str) -> Optional[dict]:
        with self.lock:
            r = self.engine.records.get(doc_id)
            if not r:
                return None
            out = {k: v for k, v in r.items() if not k.startswith("_")}
            out["text"] = self.texts.get(doc_id, "")
            out["correct"] = r.get("_correct")
            out["wrong_fields"] = r.get("_wrong_fields")
            out["truth"] = self.truth[doc_id].to_dict() if doc_id in self.truth else None
            out["damaged"] = self.damage.get(doc_id, {})
            v = self.stream.vendor(self.vendor_of[doc_id])
            out["scan"] = v.scan
            return out

    def vendor(self, vid: str) -> dict:
        with self.lock:
            vs = self.engine.vendors[vid]
            v = self.stream.vendor(vid)
            return {"vendor_id": vid, "name": v.name, "state": vs.monitor.state, "reason": vs.monitor.reason,
                    "history": self.engine.reg.parser_history(vid),
                    "active_spec": vs.active.spec if vs.active else None,
                    "shadow_spec": vs.shadow.spec if vs.shadow else None,
                    "drift_options": self.stream.eligible_drifts(vid), "template_rev": v.template_rev,
                    "monitor_events": vs.monitor.events[-10:]}

    def apply_drift(self, vid: str, kind: str) -> str:
        with self.lock:
            desc = self.stream.apply_drift(vid, kind)
            seq = self.engine.seq
            self.drifts.append({"seq": seq, "vendor_id": vid, "kind": kind, "class": DRIFT_KINDS[kind], "desc": desc})
            self.engine._event("template_changed", vid,
                               f"Template changed for {self.stream.vendor(vid).name}: {desc} ({DRIFT_KINDS[kind]})",
                               drift_kind=kind)
            return desc


    def warm(self, n: int) -> None:
        """Process n documents as fast as possible (no pacing), in the background."""
        def run():
            self.warming = n
            try:
                for _ in range(n):
                    if self.warming == 0:
                        break
                    self.step()
            finally:
                self.warming = 0
        threading.Thread(target=run, daemon=True).start()

    def switch(self, dataset: str, warm: int = 0) -> None:
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset {dataset!r}")
        self.running = False
        self.warming = 0
        time.sleep(0.05)
        self.dataset = dataset
        self.reset()
        if warm:
            self.warm(warm)


PUBLIC = False  # hosted for anyone: no reset, and the stream pauses itself when unattended
PUBLIC_RUN_SECONDS = 300
VIEWERS = 0  # open dashboard pages

SIM: Optional[Simulation] = None  # the stream on screen
SIMS: dict[str, Simulation] = {}  # one live stream per dataset, so switching keeps each one's progress
SIM_ARGS: dict = {}


# ---------------------------------------------------------------------- routes
async def index(request: Request):
    return FileResponse(WEB / "index.html")


async def how(request: Request):
    return FileResponse(WEB / "how.html")


async def state(request: Request):
    return JSONResponse(SIM.snapshot())


async def stream(request: Request):
    async def gen():
        global VIEWERS
        VIEWERS += 1
        try:
            while True:
                if await request.is_disconnected():
                    break
                yield f"data: {json.dumps(SIM.snapshot(), default=str)}\n\n"
                await asyncio.sleep(0.5)
        finally:
            VIEWERS -= 1
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def use_dataset(ds: str) -> None:
    """Show another dataset's stream. Each stream keeps its own progress; the one left behind pauses."""
    global SIM
    if ds not in DATASETS:
        raise ValueError(f"unknown dataset {ds!r}")
    if SIM is not None and SIM.dataset == ds:
        return
    if ds not in SIMS:
        SIMS[ds] = Simulation(dataset=ds, **SIM_ARGS)
    if SIM is not None:
        SIM.running = False
    SIM = SIMS[ds]


async def control(request: Request):
    body = await request.json()
    a = body.get("action")
    if a == "start":
        SIM.running = True
        SIM.started_at = time.time()
    elif a == "pause":
        SIM.running = False
    elif a == "reset":
        if PUBLIC:
            return JSONResponse({"ok": False, "error": "reset is off on the public dashboard"}, status_code=403)
        SIM.running = False
        SIM.reset()
    elif a == "step":
        SIM.step()
    elif a == "dataset":
        try:
            use_dataset(body.get("dataset", "synthetic"))
        except (ValueError, FileNotFoundError) as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    elif a == "warm":
        SIM.warm(int(body.get("n", 1000)))
    if "speed" in body:
        SIM.speed = max(0.2, min(200.0, float(body["speed"])))
    if body.get("profile") in ("economy", "balanced", "assurance"):
        SIM.profile = body["profile"]
        SIM.engine.cfg.profile = body["profile"]
    return JSONResponse({"ok": True, "running": SIM.running, "speed": SIM.speed, "profile": SIM.profile})


async def drift(request: Request):
    body = await request.json()
    try:
        desc = SIM.apply_drift(body["vendor_id"], body["kind"])
    except (KeyError, ValueError, StopIteration) as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    n = int(body.get("send", 0))
    for _ in range(min(n, 50)):
        SIM.pending.append(body["vendor_id"])
    return JSONResponse({"ok": True, "desc": desc})


async def send(request: Request):
    body = await request.json()
    for _ in range(min(int(body.get("n", 10)), 50)):
        SIM.pending.append(body["vendor_id"])
    return JSONResponse({"ok": True})


async def doc(request: Request):
    d = SIM.doc(request.path_params["doc_id"])
    return JSONResponse(d, status_code=200 if d else 404)


async def vendor(request: Request):
    try:
        return JSONResponse(SIM.vendor(request.path_params["vid"]))
    except (KeyError, StopIteration):
        return JSONResponse({"error": "unknown vendor"}, status_code=404)


async def rollback(request: Request):
    body = await request.json()
    with SIM.lock:
        pid = SIM.engine.rollback(body["vendor_id"])
    return JSONResponse({"ok": pid is not None, "parser_id": pid})


async def benchmark(request: Request):
    p = Path("results/benchmark.json")
    if not p.exists():
        return JSONResponse({"error": "run `python -m distill.bench` first"}, status_code=404)
    data = json.loads(p.read_text())
    for r in data:
        r.pop("events", None)
    return JSONResponse(data)


def build_app() -> Starlette:
    return Starlette(routes=[
        Route("/", index), Route("/how", how), Route("/api/state", state), Route("/api/stream", stream),
        Route("/api/control", control, methods=["POST"]), Route("/api/drift", drift, methods=["POST"]),
        Route("/api/send", send, methods=["POST"]), Route("/api/doc/{doc_id}", doc),
        Route("/api/vendor/{vid}", vendor), Route("/api/rollback", rollback, methods=["POST"]),
        Route("/api/benchmark", benchmark),
    ])


def main(argv=None):
    global SIM
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="use real Claude (ANTHROPIC_API_KEY)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--db", default="distill.db")
    ap.add_argument("--dataset", choices=list(DATASETS), default="synthetic",
                    help="synthetic: generated invoices with template changes; fatura: FATURA invoices")
    ap.add_argument("--fatura", default="data/fatura_invoices.jsonl.gz")
    ap.add_argument("--public", action="store_true",
                    help="hosted for anyone: hide reset, pause when nobody is watching or after 5 minutes of running")
    ap.add_argument("--warm", type=int, default=0,
                    help="process this many invoices of each dataset before the page opens (simulated mode), so the "
                         "demo starts with parsers in place, e.g. --warm 5000")
    args = ap.parse_args(argv)
    global SIMS, SIM_ARGS, PUBLIC
    PUBLIC = args.public
    SIM_ARGS = {"real": args.real, "seed": args.seed, "db": ":memory:", "fatura_path": args.fatura}
    SIM = SIMS[args.dataset] = Simulation(dataset=args.dataset, **{**SIM_ARGS, "db": args.db})
    if args.real:
        SIM.speed = 1.0
    elif args.warm:
        # warm both streams, so switching data on stage is instant and neither starts from zero
        order = [args.dataset] + [d for d in DATASETS if d != args.dataset]
        for ds in order:
            if ds == "fatura" and not Path(args.fatura).exists():
                print(f"  (no {args.fatura}: FATURA stream not available)", flush=True)
                continue
            sim = SIMS.get(ds) or SIMS.setdefault(ds, Simulation(dataset=ds, **SIM_ARGS))
            t0 = time.time()
            for i in range(1, args.warm + 1):
                sim.step()
                if i % 1000 == 0 or i == args.warm:
                    print(f"  warm start, {DATASETS[ds]}: {i}/{args.warm} invoices", flush=True)
            print(f"  done in {time.time() - t0:.0f} s", flush=True)
    import uvicorn
    print(f"Distill dashboard: http://{args.host}:{args.port}  ({SIM.snapshot()['mode']})")
    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
