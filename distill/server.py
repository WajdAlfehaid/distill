"""Live demo server: streams invoices through Distill and serves the dashboard.

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
import time
from collections import deque
from pathlib import Path
from typing import Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from .datagen import DEMO_OVERRIDES, DRIFT_KINDS, InvoiceStream
from .doctypes import InvoiceType
from .engine import Config, Engine
from .llm import LARGE_MODEL, SMALL_MODEL, AnthropicLLM, SimulatedLLM, cost_of, SYSTEM_PROMPT
from .model import Invoice, diff
from .registry import Registry

WEB = Path(__file__).resolve().parent.parent / "web"


class Simulation:
    def __init__(self, real: bool = False, seed: int = 11, db: str = ":memory:", profile: str = "balanced"):
        self.real, self.seed, self.db, self.profile = real, seed, db, profile
        self.lock = threading.RLock()
        self.running = False
        self.speed = 8.0  # documents per second
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
            self.stream = InvoiceStream(seed=self.seed, vendor_overrides=DEMO_OVERRIDES)
            self.truth: dict[str, Invoice] = {}
            self.damage: dict[str, dict] = {}
            self.vendor_of: dict[str, str] = {}
            self.texts: dict[str, str] = {}
            llm = AnthropicLLM() if self.real else SimulatedLLM(lambda d: self.truth[d], seed=self.seed,
                                                                damage=lambda d: self.damage.get(d))
            self.engine = Engine(InvoiceType(self.stream.vendor_master(), self.stream.po_master), llm,
                                 Config(mode="distill", profile=self.profile), Registry(self.db))
            self.cf = {"large": 0.0, "small": 0.0}  # counterfactual: every document through one model
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
            rec = self.engine.process(d.doc_id, d.text, d.received_at)
            tin, tout = SimulatedLLM.tokens_for(d.text, d.truth)
            self.cf["large"] += cost_of(LARGE_MODEL, tin, tout)
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

    def _verify(self, rec: dict) -> None:
        """Compare a result with the answer key (available because the demo data is generated)."""
        truth = self.truth[rec["doc_id"]]
        pred = Invoice.from_dict(rec["final"]) if rec["final"] else Invoice()
        bad = diff(pred, truth)
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
                "mode": "real Claude" if self.real else "simulated LLM tiers",
                "running": self.running, "speed": self.speed, "profile": self.profile,
                "models": {"small": SMALL_MODEL, "large": LARGE_MODEL},
                "totals": T, "wrong": self.wrong, "accuracy": 1 - self.wrong / n if T["docs"] else None,
                "cost_per_1k": 1000 * T["cost_usd"] / n,
                "cost_per_1k_recent": 1000 * sum(r["cost_usd"] for r in recent) / max(len(recent), 1),
                "code_share_recent": last_code,
                "counterfactual": self.cf,
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


SIM: Optional[Simulation] = None


# ---------------------------------------------------------------------- routes
async def index(request: Request):
    return FileResponse(WEB / "index.html")


async def state(request: Request):
    return JSONResponse(SIM.snapshot())


async def stream(request: Request):
    async def gen():
        while True:
            if await request.is_disconnected():
                break
            yield f"data: {json.dumps(SIM.snapshot(), default=str)}\n\n"
            await asyncio.sleep(0.5)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


async def control(request: Request):
    body = await request.json()
    a = body.get("action")
    if a == "start":
        SIM.running = True
    elif a == "pause":
        SIM.running = False
    elif a == "reset":
        SIM.running = False
        SIM.reset()
    elif a == "step":
        SIM.step()
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
        Route("/", index), Route("/api/state", state), Route("/api/stream", stream),
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
    args = ap.parse_args(argv)
    SIM = Simulation(real=args.real, seed=args.seed, db=args.db)
    if args.real:
        SIM.speed = 1.0
    import uvicorn
    print(f"Distill dashboard: http://{args.host}:{args.port}  ({SIM.snapshot()['mode']})")
    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
