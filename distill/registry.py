"""Parser registry and document ledger (SQLite).

Every parser version is stored with its spec, lineage, status history and the
evidence that justified each transition. Every document stores which tier and
which parser version produced its result, the checks it passed, its cost, and
the reasons for its route, so any number on the dashboard can be traced back.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS parsers (
  parser_id TEXT PRIMARY KEY, vendor_id TEXT, version INTEGER, status TEXT, generator TEXT,
  parent_id TEXT, created_seq INTEGER, promoted_seq INTEGER, retired_seq INTEGER, retire_reason TEXT,
  spec TEXT, evidence TEXT
);
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY, seq INTEGER, vendor_id TEXT, received_at TEXT, source TEXT, parser_id TEXT,
  route TEXT, extraction TEXT, checks TEXT, cost_usd REAL, latency_ms REAL, flags TEXT, trace TEXT
);
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER, vendor_id TEXT, kind TEXT, message TEXT, data TEXT
);
"""


@dataclass
class ParserVersion:
    parser_id: str
    vendor_id: str
    version: int
    spec: dict
    generator: str
    status: str = "shadow"  # shadow | active | suspended | retired | rejected
    parent_id: Optional[str] = None
    created_seq: int = 0
    promoted_seq: Optional[int] = None
    retired_seq: Optional[int] = None
    retire_reason: str = ""
    validation: dict = field(default_factory=dict)
    shadow_agree: int = 0
    shadow_disagree: int = 0
    audits_clean: int = 0
    audits_failed: int = 0
    docs: int = 0

    def evidence(self) -> dict:
        return {"validation": self.validation, "shadow_agree": self.shadow_agree, "shadow_disagree": self.shadow_disagree,
                "audits_clean": self.audits_clean, "audits_failed": self.audits_failed, "docs": self.docs}


class Registry:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def save_parser(self, p: ParserVersion) -> None:
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO parsers VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (p.parser_id, p.vendor_id, p.version, p.status, p.generator, p.parent_id, p.created_seq,
                 p.promoted_seq, p.retired_seq, p.retire_reason, json.dumps(p.spec), json.dumps(p.evidence())))

    def save_doc(self, rec: dict) -> None:
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec["doc_id"], rec["seq"], rec.get("vendor_id"), rec.get("received_at"), rec.get("source"),
                 rec.get("parser_id"), json.dumps(rec.get("route")), json.dumps(rec.get("final")),
                 json.dumps(rec.get("checks")), rec.get("cost_usd", 0.0), rec.get("latency_ms", 0.0),
                 json.dumps(rec.get("flags", [])), json.dumps(rec.get("trace", []))))

    def log(self, seq: int, vendor_id: Optional[str], kind: str, message: str, data: Optional[dict] = None) -> None:
        with self.lock:
            self.db.execute("INSERT INTO events VALUES (?,?,?,?,?)", (seq, vendor_id, kind, message, json.dumps(data or {})))

    def commit(self) -> None:
        with self.lock:
            self.db.commit()

    def parser_history(self, vendor_id: str) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT parser_id, version, status, generator, parent_id, created_seq, promoted_seq, "
                                   "retired_seq, retire_reason, evidence FROM parsers WHERE vendor_id=? ORDER BY version",
                                   (vendor_id,)).fetchall()
        keys = ["parser_id", "version", "status", "generator", "parent_id", "created_seq", "promoted_seq", "retired_seq",
                "retire_reason", "evidence"]
        return [dict(zip(keys, r[:-1] + (json.loads(r[-1]),))) for r in rows]
