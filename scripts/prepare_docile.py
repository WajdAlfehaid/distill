"""Convert the DocILE labeled-trainval download into one compact file for Distill.

    curl -O https://docile-dataset-rossum.s3.eu-west-1.amazonaws.com/<TOKEN>/labeled-trainval.zip
    python3 scripts/prepare_docile.py labeled-trainval.zip            # or an unzipped folder
    -> data/docile_trainval_words.jsonl.gz  (OCR text + word positions + labels + cluster ids, no PDFs, ~25 MB)

Standard library only. For every document it keeps:
  docid, split, cluster_id, document_type, currency, language, page_count,
  text    the precomputed DocTR OCR, rebuilt into reading-order lines (words on the same
          visual row joined; wide horizontal gaps kept as 3 spaces so columns survive)
  fields  {fieldtype: [annotated value texts in reading order]}  (KILE fields)
  words   [[page, x0, y0, x1, y1, text], ...] OCR words with positions (for geometric templates)
  items   [{fieldtype: text, ...} per line item]                  (LIR fields)

DocILE is distributed by Rossum under its own terms; this file is a derived
representation for local experiments. Check the terms you accepted before sharing it.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath


class Source:
    """Read files from either the downloaded zip or an unzipped folder."""

    def __init__(self, path: Path):
        self.path = path
        self.zip = zipfile.ZipFile(path) if path.suffix == ".zip" else None
        if self.zip:
            names = self.zip.namelist()
            # files may sit under a top-level folder inside the zip
            idx = next((n for n in names if n.endswith("annotations/") or "/annotations/" in n or n.startswith("annotations/")), "")
            self.prefix = idx.split("annotations/")[0] if "annotations/" in idx else ""
            self.names = set(names)

    def read(self, rel: str) -> bytes | None:
        if self.zip:
            name = self.prefix + rel
            return self.zip.read(name) if name in self.names else None
        p = self.path / rel
        return p.read_bytes() if p.exists() else None

    def docids(self) -> list[tuple[str, str]]:
        out = []
        for split in ("train", "val", "trainval", "test", "dev"):
            raw = self.read(f"{split}.json")
            if raw:
                for d in json.loads(raw):
                    out.append((d, split))
        if any(s in ("train", "val") for _, s in out):
            out = [(d, s) for d, s in out if s in ("train", "val")]
        if not out:  # no index file: take every annotation present
            if self.zip:
                ids = [PurePosixPath(n).stem for n in self.names if n.startswith(self.prefix + "annotations/") and n.endswith(".json")]
            else:
                ids = [p.stem for p in (self.path / "annotations").glob("*.json")]
            out = [(d, "all") for d in sorted(ids)]
        seen, uniq = set(), []
        for d, s in out:
            if d not in seen:
                seen.add(d)
                uniq.append((d, s))
        return uniq


def ocr_to_lines(ocr: dict) -> list[str]:
    """Rebuild reading-order text lines from DocTR output (pages > blocks > lines > words)."""
    pages_out = []
    for page in ocr.get("pages", []):
        words = []
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                for w in line.get("words", []):
                    (x0, y0), (x1, y1) = w["geometry"]
                    words.append((x0, y0, x1, y1, w["value"]))
        if not words:
            continue
        hs = sorted(w[3] - w[1] for w in words)
        med_h = hs[len(hs) // 2] or 0.01
        words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
        rows: list[list] = []
        for w in words:
            cy = (w[1] + w[3]) / 2
            if rows and abs(rows[-1][0] - cy) <= med_h * 0.55:
                r = rows[-1]
                r[1].append(w)
                r[0] = (r[0] * (len(r[1]) - 1) + cy) / len(r[1])
            else:
                rows.append([cy, [w]])
        lines = []
        for _, ws in rows:
            ws.sort(key=lambda w: w[0])
            parts, prev_x1 = [], None
            char_w = sum((w[2] - w[0]) / max(len(w[4]), 1) for w in ws) / len(ws)
            for w in ws:
                if prev_x1 is not None:
                    parts.append("   " if w[0] - prev_x1 > 2.5 * char_w else " ")
                parts.append(w[4])
                prev_x1 = w[2]
            lines.append("".join(parts))
        pages_out.append("\n".join(lines))
    return "\n\n".join(pages_out).splitlines()


def ocr_words(ocr: dict) -> list[list]:
    """Every OCR word as [page, x0, y0, x1, y1, text] with coordinates normalised to 0..1."""
    out = []
    for pi, page in enumerate(ocr.get("pages", [])):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                for w in line.get("words", []):
                    (x0, y0), (x1, y1) = w["geometry"]
                    out.append([pi, round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4), w["value"]])
    return out


def reading_order(f: dict) -> tuple:
    b = f.get("bbox") or [0, 0, 0, 0]
    return (f.get("page", 0), round(b[1], 3), b[0])


def convert(src: Source, out: Path, limit: int | None = None) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        for docid, split in src.docids():
            ann_raw, ocr_raw = src.read(f"annotations/{docid}.json"), src.read(f"ocr/{docid}.json")
            if ann_raw is None or ocr_raw is None:
                continue
            ann, ocr = json.loads(ann_raw), json.loads(ocr_raw)
            fields: dict[str, list[str]] = {}
            for f in sorted(ann.get("field_extractions", []), key=reading_order):
                if f.get("text"):
                    fields.setdefault(f["fieldtype"], []).append(f["text"])
            items: dict[int, dict] = {}
            for f in sorted(ann.get("line_item_extractions", []), key=reading_order):
                if f.get("text") and f.get("line_item_id") is not None:
                    it = items.setdefault(f["line_item_id"], {})
                    it[f["fieldtype"]] = (it[f["fieldtype"]] + " " + f["text"]) if f["fieldtype"] in it else f["text"]
            md = ann.get("metadata", {})
            rec = {"docid": docid, "split": split, "cluster_id": md.get("cluster_id"),
                   "document_type": md.get("document_type"), "currency": md.get("currency"),
                   "language": md.get("language"), "page_count": md.get("page_count"),
                   "text": "\n".join(ocr_to_lines(ocr)), "words": ocr_words(ocr), "fields": fields,
                   "items": [items[k] for k in sorted(items)]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 500 == 0:
                print(f"  {n} documents", flush=True)
            if limit and n >= limit:
                break
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="labeled-trainval.zip or the folder it unzips to")
    ap.add_argument("--out", default="data/docile_trainval_words.jsonl.gz")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    src = Source(Path(args.source))
    n = convert(src, Path(args.out), args.limit)
    if n == 0:
        sys.exit("no documents found: is this the labeled-trainval download?")
    size = Path(args.out).stat().st_size / 1e6
    print(f"wrote {n} documents to {args.out} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
