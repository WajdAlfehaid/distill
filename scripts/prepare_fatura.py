"""Convert FATURA (Limam et al. 2023, CC BY 4.0; https://zenodo.org/records/10371464) for Distill.

    python scripts/prepare_fatura.py data/fatura/FATURA2.zip      # -> data/fatura_invoices.jsonl.gz

Each FATURA annotation carries the page's text layer ("OTHER") and the text of every labelled region.
We keep the single-line header fields an accounts-payable system extracts, and take the value as the region
text after its printed label ("TOTAL : 441.14 EUR" -> "441.14 EUR"). Multi-line address blocks (buyer,
seller address) are left out: the parser DSL reads single-line values (a stated limitation). The seller name
is left out because FATURA renders it inside the logo image, so it is not in the text layer.
"""
import gzip
import json
import re
import sys
import zipfile

FIELDS = {"NUMBER": "invoice_number", "DATE": "invoice_date", "DUE_DATE": "due_date", "PO_NUMBER": "po_number",
          "SUB_TOTAL": "subtotal", "TAX": "tax", "DISCOUNT": "discount",
          "TOTAL": "total", "AMOUNT_DUE": "amount_due"}
LABEL = re.compile(r"^\s*[A-Za-z][A-Za-z .()#/&%\d_-]{0,40}?\s*(?::|#|\s{2,})\s*")


def value_of(text: str) -> str:
    """Region text as printed (label included); distill.realdata.fatura_value strips the label when loading."""
    return " ".join(text.split())


def main(src, out="data/fatura_invoices.jsonl.gz"):
    z = zipfile.ZipFile(src)
    names = [n for n in z.namelist() if "/Original_Format/" in n and n.endswith(".json")]
    n = 0
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        for name in sorted(names):
            m = re.search(r"Template(\d+)_Instance(\d+)", name)
            if not m:
                continue
            a = json.loads(z.read(name))
            text = (a.get("OTHER") or {}).get("text") if isinstance(a.get("OTHER"), dict) else None
            if not text:
                continue
            fields = {}
            for k, f in FIELDS.items():
                r = a.get(k)
                if isinstance(r, dict) and r.get("text", "").strip():
                    fields[f] = value_of(r["text"])
            fh.write(json.dumps({"docid": f"T{m.group(1)}-{m.group(2)}", "template": int(m.group(1)),
                                 "instance": int(m.group(2)), "text": text, "fields": fields}) + "\n")
            n += 1
    print(f"{n} invoices -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
