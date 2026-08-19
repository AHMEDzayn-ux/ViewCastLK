"""Extract SAD v2.0 into a structured, ordered content model.

Emits a list of blocks: ('h', level, text) | ('p', text) | ('b', text) bullet
| ('fig', caption) | ('tbl', rows). Tables are spliced in at their vertical
position on the page so document order is preserved.
"""
import json
import os
import re

import fitz

SRC = r"C:\Users\sabit\Downloads\ViewCastLK_Software_Architecture_Document_v2.0.pdf"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sad_blocks.json")

HEADER = re.compile(r"ViewCastLK\s*\|\s*Software Architecture Document")
FOOTER = re.compile(r"ViewCastLK-SAD-2\.0|©\s*University of Moratuwa|·\s*Page\s*\d+")
HEADING = re.compile(r"^(\d+(?:\.\d+){0,2})\.?\s+([A-Z].*)$")
FIGCAP = re.compile(r"^Figure\s+(\d+)\.\s*(.*)$")
APPENDIX = re.compile(r"^Appendix\s+([AB])\.\s*(.*)$")


def clean(s):
    s = s.replace("\u00a0", " ").replace("\u2019", "'").replace("\u201c", '"')
    s = s.replace("\u201d", '"').replace("\u2014", "—").replace("\u2013", "–")
    return re.sub(r"[ \t]+", " ", s).strip()


def main():
    doc = fitz.open(SRC)
    blocks = []
    for pno in range(len(doc)):
        page = doc[pno]

        # --- tables on this page, with their vertical position
        tabs = []
        try:
            for t in page.find_tables().tables:
                rows = [[clean((c or "").replace("\n", " ")) for c in r]
                        for r in t.extract()]
                rows = [r for r in rows if any(r)]
                if rows:
                    tabs.append((t.bbox[1], rows, t.bbox))
        except Exception:
            pass
        tab_zones = [(b[1], b[3]) for _, _, b in tabs]

        # --- text blocks with position, skipping anything inside a table bbox
        tblocks = []
        for x0, y0, x1, y1, txt, *_ in page.get_text("blocks"):
            if any(y0 >= z0 - 2 and y1 <= z1 + 2 for z0, z1 in tab_zones):
                continue
            for line in txt.split("\n"):
                line = clean(line)
                if not line or HEADER.search(line) or FOOTER.search(line):
                    continue
                tblocks.append((y0, line))

        merged = sorted(tblocks + [(y, ("__TABLE__", rows)) for y, rows, _ in tabs],
                        key=lambda x: x[0])

        # --- classify
        buf = []

        def flush():
            if buf:
                blocks.append(("p", " ".join(buf)))
                buf.clear()

        for _, item in merged:
            if isinstance(item, tuple):
                flush()
                blocks.append(("tbl", item[1]))
                continue
            line = item
            m = APPENDIX.match(line)
            if m:
                flush()
                blocks.append(("h", 1, f"Appendix {m.group(1)}. {m.group(2)}"))
                continue
            m = FIGCAP.match(line)
            if m:
                flush()
                blocks.append(("fig", int(m.group(1)), m.group(2)))
                continue
            m = HEADING.match(line)
            if m and len(line) < 90 and not line.endswith("."):
                flush()
                blocks.append(("h", m.group(1).count(".") + 1,
                               f"{m.group(1)} {m.group(2)}"))
                continue
            if line.startswith(("•", "●", "-\t")):
                flush()
                blocks.append(("b", clean(line.lstrip("•● \t"))))
                continue
            buf.append(line)
        flush()

    # --- merge consecutive body paragraphs that were split across pages
    out = []
    for b in blocks:
        if (b[0] == "p" and out and out[-1][0] == "p"
                and not out[-1][1].rstrip().endswith((".", ":", "?", "!"))):
            out[-1] = ("p", out[-1][1] + " " + b[1])
        else:
            out.append(b)

    json.dump(out, open(OUT, "w", encoding="utf8"), ensure_ascii=False, indent=1)
    kinds = {}
    for b in out:
        kinds[b[0]] = kinds.get(b[0], 0) + 1
    print("blocks:", kinds, "total", len(out))
    for b in out[:8]:
        print(" ", str(b)[:130])


if __name__ == "__main__":
    main()
