"""Check that text fits inside each drawio cell (estimate wrap + line height vs box size)."""
import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path(__file__).resolve().parent
DRAWIO = BASE / "system_flow.drawio"

# Approximate Helvetica average glyph width as a fraction of font size
CHAR_W = 0.52
LINE_H = 1.28


def cell_text(value: str) -> list[str]:
    """Decode the HTML-escaped value into plain text lines (split on <br>)."""
    value = html.unescape(value)
    value = re.sub(r"<sub>(.*?)</sub>", r"\1", value)
    lines = re.split(r"<br\s*/?>", value)
    return [re.sub(r"<[^>]+>", "", ln).strip() for ln in lines]


def wrapped_lines(text: str, box_w: float, font_size: float, spacing: float) -> int:
    avail = max(10.0, box_w - 2 * spacing)
    max_chars = max(4, int(avail / (font_size * CHAR_W)))
    n = 0
    for line in text.split("\n"):
        words = line.split()
        if not words:
            n += 1
            continue
        cur = words[0]
        n += 1
        for w in words[1:]:
            if len(cur) + 1 + len(w) > max_chars:
                n += 1
                cur = w
            else:
                cur = f"{cur} {w}"
    return n


def main() -> None:
    root = ET.parse(DRAWIO).getroot()
    problems = []
    for cell in root.iter("mxCell"):
        if cell.attrib.get("vertex") != "1":
            continue
        style = cell.attrib.get("style", "")
        if style.startswith("text;"):
            continue  # free text cells have no box to overflow
        geo = cell.find("mxGeometry")
        if geo is None:
            continue
        w = float(geo.attrib.get("width", 0))
        h = float(geo.attrib.get("height", 0))
        fs = float(re.search(r"fontSize=(\d+)", style).group(1)) if re.search(r"fontSize=(\d+)", style) else 12
        spacing = float(re.search(r"spacing=(\d+)", style).group(1)) if re.search(r"spacing=(\d+)", style) else 2
        lines = cell_text(cell.attrib.get("value", ""))
        n = sum(wrapped_lines(ln, w, fs, spacing) for ln in lines if ln)
        need = n * fs * LINE_H + 2 * spacing
        if need > h:
            problems.append((cell.attrib.get("id"), w, h, fs, n, round(need, 1)))

    if problems:
        print("OVERFLOW:")
        for cid, w, h, fs, n, need in problems:
            print(f"  {cid}: box {w:.0f}x{h:.0f} fs={fs:.0f} lines={n} needs~{need}px")
        raise SystemExit(1)
    print("All boxed cells fit.")


if __name__ == "__main__":
    main()
