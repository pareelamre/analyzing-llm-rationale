import html
import re
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import cairosvg


BASE = Path(__file__).resolve().parent
DRAWIO = BASE / "probabilistic_reasoning_temporal_inference.drawio"
SVG = BASE / "probabilistic_reasoning_temporal_inference.svg"
PDF = BASE / "probabilistic_reasoning_temporal_inference.pdf"


SUB = str.maketrans("0123456789:t+k", "₀₁₂₃₄₅₆₇₈₉:ₜ₊ₖ")


def style_dict(style):
    out = {}
    for part in (style or "").split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def esc(value):
    return html.escape(str(value), quote=True)


def parse_value(value):
    value = html.unescape(value or "")
    value = re.sub(r"<sub>(.*?)</sub>", lambda m: m.group(1).translate(SUB), value)
    value = value.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    raw_lines = value.split("\n")
    lines = []
    for raw in raw_lines:
        bold = "<b>" in raw or "font-style: 1" in raw
        small = "font-size: 12px" in raw
        clean = re.sub(r"<[^>]+>", "", raw).strip()
        if clean:
            lines.append((clean, bold, small))
    return lines


def wrap_line(text, width, font_size):
    max_chars = max(10, int(width / (font_size * 0.58)))
    return textwrap.wrap(text, width=max_chars) or [text]


def text_svg(x, y, width, height, value, style, boxed=False):
    st = style_dict(style)
    fs = int(float(st.get("fontSize", "14" if boxed else "12")))
    color = st.get("fontColor", "#111827")
    font_weight = "700" if st.get("fontStyle") == "1" else "400"
    align = st.get("align", "center")
    if align == "left":
        anchor, tx = "start", x + 10
    elif align == "right":
        anchor, tx = "end", x + width - 10
    else:
        anchor, tx = "middle", x + width / 2

    lines_in = parse_value(value)
    rendered = []
    for text, bold, small in lines_in:
        line_fs = 12 if small else fs
        for wrapped in wrap_line(text, max(20, width - 20), line_fs):
            rendered.append((wrapped, bold or font_weight == "700", line_fs))

    if not rendered:
        return ""

    line_heights = [size * 1.25 for _, _, size in rendered]
    total = sum(line_heights)
    start_y = y + (height - total) / 2 + rendered[0][2]
    parts = [
        f'<text x="{tx:.1f}" y="{start_y:.1f}" font-family="DejaVu Sans, Arial, Helvetica, sans-serif" '
        f'fill="{esc(color)}" text-anchor="{anchor}">'
    ]
    cy = start_y
    for i, (line, bold, line_fs) in enumerate(rendered):
        dy = 0 if i == 0 else line_heights[i - 1]
        parts.append(
            f'<tspan x="{tx:.1f}" dy="{dy:.1f}" font-size="{line_fs}" '
            f'font-weight="{"700" if bold else "400"}">{esc(line)}</tspan>'
        )
    parts.append("</text>")
    return "".join(parts)


def rect_svg(x, y, w, h, style):
    st = style_dict(style)
    fill = st.get("fillColor", "none")
    stroke = st.get("strokeColor", "none")
    sw = st.get("strokeWidth", "1")
    dash = ' stroke-dasharray="8 4"' if st.get("dashed") == "1" else ""
    rounded = st.get("rounded") == "1"
    rx = min(w, h) * 0.025 if rounded else 0
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx:.1f}" ry="{rx:.1f}" '
        f'fill="{esc(fill)}" stroke="{esc(stroke)}" stroke-width="{esc(sw)}"{dash}/>'
    )


def boundary_point(src, tgt):
    sx, sy, sw, sh = src
    tx, ty, tw, th = tgt
    scx, scy = sx + sw / 2, sy + sh / 2
    tcx, tcy = tx + tw / 2, ty + th / 2
    dx, dy = tcx - scx, tcy - scy
    if abs(dx) >= abs(dy):
        return (sx + sw if dx >= 0 else sx, scy)
    return (scx, sy + sh if dy >= 0 else sy)


def route_edge(cell, cells):
    src = cells[cell.attrib["source"]]
    tgt = cells[cell.attrib["target"]]
    geo = cell.find("mxGeometry")
    points = []
    if geo is not None:
        arr = geo.find("Array")
        if arr is not None:
            points = [(float(p.attrib["x"]), float(p.attrib["y"])) for p in arr.findall("mxPoint")]
    start = boundary_point(src, tgt if not points else (points[0][0], points[0][1], 1, 1))
    end = boundary_point(tgt, src if not points else (points[-1][0], points[-1][1], 1, 1))
    if points:
        return [start] + points + [end]

    sx, sy = start
    ex, ey = end
    if abs(sx - ex) < 2 or abs(sy - ey) < 2:
        return [start, end]
    midx = (sx + ex) / 2
    return [start, (midx, sy), (midx, ey), end]


def polyline_svg(points, style):
    st = style_dict(style)
    stroke = st.get("strokeColor", "#111827")
    sw = st.get("strokeWidth", "2")
    dash = ' stroke-dasharray="5 5"' if st.get("dashed") == "1" else ""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    marker = f'marker-end="url(#{marker_id(stroke)})"' if st.get("endArrow") else ""
    return f'<polyline points="{pts}" fill="none" stroke="{esc(stroke)}" stroke-width="{esc(sw)}" {dash} {marker}/>'


def marker_id(color):
    return "arrow_" + re.sub(r"[^A-Za-z0-9]", "", color)


def main():
    root = ET.parse(DRAWIO).getroot()
    graph = root.find(".//mxGraphModel")
    page_w = float(graph.attrib.get("pageWidth", 1400))
    page_h = float(graph.attrib.get("pageHeight", 900))
    mxroot = graph.find("root")

    cells = {}
    vertices = []
    edges = []
    for cell in mxroot.findall("mxCell"):
        geo = cell.find("mxGeometry")
        if geo is not None and cell.attrib.get("vertex") == "1":
            x = float(geo.attrib.get("x", 0))
            y = float(geo.attrib.get("y", 0))
            w = float(geo.attrib.get("width", 0))
            h = float(geo.attrib.get("height", 0))
            cells[cell.attrib["id"]] = (x, y, w, h)
            vertices.append((cell, x, y, w, h))
        elif cell.attrib.get("edge") == "1":
            edges.append(cell)

    colors = sorted({style_dict(e.attrib.get("style")).get("strokeColor", "#111827") for e in edges})
    defs = []
    for color in colors:
        defs.append(
            f'<marker id="{marker_id(color)}" markerWidth="10" markerHeight="10" refX="9" refY="3" '
            f'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="{esc(color)}"/></marker>'
        )

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{page_w}" height="{page_h}" viewBox="0 0 {page_w} {page_h}">',
        "<defs>",
        *defs,
        "</defs>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    for cell, x, y, w, h in vertices:
        style = cell.attrib.get("style", "")
        if not style.startswith("text;"):
            out.append(rect_svg(x, y, w, h, style))

    for edge in edges:
        out.append(polyline_svg(route_edge(edge, cells), edge.attrib.get("style", "")))

    for cell, x, y, w, h in vertices:
        style = cell.attrib.get("style", "")
        boxed = not style.startswith("text;")
        out.append(text_svg(x, y, w, h, cell.attrib.get("value", ""), style, boxed=boxed))

    out.append("</svg>")
    SVG.write_text("\n".join(out), encoding="utf-8")
    cairosvg.svg2pdf(url=str(SVG), write_to=str(PDF))
    print(f"Wrote {SVG}")
    print(f"Wrote {PDF}")


if __name__ == "__main__":
    main()
