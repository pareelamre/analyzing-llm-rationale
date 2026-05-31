"""
Convert draw.io SVG foreignObject text elements to native SVG <text> elements
so that cairosvg / Inkscape can render them.
"""
import re
import sys

def strip_tags(html):
    """Remove HTML tags, return plain text lines split on <br>."""
    html = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'<[^>]+>', '', html)
    html = html.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    html = html.replace('&quot;', '"').replace('&#xa;', '\n').replace('&#10;', '\n')
    return html

def css_val(style, prop):
    """Extract a CSS property value from an inline style string."""
    m = re.search(rf'{re.escape(prop)}\s*:\s*([^;]+)', style)
    return m.group(1).strip() if m else None

def px(val):
    """Parse a pixel value string like '104px' -> 104.0"""
    if val is None:
        return 0.0
    m = re.match(r'([\d.]+)', str(val))
    return float(m.group(1)) if m else 0.0

def make_svg_text(x, y, text, anchor, font_size, font_weight, color, font_family, line_height_factor=1.25):
    lines = text.split('\n')
    n = len(lines)
    fs = float(font_size)
    lh = fs * line_height_factor

    # padding-top is the baseline of the first line (draw.io encodes it this way
    # for both top-aligned multi-line cells and single-line centred cells).
    y0 = y

    color = color.strip()
    # strip light-dark() wrapper draw.io sometimes emits
    ld = re.match(r'light-dark\(\s*([^,]+),', color)
    if ld:
        color = ld.group(1).strip()

    weight_attr = f' font-weight="{font_weight}"' if font_weight and font_weight != 'normal' else ''
    family_attr = f' font-family="{font_family}"' if font_family else ' font-family="Helvetica"'

    if n == 1:
        return (
            f'<text x="{x:.1f}" y="{y0:.1f}" text-anchor="{anchor}"'
            f' font-size="{fs:.0f}"{weight_attr}{family_attr} fill="{color}">'
            f'{lines[0]}'
            f'</text>'
        )

    tspans = []
    for i, line in enumerate(lines):
        dy = "0" if i == 0 else f"{lh:.1f}"
        tspans.append(f'<tspan x="{x:.1f}" dy="{dy}">{line}</tspan>')
    return (
        f'<text x="{x:.1f}" y="{y0:.1f}" text-anchor="{anchor}"'
        f' font-size="{fs:.0f}"{weight_attr}{family_attr} fill="{color}">'
        + ''.join(tspans) +
        '</text>'
    )

def convert(svg_path, out_path):
    with open(svg_path, encoding='utf-8') as f:
        content = f.read()

    # Pattern: <g transform="translate(...)"><switch><foreignObject ...>...</foreignObject><text ...></text></switch></g>
    # or without the fallback <text> element.
    # We replace the entire <switch>...</switch> block with native SVG text.

    switch_pat = re.compile(
        r'<g[^>]*>\s*<switch>\s*<foreignObject[^>]*>(.*?)</foreignObject>.*?</switch>\s*</g>',
        re.DOTALL
    )

    def replace_switch(m):
        inner_html = m.group(1)

        # --- outer flex div: position info ---
        outer_div = re.search(
            r'<div[^>]+style="display:\s*flex;([^"]+)"',
            inner_html
        )
        if not outer_div:
            return m.group(0)  # can't parse, leave as-is

        outer_style = outer_div.group(1)
        width  = px(css_val(outer_style, 'width'))
        pad_y  = px(css_val(outer_style, 'padding-top'))
        mar_x  = px(css_val(outer_style, 'margin-left'))
        valign = css_val(outer_style, 'align-items') or 'center'
        halign = css_val(outer_style, 'justify-content') or 'center'

        # --- middle div: text-align and color ---
        mid_div = re.search(r'<div[^>]+style="box-sizing[^"]*"[^>]*>', inner_html)
        mid_style = mid_div.group(0) if mid_div else ''
        text_align = css_val(mid_style, 'text-align') or ('center' if 'center' in halign else 'left')
        color = css_val(mid_style, 'color') or '#000000'

        # --- innermost div: font props + text ---
        inner_div = re.search(
            r'<div[^>]+style="display:\s*inline-block;([^"]+)"[^>]*>(.*?)</div>',
            inner_html, re.DOTALL
        )
        if not inner_div:
            return m.group(0)

        inner_style = inner_div.group(1).replace('&quot;', '"').replace('&amp;', '&')
        inner_text  = inner_div.group(2)

        font_size   = px(css_val(inner_style, 'font-size')) or 12
        font_weight = css_val(inner_style, 'font-weight') or 'normal'
        font_family = css_val(inner_style, 'font-family') or 'Helvetica'
        font_family = font_family.replace('&quot;', '').replace('"', '').replace("'", '').strip()

        # Override color from innermost div if present
        inner_color = css_val(inner_style, 'color')
        if inner_color:
            color = inner_color

        text = strip_tags(inner_text).strip()
        if not text:
            return ''  # empty cell
        # Escape XML special chars in text content
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        # Compute SVG anchor and x
        anchor_map = {
            'center': 'middle', 'unsafe center': 'middle',
            'flex-start': 'start', 'unsafe flex-start': 'start',
            'flex-end': 'end',   'unsafe flex-end': 'end',
        }
        anchor = anchor_map.get(halign.strip(), 'middle')
        if text_align == 'left':
            anchor = 'start'
        elif text_align == 'right':
            anchor = 'end'

        if anchor == 'middle':
            x = mar_x + width / 2.0
        elif anchor == 'start':
            x = mar_x + 2
        else:
            x = mar_x + width - 2

        y = pad_y

        svg_text = make_svg_text(x, y, text, anchor, font_size, font_weight, color, font_family)
        return svg_text

    new_content = switch_pat.sub(replace_switch, content)

    # Remove any remaining SVG fallback text left over from <switch> elements
    new_content = re.sub(
        r'<text[^>]*>\s*Text is not SVG[^<]*</text>',
        '', new_content
    )

    removed = content.count('<foreignObject')
    remaining = new_content.count('<foreignObject')
    converted = removed - remaining
    print(f"Converted {converted}/{removed} foreignObject elements.")

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    base = '/data/horse/ws/paam844f-codabench-restored/paam844f-codabench-1777266844/analyzing-llm-rationale/paper'
    convert(f'{base}/system_flow.svg', f'{base}/system_flow_fixed.svg')
