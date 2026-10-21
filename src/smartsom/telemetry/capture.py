"""Portable terminal SVG evidence with the same cell geometry as Rich output."""

import re
from xml.etree import ElementTree as ET

from rich.cells import cell_len
from rich.text import Text

SVG = "http://www.w3.org/2000/svg"
FONT = "monospace"


def cell_svg(svg, *, columns):
    """Position glyphs by terminal cells, including in existing Rich SVG files.

    Rich 14's SVG textLength counts code points while its x offsets count cells.
    CJK text is consequently squeezed to half its allocated width. Explicit
    glyph positions also work in SVG viewers that ignore textLength entirely.
    No installed-font measurement or remote web font determines column positions.
    """
    root = ET.fromstring(svg)
    line = next(
        element
        for element in root.iter(f"{{{SVG}}}clipPath")
        if element.get("id", "").endswith("-clip-terminal")
    )
    cell_width = (float(line[0].get("width")) + 1) / columns
    for style in root.iter(f"{{{SVG}}}style"):
        css = re.sub(r"@font-face\s*\{[^}]*\}", "", style.text or "")
        css = css.replace("Fira Code, monospace", FONT)
        css = css.replace("font-variant-east-asian: full-width;", "")
        css = css.replace("font-family: arial;", f"font-family: {FONT};")
        css = css.replace("font-size: 18px;", "font-size: 14px;")
        style.text = css

    for parent in list(root.iter()):
        children = []
        last_glyph = None
        for element in parent:
            if (
                element.tag != f"{{{SVG}}}text"
                or "textLength" not in element.attrib
                or element.get("class", "").endswith("-title")
            ):
                children.append(element)
                continue
            x = float(element.get("x"))
            for char in element.text or "":
                if ord(char) < 32:
                    continue
                cells = cell_len(char)
                if not cells:
                    if last_glyph is not None:
                        last_glyph.text += char
                    continue
                glyph = ET.Element(element.tag, element.attrib)
                glyph.set("x", f"{x:g}")
                glyph.set("textLength", f"{cells * cell_width:g}")
                glyph.set("lengthAdjust", "spacingAndGlyphs")
                # The terminal viewport clips whole rows; individual glyphs
                # must not depend on Rich's missing final-line clip definition.
                glyph.attrib.pop("clip-path", None)
                glyph.text = char
                children.append(glyph)
                last_glyph = glyph
                x += cells * cell_width
        parent[:] = children

    # Keep long capture names out of the window buttons even on tiny screens.
    canvas_width = float(root.get("viewBox").split()[2])
    for title in root.iter(f"{{{SVG}}}text"):
        if title.get("class", "").endswith("-title"):
            label = Text(title.text or "")
            label.truncate(max(1, int((canvas_width - 120) / 8.4)), overflow="ellipsis")
            title.text = label.plain
            title.set("x", f"{(90 + canvas_width - 16) / 2:g}")
            title.set("textLength", f"{cell_len(label.plain) * 8.4:g}")
            title.set("lengthAdjust", "spacingAndGlyphs")
    ET.register_namespace("", SVG)
    return ET.tostring(root, encoding="unicode")


def export_terminal_svg(console, *, title="SmartSOM"):
    """Export a recorded Console without changing its contents or theme."""
    return cell_svg(console.export_svg(title=title, clear=False), columns=console.width)
