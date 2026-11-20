"""Convert the CFF-flavored OpenType fonts used in the figures (TeX Gyre Termes X, Fira) to TrueType once.

Matplotlib embeds fonts in PDFs as TrueType (Type 42); feeding it CFF outlines produces PDFs whose font type does
not match the embedded font program, which some publisher preflight checks reject. Quadratic TrueType copies are
cached in scripts/.fontcache and registered with Matplotlib instead of the originals.
"""
from pathlib import Path

from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable

CACHE = Path(__file__).resolve().parent / ".fontcache"


def _quadratic_glyphs(glyph_set, max_err=1.0):
    out = {}
    for name in glyph_set.keys():
        tt_pen = TTGlyphPen(glyph_set)
        glyph_set[name].draw(Cu2QuPen(tt_pen, max_err, reverse_direction=True))
        out[name] = tt_pen.glyph()
    return out


def otf_to_ttf(src: Path) -> Path:
    """Return a TrueType copy of `src` (converted on first use)."""
    CACHE.mkdir(exist_ok=True)
    dst = CACHE / (src.stem + ".ttf")
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    font = TTFont(str(src))
    if "CFF " not in font:  # already TrueType outlines
        return src
    order = font.getGlyphOrder()
    font["loca"] = newTable("loca")
    font["glyf"] = glyf = newTable("glyf")
    glyf.glyphOrder = order
    glyf.glyphs = _quadratic_glyphs(font.getGlyphSet())
    del font["CFF "]
    for t in ("VORG",):
        if t in font:
            del font[t]
    glyf.compile(font)
    hmtx = font["hmtx"]
    for name, g in glyf.glyphs.items():
        if hasattr(g, "xMin"):
            hmtx[name] = (hmtx[name][0], g.xMin)
    font["maxp"] = maxp = newTable("maxp")
    maxp.tableVersion = 0x00010000
    for k in ("maxZones",):
        setattr(maxp, k, 1)
    for k in ("maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs", "maxStackElements",
              "maxSizeOfInstructions"):
        setattr(maxp, k, 0)
    maxp.maxComponentElements = max((len(getattr(g, "components", []) or []) for g in glyf.glyphs.values()), default=0)
    maxp.compile(font)
    post = font["post"]
    post.formatType, post.extraNames, post.mapping, post.glyphOrder = 2.0, [], {}, order
    try:
        post.compile(font)
    except OverflowError:
        post.formatType = 3
    font.sfntVersion = "\000\001\000\000"
    font.save(str(dst))
    return dst
