"""Fig. 1 (how prices are written on real boards) and the numbers quoted with it.

The crops come from the redacted release (StreetMenuVN_final/images, i.e. Data_redacted); no unredacted photograph
is read. Every transcription and reading printed under a crop is checked against the gold label of that offer, so the
figure cannot drift from the data. Crops are cached in data/crops/ (small JPEGs), which lets the figure be rebuilt
without the image folder. Image folder: $SMV_IMAGES, else ../StreetMenuVN_final/images next to the paper folder.
"""
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from common import DATA, INK, INK2, MUTED, ROOT, TEXTWIDTH, Numbers, fmt_int, load_gold, save

SERIF = "TeX Gyre TermesX"  # the body font of the paper; every string of the figure uses it
CACHE = DATA / "crops"
IMG_DIR = Path(os.environ.get("SMV_IMAGES") or ROOT.parent / "data" / "images")

gold = {r["file_name"]: r for r in load_gold()}


def offers(fn):
    return [(it, o) for it in gold[fn]["items"] for o in it["offers"]]


def check(fn, raw, price, kind=None, source=None, n=1, old=None):
    """Assert that image `fn` has at least `n` gold offers written `raw` with normalized price `price`."""
    hits = [o for _, o in offers(fn) if o["raw_price"] == raw and o["price_vnd"] == price
            and (kind is None or o["price_kind"] == kind) and (source is None or o["price_source"] == source)
            and (old is None or o["old_price_vnd"] == old)]
    assert len(hits) >= n, (fn, raw, price, kind, source, len(hits))


# ----------------------------------------------------------------------------- numbers for the text
N = Numbers("prices")
o15 = [(r, o) for r in gold.values() for it in r["items"] for o in it["offers"] if o["price_vnd"] == 15000]
N["nFifteenOffers"] = fmt_int(len(o15))
N["nFifteenImages"] = len({r["file_name"] for r, _ in o15})
N["nFifteenForms"] = len({o["raw_price"] for _, o in o15})
freq = Counter(o["price_vnd"] for r in gold.values() for it in r["items"] for o in it["offers"] if o["price_vnd"])
N["fifteenRank"] = ["most", "second most", "third most", "fourth most", "fifth most"][
    [v for v, _ in freq.most_common()].index(15000)]
N.write()

# ----------------------------------------------------------------------------- crops (file, box in pixels of the
# EXIF-corrected redacted image) and the gold facts that the labels state
TOKENS = [  # (a) one amount, eight spellings; label = raw string as written in the gold
    ("tok_1589", "IMG_1589.jpg", (1265, 45, 1565, 205), "15K"),
    ("tok_2028", "IMG_2028.jpg", (452, 360, 565, 424), "15k"),
    ("tok_1608", "IMG_1608.jpg", (498, 212, 568, 266), "15"),
    ("tok_1985", "IMG_1985.jpg", (2262, 515, 2505, 600), "15.000"),
    ("tok_2114", "IMG_2114.jpg", (632, 530, 772, 600), "15,000"),
    ("tok_1617", "IMG_1617.jpg", (1490, 1026, 1636, 1080), "15.000đ"),
    ("tok_1568", "IMG_1568.jpg", (856, 238, 990, 322), "15"),
    ("tok_1572", "IMG_1572.jpg", (350, 78, 448, 138), "15K"),
]
for _, fn, _, raw in TOKENS:
    check(fn, raw, 15000)
MEDIUM = {"băng rôn / bảng hiệu": "signboard", "giấy in dán tạm": "printout", "viết tay trên giấy": "handwritten",
          "bảng phấn / kính": "chalkboard", "bảng đèn": "light box", "in sẵn để bàn": "table menu"}
TIER = {"chain": "chain outlet", "small_shop": "small shop", "street_vendor": "street stall"}
TOKEN_NOTE = [MEDIUM.get(gold[fn]["medium"]) or TIER[gold[fn]["venue_tier"]] for _, fn, _, _ in TOKENS]

check("IMG_1652.jpg", "15K /1 nải", 15000, kind="unit_price")
check("IMG_1652.jpg", "2 Trái - 15K", 15000, kind="bundle")
check("IMG_1492.jpg", "15K", 15000, source="handwritten_override", old=13000)
check("IMG_1492.jpg", "17K", 17000, source="handwritten_override", old=15000)
check("IMG_2022.jpg", "15.000Đ", 15000, source="section_header", n=10)
check("IMG_2066.jpg", "CHỈ TỪ 399 NGÀN", 399000, kind="starting_from")  # ngàn: thousand, Southern usage
assert gold["IMG_1586.jpg"]["price_unit_default"] == "k_vnd" and "nghìn đồng" in gold["IMG_1586.jpg"]["notes"]
check("IMG_1586.jpg", "70", 70000, n=2)  # bare numbers read in thousands, as the board's unit note says
check("IMG_1575.jpg", "1tr9", 1900000)  # tr: triệu, million
check("IMG_1575.jpg", "2tr2", 2200000)
check("IMG_2063.jpg", "5.999K", 5999000, old=6970000)  # the dot separates thousands: 5,999 thousand
check("IMG_2063.jpg", "559K", 559000, old=695000)

# Each label line is a list of (text, bold) runs; bold runs are the strings as written on the board.
CONTEXT = [
    ("ctx_1652", "IMG_1652.jpg", (175, 268, 2440, 1565),
     [[("15K /1 nải", 1), (": 15,000 per bunch", 0)], [("2 Trái - 15K", 1), (": 15,000 for two", 0)]]),
    ("ctx_1492", "IMG_1492.jpg", (15, 1118, 420, 1345),
     [[("15K", 1), (" in pen over 13K: now 15,000", 0)], [("15K", 1), (" printed, stale: now 17,000", 0)]]),
    ("ctx_2022", "IMG_2022.jpg", (60, 2075, 1700, 2615),
     [[("ĐỒNG GIÁ 15.000Đ", 1), (" in the header:", 0)], [("15,000 for each of 10 toppings", 0)]]),
]
OTHER = [
    ("oth_2066", "IMG_2066.jpg", (835, 25, 1150, 220),  # badge only; the redacted hotline lies below y = 240
     [[("399 NGÀN", 1)], [("from 399,000", 0)]]),
    ("oth_1586", "IMG_1586.jpg", (3438, 978, 3588, 1006),  # the unit note line only, so that it can be read
     [[("Đơn vị tính: nghìn đồng", 1)], [("unit note: a bare ", 0), ("70", 1), (" is 70,000", 0)]]),
    ("oth_1575", "IMG_1575.jpg", (552, 128, 779, 232),
     [[("1tr9", 1), (" = 1,900,000", 0)], [("2tr2", 1), (" = 2,200,000", 0)]]),
    ("oth_2063a", "IMG_2063.jpg", (128, 378, 478, 494),  # two panels of one banner, cropped apart (it is tilted)
     [[("5.999K", 1), (" = 5,999,000", 0)], [("dot separates thousands", 0)]]),
    ("oth_2063b", "IMG_2063.jpg", (655, 342, 998, 468),
     [[("559K", 1), (" = 559,000", 0)], [("on the same banner", 0)]]),
]


def crop(key, fn, box):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{key}.jpg"
    if (IMG_DIR / fn).exists():
        im = ImageOps.exif_transpose(Image.open(IMG_DIR / fn)).convert("RGB").crop(box)
        im.thumbnail((900, 900), Image.LANCZOS)  # the figure needs < 600 px per crop
        im.save(path, quality=92)
    elif not path.exists():
        raise FileNotFoundError(f"{IMG_DIR / fn} (set SMV_IMAGES to the redacted image folder)")
    return Image.open(path).convert("RGB")


# ----------------------------------------------------------------------------- layout: justified rows
# Every string is set in the body serif (bold = the string as written on the board) and placed on a common baseline;
# every label starts at the left edge of its crop, and the script stops if a label would run into the next one.
W, GAP, DPI = TEXTWIDTH, 0.055, 300
FS_HEAD, FS_LAB = 7.0, 6.5
PT = 1 / 72
LINE = FS_LAB * 1.22 * PT                         # baseline-to-baseline distance of label lines
ASC_HEAD, ASC_LAB, DESC = 0.95 * FS_HEAD * PT, 0.95 * FS_LAB * PT, 0.25 * FS_LAB * PT  # room for stacked marks
HEAD, PAD_LAB, PAD_ROW = 0.16, 0.03, 0.075


def row_height(ims):
    return (W - GAP * (len(ims) - 1)) / sum(im.width / im.height for im in ims)


rows = [
    ("a", "One amount, eight spellings: each crop is a gold offer of 15,000 VND",
     [crop(k, f, b) for k, f, b, _ in TOKENS], [[[(raw, 1)], [(note, 0)]] for (_, _, _, raw), note in
                                                  zip(TOKENS, TOKEN_NOTE)]),
    ("b", "The same 15K, three readings", [crop(k, f, b) for k, f, b, _ in CONTEXT], [lab for *_, lab in CONTEXT]),
    ("c", "Words for thousand and million, plus a dot as thousands separator",
     [crop(k, f, b) for k, f, b, _ in OTHER], [lab for *_, lab in OTHER]),
]


def block(ims, labs):
    return HEAD + row_height(ims) + PAD_LAB + ASC_LAB + LINE * (max(len(lab) for lab in labs) - 1) + DESC


height = sum(block(ims, labs) for _, _, ims, labs in rows) + PAD_ROW * (len(rows) - 1)
fig = plt.figure(figsize=(W, height))
renderer = fig.canvas.get_renderer()


def text(x, y, s, bold, size):
    return fig.text(x / W, 1 - y / height, s, fontsize=size, family=SERIF, fontweight="bold" if bold else "normal",
                    color=INK if bold else INK2, ha="left", va="baseline")


def width(parts, size=FS_LAB):
    total = 0.0
    for s, bold in parts:
        t = text(0, 0, s, bold, size)
        total += t.get_window_extent(renderer).width / fig.dpi
        t.remove()
    return total


def runs(x, y, parts, size=FS_LAB):
    """Draw text runs (text, bold) from x inches on the baseline y inches below the top of the figure."""
    for s, bold in parts:
        x += text(x, y, s, bold, size).get_window_extent(renderer).width / fig.dpi


y = 0.0
for letter, title, ims, labs in rows:
    runs(0, y + ASC_HEAD, [(f"({letter}) ", 1), (title, 0)], size=FS_HEAD)
    h = row_height(ims)
    top = y + HEAD
    x = 0.0
    for i, (im, lab) in enumerate(zip(ims, labs)):
        w = h * im.width / im.height
        room = w + GAP - 0.045 if i < len(ims) - 1 else w  # a visible space before the next label; the last ends at W
        need = max(width(parts) for parts in lab)
        assert need <= room, f"row ({letter}), crop {i + 1}: label needs {need:.3f} in, room is {room:.3f} in"
        px = (max(2, round(w * DPI)), max(2, round(h * DPI)))
        arr = np.asarray(im.resize(px, Image.LANCZOS))
        ax = fig.add_axes([x / W, 1 - (top + h) / height, w / W, h / height])
        ax.imshow(arr, interpolation="none", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(MUTED)
            sp.set_linewidth(0.4)
        for j, parts in enumerate(lab):
            runs(x, top + h + PAD_LAB + ASC_LAB + j * LINE, parts)
        x += w + GAP
    y += block(ims, labs) + PAD_ROW

save(fig, "fig_prices")
print(f"price examples: {N.items}")
