"""Synthetic Vietnamese menus (PIL) + matching v3 labels, for tests and smoke runs.

Layout mirrors the web tool: <out>/images/<name>.<ext> and <out>/images/_labels/<annotator>/<name>.<ext>.json.
Covers: 25k / 25.000đ / bare "25", M/L columns, section price, crossed price, nóng/đá, nóng|đá, range, delta,
no prices, unlabeled tiers, description lines and item codes, quantity tiers, unit prices, combo, two columns,
empty menu, duplicate item. Also HEIC/WEBP/PNG files, an EXIF-rotated JPEG, a draft label, a second annotator,
an orphan image and an orphan label."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FONT_CANDIDATES = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]
BOLD_CANDIDATES = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] + FONT_CANDIDATES


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    for p in (BOLD_CANDIDATES if bold else FONT_CANDIDATES):
        if Path(p).is_file():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def O(price: int | None, raw: str, **kw) -> dict[str, Any]:  # noqa: E743 - short constructor for offers
    d = {"size": "", "size_rank": None, "temp": "", "portion": "", "quantity": None, "qty_unit": "", "raw_price": raw,
         "price_vnd": price, "old_price_vnd": None, "price_kind": "fixed", "price_min_vnd": None, "price_max_vnd": None,
         "unit_qty": None, "unit": "", "price_source": "item_line", "evidence": "printed", "notes": ""}
    d.update(kw)
    return d


@dataclass
class It:
    line: str                       # text drawn on the menu
    raw_name: str
    norm_name: str
    offers: list[dict]
    sub: list[str] = field(default_factory=list)   # smaller lines under the item (description, components)
    strike: str | None = None       # substring of `line` drawn crossed out
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sec:
    title: str | None
    items: list[It]
    norm_title: str = ""
    shared_price_raw: str = ""


@dataclass
class Menu:
    name: str
    ext: str
    title: str
    sections: list[Sec]
    meta: dict[str, Any]
    footer: list[str] = field(default_factory=list)
    columns: int = 1


def menus() -> list[Menu]:
    m: list[Menu] = []
    base = {"venue_tier": "small_shop", "venue_type": "cà phê", "medium": "in sẵn để bàn", "lighting": "đủ sáng",
            "angle": "thẳng", "source": "tự chụp", "languages": ["vi"], "price_unit_default": "k_vnd",
            "tax_note": "not_stated", "has_prices": True}
    m.append(Menu("s01_k_suffix", "jpg", "QUÁN CHÈ CÔ BA", [Sec("Chè", [
        It("Chè đậu xanh .......... 15k", "Chè đậu xanh", "chè đậu xanh", [O(15000, "15k")]),
        It("Chè bưởi .......... 20k", "Chè bưởi", "chè bưởi", [O(20000, "20k")]),
        It("Chè thái sầu riêng .......... 25k", "Chè thái sầu riêng", "chè thái sầu riêng", [O(25000, "25k")]),
        It("Rau câu dừa .......... 12K", "Rau câu dừa", "rau câu dừa", [O(12000, "12K")]),
    ], "chè")], {**base, "ccl": 1, "tags": ["P_FORMAT"], "venue_type": "tiệm bánh – tráng miệng"}))
    m.append(Menu("s02_full_dots", "jpg", "CƠM TẤM SÀI GÒN", [Sec(None, [
        It("Cơm sườn  45.000đ", "Cơm sườn", "cơm sườn", [O(45000, "45.000đ")]),
        It("Cơm sườn bì chả  55.000đ", "Cơm sườn bì chả", "cơm sườn bì chả", [O(55000, "55.000đ")]),
        It("Cơm gà nướng  50.000 đ", "Cơm gà nướng", "cơm gà nướng", [O(50000, "50.000 đ")]),
        It("Canh chua  15.000đ", "Canh chua", "canh chua", [O(15000, "15.000đ")]),
    ])], {**base, "ccl": 1, "tags": ["P_FORMAT"], "venue_type": "cơm", "price_unit_default": "vnd"}))
    m.append(Menu("s03_implicit_k", "png", "NƯỚC ÉP - SINH TỐ", [Sec(None, [
        It("Nước ép cam      30", "Nước ép cam", "nước ép cam", [O(30000, "30")]),
        It("Nước ép thơm      25", "Nước ép thơm", "nước ép dứa", [O(25000, "25")], extra={"alt_norm_names": ["nước ép thơm"]}),
        It("Sinh tố bơ      35", "Sinh tố bơ", "sinh tố bơ", [O(35000, "35")]),
        It("Sinh tố sapoche      35", "Sinh tố sapoche", "sinh tố sa-pô-chê", [O(35000, "35")],
           extra={"alt_norm_names": ["sinh tố sapoche"]}),
    ])], {**base, "ccl": 2, "tags": ["P_IMPLICIT_K", "N_NORMALIZE"], "venue_type": "nước ép – sinh tố"}))
    m.append(Menu("s04_size_columns", "jpg", "TRÀ SỮA  (Size:  M     L)", [Sec(None, [
        It("Trà sữa trân châu      25    30", "Trà sữa trân châu", "trà sữa trân châu",
           [O(25000, "25", size="M", size_rank=1, price_source="column_header"),
            O(30000, "30", size="L", size_rank=2, price_source="column_header")]),
        It("Trà sữa matcha      30    35", "Trà sữa matcha", "trà sữa matcha",
           [O(30000, "30", size="M", size_rank=1, price_source="column_header"),
            O(35000, "35", size="L", size_rank=2, price_source="column_header")]),
        It("Hồng trà sữa      25    32", "Hồng trà sữa", "hồng trà sữa",
           [O(25000, "25", size="M", size_rank=1, price_source="column_header"),
            O(32000, "32", size="L", size_rank=2, price_source="column_header")]),
    ])], {**base, "ccl": 3, "tags": ["P_IMPLICIT_K", "P_INHERITED", "L_VARIANT"], "venue_type": "trà sữa"}))
    m.append(Menu("s05_section_price", "jpg", "TÀU HŨ - SINH TỐ", [
        Sec("SINH TỐ (ĐỒNG GIÁ 30K)", [
            It("Dâu", "Dâu", "sinh tố dâu", [O(30000, "30K (tiêu đề)", price_source="section_header")]),
            It("Mãng cầu", "Mãng cầu", "sinh tố mãng cầu", [O(30000, "30K (tiêu đề)", price_source="section_header")]),
            It("Xoài", "Xoài", "sinh tố xoài", [O(30000, "30K (tiêu đề)", price_source="section_header")]),
        ], "sinh tố", "ĐỒNG GIÁ 30K"),
        Sec("TÀU HŨ", [
            It("Tàu hũ nước đường .... 12k", "Tàu hũ nước đường", "tàu hũ nước đường", [O(12000, "12k")]),
            It("Tàu hủ trân châu .... 18k", "Tàu hủ trân châu", "tàu hũ trân châu", [O(18000, "18k")]),
        ], "tàu hũ")],
        {**base, "ccl": 3, "tags": ["P_INHERITED", "P_FORMAT", "N_SECTION_COMPOSE", "N_NORMALIZE"],
         "venue_type": "nước ép – sinh tố"}))
    m.append(Menu("s06_crossed_price", "webp", "BÁNH MÌ HUỲNH", [Sec(None, [
        It("Bánh mì thịt   30k  25k", "Bánh mì thịt", "bánh mì thịt", [O(25000, "25k", old_price_vnd=30000)], strike="30k"),
        It("Bánh mì chả cá   25k", "Bánh mì chả cá", "bánh mì chả cá", [O(25000, "25k")]),
        It("Bánh mì ốp la   20k", "Bánh mì ốp la", "bánh mì ốp la", [O(20000, "20k")]),
    ])], {**base, "ccl": 3, "tags": ["P_PROMO", "P_FORMAT"], "venue_type": "bánh mì – xôi"}))
    m.append(Menu("s07_hot_iced", "jpg", "CÀ PHÊ", [Sec(None, [
        It("Cà phê đen nóng/đá   20 / 25", "Cà phê đen nóng/đá", "cà phê đen",
           [O(20000, "20 / 25", temp="nóng"), O(25000, "20 / 25", temp="đá")]),
        It("Bạc xỉu nóng/đá   30", "Bạc xỉu nóng/đá", "bạc xỉu", [O(30000, "30", temp="nóng|đá")]),
        It("Cf sữa đá   28", "Cf sữa đá", "cà phê sữa đá", [O(28000, "28")]),
    ])], {**base, "ccl": 3, "tags": ["P_IMPLICIT_K", "L_VARIANT", "N_NORMALIZE"]}))
    m.append(Menu("s08_range_delta", "jpg", "QUÁN NHẬU BA KHÍA", [
        Sec("MÓN XÀO", [
            It("Mì xào hải sản   45k-55k", "Mì xào hải sản", "mì xào hải sản",
               [O(45000, "45k-55k", price_kind="range", price_min_vnd=45000, price_max_vnd=55000)]),
            It("Rau muống xào tỏi   35k", "Rau muống xào tỏi", "rau muống xào tỏi", [O(35000, "35k")]),
        ], "món xào"),
        Sec("THÊM", [
            It("Thêm trứng   +5k", "Thêm trứng", "thêm trứng", [O(5000, "+5k", price_kind="delta")],
               extra={"is_add_on": True}),
            It("Thêm mì   +10k", "Thêm mì", "thêm mì", [O(10000, "+10k", price_kind="delta")], extra={"is_add_on": True}),
        ], "món thêm")],
        {**base, "ccl": 3, "tags": ["P_MULTI_UNLABELED", "L_ADDON", "P_FORMAT"], "venue_type": "quán nhậu – ốc"}))
    m.append(Menu("s09_no_prices", "heic", "GỎI ĐU ĐỦ - BÁNH TRÁNG", [Sec(None, [
        It("Gỏi đu đủ khô bò", "Gỏi đu đủ khô bò", "gỏi đu đủ khô bò", []),
        It("Bánh tráng trộn", "Bánh tráng trộn", "bánh tráng trộn", []),
        It("Bánh tráng nướng", "Bánh tráng nướng", "bánh tráng nướng", []),
    ])], {**base, "ccl": 1, "tags": ["P_MISSING"], "has_prices": False, "venue_type": "ăn vặt",
          "venue_tier": "street_vendor"}))
    m.append(Menu("s10_unlabeled_tiers", "jpg", "XÔI - BÁNH MÌ", [Sec(None, [
        It("Xôi gà xé   20k 25k 30k", "Xôi gà xé", "xôi gà xé",
           [O(20000, "20k 25k 30k", size="#1", size_rank=1), O(25000, "20k 25k 30k", size="#2", size_rank=2),
            O(30000, "20k 25k 30k", size="#3", size_rank=3)],
           extra={"ambiguous": True, "ambig_reason": "variant_label_unknown"}),
        It("Bánh mì ốp la   18k 20k", "Bánh mì ốp la", "bánh mì ốp la",
           [O(18000, "18k 20k", size="#1", size_rank=1), O(20000, "18k 20k", size="#2", size_rank=2)],
           extra={"ambiguous": True, "ambig_reason": "variant_label_unknown"}),
        It("Bún thịt nướng   30k", "Bún thịt nướng", "bún thịt nướng", [O(30000, "30k")]),
    ])], {**base, "ccl": 3, "tags": ["P_MULTI_UNLABELED", "L_VARIANT"], "venue_type": "bánh mì – xôi",
          "venue_tier": "street_vendor"}))
    m.append(Menu("s11_description", "jpg", "BÚN BÒ HUẾ O HAI", [Sec(None, [
        It("A01  Bún bò đặc biệt ......... 60.000", "Bún bò đặc biệt", "bún bò đặc biệt", [O(60000, "60.000")],
           sub=["(Bò, giò, chả cua, huyết)"], extra={"item_code": "A01", "description": "(Bò, giò, chả cua, huyết)"}),
        It("A02  Bún bò thường ......... 45.000", "Bún bò thường", "bún bò thường", [O(45000, "45.000")],
           extra={"item_code": "A02"}),
        It("A03  Bún giò ......... 50.000", "Bún giò", "bún giò", [O(50000, "50.000")], extra={"item_code": "A03"}),
    ])], {**base, "ccl": 2, "tags": ["D_DISTRACTOR", "P_FORMAT"], "venue_type": "bún – phở – hủ tiếu",
          "price_unit_default": "vnd"}, footer=["ĐT: 0909 123 456   Wi-Fi: ohai2024"]))
    m.append(Menu("s12_quantity_unit", "jpg", "GÀ RÁN - ĐỒ NƯỚNG", [Sec(None, [
        It("Gà rán   1 miếng 35.000  /  2 miếng 68.000", "Gà rán", "gà rán",
           [O(35000, "35.000", quantity=1, qty_unit="miếng"), O(68000, "68.000", quantity=2, qty_unit="miếng")]),
        It("Bò nướng lá lốt   50k/phần", "Bò nướng lá lốt", "bò nướng lá lốt",
           [O(50000, "50k/phần", price_kind="unit_price", unit="phần", unit_qty=1)]),
        It("Ốc hương   120k/kg", "Ốc hương", "ốc hương", [O(120000, "120k/kg", price_kind="unit_price", unit="kg", unit_qty=1)]),
    ])], {**base, "ccl": 3, "tags": ["L_VARIANT", "P_UNIT_BUNDLE", "P_FORMAT"], "venue_type": "lẩu – nướng"}))
    m.append(Menu("s13_combo", "jpg", "COMBO TRƯA", [Sec(None, [
        It("Combo 2 người ............ 199k", "Combo 2 người", "combo 2 người", [O(199000, "199k")],
           sub=["- 2 cơm gà", "- 2 canh rong biển", "- 2 trà đá"],
           extra={"is_combo": True, "combo_components": [{"name": "cơm gà", "qty": "2", "ref_price_vnd": None},
                                                          {"name": "canh rong biển", "qty": "2", "ref_price_vnd": None},
                                                          {"name": "trà đá", "qty": "2", "ref_price_vnd": None}]}),
        It("Cơm gà xối mỡ ............ 55k", "Cơm gà xối mỡ", "cơm gà xối mỡ", [O(55000, "55k")]),
    ])], {**base, "ccl": 3, "tags": ["L_COMBO_SET", "P_FORMAT"], "venue_type": "cơm"}))
    drinks = [("Trà đào", 25), ("Trà vải", 25), ("Trà chanh", 15), ("Trà tắc", 15), ("Nước cam", 30), ("Nước chanh", 15),
              ("Soda việt quất", 30), ("Soda bạc hà", 28), ("Sữa chua đá", 20), ("Sữa chua dâu", 25), ("Cacao đá", 25),
              ("Matcha đá xay", 40), ("Cookie đá xay", 40), ("Trà xanh", 10), ("Nước suối", 10), ("Sting", 15)]
    m.append(Menu("s14_two_columns", "jpg", "MENU ĐỒ UỐNG", [Sec(None, [
        It(f"{n}   {p}k", n, n.lower(), [O(p * 1000, f"{p}k")]) for n, p in drinks])],
        {**base, "ccl": 4, "tags": ["S_LAYOUT", "P_FORMAT"], "venue_type": "nước giải khát" if False else "khác"},
        columns=2))
    m.append(Menu("s15_empty", "jpg", "QUÁN ĂN GIA ĐÌNH", [Sec(None, [])],
                  {**base, "ccl": 1, "tags": ["P_MISSING"], "has_prices": False, "venue_type": "nhà hàng"},
                  footer=["Mở cửa 6h - 22h", "ĐT: 0283 999 888"]))
    m.append(Menu("s16_duplicate_item", "jpg", "TIỆM BÁNH NGỌT", [
        Sec("BÁNH", [It("Bánh flan ....... 15k", "Bánh flan", "bánh flan", [O(15000, "15k")]),
                     It("Bánh bông lan ....... 25k", "Bánh bông lan", "bánh bông lan", [O(25000, "25k")])], "bánh"),
        Sec("TRÁNG MIỆNG", [It("Bánh flan ....... 15k", "Bánh flan", "bánh flan", [O(15000, "15k")]),
                            It("Sữa chua nếp cẩm ....... 20k", "Sữa chua nếp cẩm", "sữa chua nếp cẩm", [O(20000, "20k")])],
            "tráng miệng")],
        {**base, "ccl": 3, "tags": ["L_ITEM_SPLIT", "P_FORMAT"], "venue_type": "tiệm bánh – tráng miệng"}))
    return m


def render(menu: Menu, path: Path, exif_rotate: bool = False) -> None:
    from PIL import Image, ImageDraw

    W = 1400 if menu.columns == 1 else 1800
    f_title, f_sec, f_item, f_sub = _font(56, True), _font(40, True), _font(38), _font(28)
    lines: list[tuple[str, Any, int, str | None, int]] = []   # text, font, x, strike, col
    for sec in menu.sections:
        if sec.title:
            lines.append((sec.title, f_sec, 60, None, 0))
        for k, it in enumerate(sec.items):
            col = (k % 2) if menu.columns == 2 else 0
            lines.append((it.line, f_item, 90, it.strike, col))
            for s in it.sub:
                lines.append((s, f_sub, 130, None, col))
    rows = [ln for ln in lines if ln[4] == 0]
    H = 220 + 62 * max(len(rows), 1) + 70 * len(menu.footer) + 60
    img = Image.new("RGB", (W, H), (252, 250, 244))
    d = ImageDraw.Draw(img)
    d.text((60, 50), menu.title, font=f_title, fill=(150, 20, 20))
    y = {0: 170, 1: 170}
    for text, font, x, strike, col in lines:
        xx = x + (W // 2 if col == 1 else 0)
        d.text((xx, y[col]), text, font=font, fill=(20, 20, 20))
        if strike and strike in text:
            pre = d.textlength(text[: text.index(strike)], font=font)
            wid = d.textlength(strike, font=font)
            yy = y[col] + font.size * 0.6
            d.line((xx + pre - 4, yy, xx + pre + wid + 4, yy), fill=(200, 0, 0), width=4)
        y[col] += 62 if font is not f_sub else 44
        if menu.columns == 1:
            y[1] = y[0]
    yf = max(y.values()) + 30
    for t in menu.footer:
        d.text((60, yf), t, font=f_sub, fill=(90, 90, 90))
        yf += 50
    path.parent.mkdir(parents=True, exist_ok=True)
    if menu.ext == "heic":
        import pillow_heif

        pillow_heif.register_heif_opener()
        img.save(path, format="HEIF", quality=90)
    elif menu.ext == "webp":
        img.save(path, "WEBP", quality=90)
    elif menu.ext == "png":
        img.save(path, "PNG")
    elif exif_rotate:
        # store rotated pixels + EXIF orientation 6 (viewer must rotate 90° CW to display upright)
        rot = img.transpose(Image.ROTATE_90)
        exif = Image.Exif()
        exif[0x0112] = 6
        rot.save(path, "JPEG", quality=92, exif=exif.tobytes())
    else:
        img.save(path, "JPEG", quality=92)


def label_for(menu: Menu, file_name: str, sha: str, annotator: str = "synth", status: str = "done",
              split: str = "unassigned") -> dict[str, Any]:
    sections, items = [], []
    iid = 1
    for si, sec in enumerate(menu.sections):
        sid = f"s{si + 1}" if sec.title else None
        if sec.title:
            sections.append({"section_id": sid, "raw_title": sec.title, "norm_title": sec.norm_title or sec.title.lower(),
                             "shared_price_raw": sec.shared_price_raw, "notes": ""})
        for it in sec.items:
            item = {"item_id": iid, "section_id": sid, "item_code": "", "raw_name": it.raw_name, "raw_name_2": "",
                    "norm_name": it.norm_name, "alt_norm_names": [], "description": "", "category": "khác",
                    "is_add_on": False, "is_combo": False, "combo_components": [], "serves": "", "option_groups": "",
                    "attributes": [], "availability": "", "item_status": "active", "legibility": "clear",
                    "ambiguous": False, "ambig_reason": "", "notes": "", "offers": [dict(o) for o in it.offers]}
            item.update(it.extra)
            items.append(item)
            iid += 1
    meta = dict(menu.meta)
    return {"schema_version": "3.0", "image_id": f"smv_{Path(file_name).stem}", "file_name": file_name,
            "file_sha256": sha, "group_id": f"G_{menu.name[:3]}", "near_duplicate_of": "", "split": split,
            "ccl": meta.pop("ccl"), "tags": meta.pop("tags"), **meta, "city": "", "pii_present": False,
            "pii_types": [], "redaction_checked": False, "consent_id": "", "sections": sections, "items": items,
            "ignore_regions": [], "annotator": annotator, "status": status, "version": 1, "notes": "synthetic"}


def generate(out: Path, with_edge_cases: bool = True) -> dict[str, Any]:
    """Write images + labels. Returns a summary dict."""
    img_dir = out / "images"
    lab_dir = img_dir / "_labels"
    written = []
    for k, menu in enumerate(menus()):
        fn = f"{menu.name}.{menu.ext}"
        p = img_dir / fn
        render(menu, p, exif_rotate=(menu.name == "s12_quantity_unit"))
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        split = "dev" if k % 4 == 0 else "unassigned"
        lab = label_for(menu, fn, sha if k % 3 else "", split=split)
        (lab_dir / "synth").mkdir(parents=True, exist_ok=True)
        (lab_dir / "synth" / f"{fn}.json").write_text(json.dumps(lab, ensure_ascii=False, indent=1), encoding="utf-8")
        written.append(fn)
        if with_edge_cases and menu.name in ("s01_k_suffix", "s04_size_columns", "s07_hot_iced", "s10_unlabeled_tiers"):
            lab2 = label_for(menu, fn, sha, annotator="synth2", split=split)
            if menu.name == "s04_size_columns":   # disagreement: one price differs, one tag missing
                lab2["items"][0]["offers"][1]["price_vnd"] = 31000
                lab2["tags"] = lab2["tags"][:-1]
                lab2["ccl"] = 2
            (lab_dir / "synth2").mkdir(parents=True, exist_ok=True)
            (lab_dir / "synth2" / f"{fn}.json").write_text(json.dumps(lab2, ensure_ascii=False, indent=1), encoding="utf-8")
    if with_edge_cases:
        # draft label from a third annotator (must not become gold)
        menu = menus()[1]
        d = label_for(menu, f"{menu.name}.{menu.ext}", "", annotator="draft_user", status="draft")
        (lab_dir / "draft_user").mkdir(parents=True, exist_ok=True)
        (lab_dir / "draft_user" / f"{menu.name}.{menu.ext}.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        # orphan label (no image) and orphan image (no label)
        o = label_for(menus()[0], "missing_image.jpg", "", annotator="synth")
        (lab_dir / "synth" / "missing_image.jpg.json").write_text(json.dumps(o, ensure_ascii=False), encoding="utf-8")
        render(Menu("orphan", "jpg", "ẢNH KHÔNG CÓ NHÃN", [Sec(None, [It("Món lạ 10k", "Món lạ", "món lạ", [O(10000, "10k")])])],
                    {}), img_dir / "sub" / "orphan_no_label.jpg")
        # a broken label file
        (lab_dir / "synth" / "broken.jpg.json").write_text("{not json", encoding="utf-8")
    return {"images": written, "dir": str(img_dir)}


def mock_gold_map(st, out_path: Path) -> int:
    """sha256 -> perfect prediction, for the mock server."""
    from smv.data.index import load_golds, load_index
    from smv.schema.prediction import annotation_to_prediction

    idx = load_index(st)
    golds = load_golds(st)
    mp = {}
    for r in idx.itertuples(index=False):
        g = golds.get(r.image_id)
        if g is not None:
            mp[r.sha256] = json.loads(annotation_to_prediction(g).model_dump_json())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mp, ensure_ascii=False), encoding="utf-8")
    return len(mp)
