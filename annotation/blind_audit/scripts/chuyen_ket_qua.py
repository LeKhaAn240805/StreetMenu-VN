#!/usr/bin/env python3
"""DÀNH CHO NGƯỜI QUẢN LÝ (An), người gán không cần chạy file này.

Chuyển nhãn gán mù từ web VietMenu Labeler sang đúng định dạng để chạy `smv agreement`:
    outputs/data/agreement_sets/<tên người gán>/<image_id>.json

Cách dùng (Python 3.9+, chỉ thư viện chuẩn):
    python chuyen_ket_qua.py <thư mục _labels/<Tên> hoặc file .jsonl xuất từ web> <tên người gán> [thư mục ra]
    ví dụ: python chuyen_ket_qua.py GanMu_Long/anh/_labels/B B outputs/data/agreement_sets

Script gán image_id, group_id, split = test theo bộ dữ liệu cuối, rồi áp ĐÚNG các bước chuẩn hoá tất định đã áp
cho nhãn gold (NFC, giá thời giá = market, khoảng giá lấy min, món không đọc được = mơ hồ, giá sửa tay = overwritten,
tính lại 8 thẻ suy từ trường và CCL theo luật v3.1). Không sửa nội dung người gán đã nhập ngoài các bước này.
"""
import json, re, sys, unicodedata
from collections import Counter
from pathlib import Path

MAPPING = {"IMG_1706.jpg": {"image_id": "smv_0175", "file_sha256": "f5381a34a3dfc956ebac124da1bc4efa98427b9d9704c56230d4a4f8ad44296d", "group_id": "G_1706", "image_width": 255, "image_height": 347}, "IMG_2083.jpg": {"image_id": "smv_0252", "file_sha256": "4fd34e78c28ed308b0c3b82710a468da31f353f5a2670a0522526337b4032541", "group_id": "G_2083", "image_width": 690, "image_height": 850}, "4.jpg": {"image_id": "smv_0020", "file_sha256": "92f0d341d36c48d54eabd4650e60619b5fe0216fc9b703b244115d767247e607", "group_id": "G_4", "image_width": 1125, "image_height": 2000}, "IMG_7869.jpg": {"image_id": "smv_0349", "file_sha256": "4ce2e5595701efca8077be67ed24c9a32cc03c7f5113ac9f53a39b77fbd5dbc5", "group_id": "G_7869", "image_width": 989, "image_height": 1755}, "IMG_2275.jpg": {"image_id": "smv_0264", "file_sha256": "e2c236bb8d3b49c16146eeda93294b360602418f32410241301bef7e81c6ae16", "group_id": "G_2275", "image_width": 1308, "image_height": 1561}, "IMG_1672.jpg": {"image_id": "smv_0164", "file_sha256": "49e6b772fa505b239e0ee8f9fd92de41e8f2d3201516fd833860db93384a8240", "group_id": "G_1672", "image_width": 2515, "image_height": 3696}, "IMG_1473.jpg": {"image_id": "smv_0071", "file_sha256": "e7b6df38c118953107c02ddf972500c410da1ae3227b19bcde9c6269ab7ae006", "group_id": "G_1470", "image_width": 3024, "image_height": 4032}, "13.jpg": {"image_id": "smv_0004", "file_sha256": "ce23c26d4404288d07dc3d86505122dd0371d8f455d3054567457e2fcd247fe8", "group_id": "G_13", "image_width": 1500, "image_height": 2000}, "IMG_1983.jpg": {"image_id": "smv_0230", "file_sha256": "c5faf5b8a97027c9875fda91bdfdb73988087430a0fd23675048ca7c4801c4d7", "group_id": "G_1983", "image_width": 1221, "image_height": 869}, "IMG_1616.jpg": {"image_id": "smv_0139", "file_sha256": "d9132fe7aa178b735346031776a5ee90e84554aeac75ca5b122f227c25021fac", "group_id": "G_1616", "image_width": 1003, "image_height": 492}, "IMG_1397.jpg": {"image_id": "smv_0057", "file_sha256": "912283f633c7ae6fc89a5dee12c4ed213ad0a267ad17c880e3c1b3d839f95580", "group_id": "G_1397", "image_width": 1080, "image_height": 795}, "IMG_1503.jpg": {"image_id": "smv_0093", "file_sha256": "957f9d995309610ca5c10fc29ba8c4e601fd2226ac4a4121eebc392fbc6b7c0b", "group_id": "G_1503", "image_width": 1208, "image_height": 1575}, "IMG_1948.jpg": {"image_id": "smv_0228", "file_sha256": "20b34dd88c59ee7e440cb3f70e33e29959778b1e97103b62a42bdb566948e78d", "group_id": "G_1948", "image_width": 861, "image_height": 1697}, "IMG_1656.jpg": {"image_id": "smv_0159", "file_sha256": "bdc3769e98c5ef60ccf828203bacf0320e615e20f0f4a285b4c418af16068ef0", "group_id": "G_1656", "image_width": 771, "image_height": 966}, "IMG_1582.jpg": {"image_id": "smv_0118", "file_sha256": "13519d28904b8be407de4388842ad513b45c423cc7b6794acbb3c68020af0917", "group_id": "G_1582", "image_width": 1431, "image_height": 638}, "IMG_2381.jpg": {"image_id": "smv_0312", "file_sha256": "5ad9d8bc547e2eb28ff65b8beba02cd74c2b7e78145c6b3677a843c48a7e743a", "group_id": "G_2381", "image_width": 1052, "image_height": 723}, "IMG_1378.jpg": {"image_id": "smv_0054", "file_sha256": "cee66cd93ab0a0ad782472d71eda123ab21c618a4dac93c5fb75dab606051913", "group_id": "G_1378", "image_width": 452, "image_height": 403}, "IMG_1686.jpg": {"image_id": "smv_0173", "file_sha256": "ac635a8a9ca3af89905baacdd7d16e953dcd070e2e6f7b7ca4b2da3c77418e11", "group_id": "G_1686", "image_width": 715, "image_height": 1383}, "IMG_1546.jpg": {"image_id": "smv_0105", "file_sha256": "934a4e04f9c21344393355fa629557c08b41fe9d4153fc2ce0270042834c19bf", "group_id": "G_1546", "image_width": 673, "image_height": 553}, "IMG_1917.jpg": {"image_id": "smv_0222", "file_sha256": "c4fe637ccf30d82dba7e3e608e4dce72fa8da6d2dfbec390880a173b21a03aa0", "group_id": "G_1917", "image_width": 454, "image_height": 322}, "IMG_1454.jpg": {"image_id": "smv_0062", "file_sha256": "8cd9a9b04ad5dde7ad5e81741c8bcc89ef00bf0730762a3ce8aa9c86ae49a1fb", "group_id": "G_1454", "image_width": 1135, "image_height": 1461}, "IMG_1943.jpg": {"image_id": "smv_0226", "file_sha256": "789ff7884c5a12674554daac0fd8466deec701e8ad81dd5d9942e2ac099ab5fa", "group_id": "G_1943", "image_width": 896, "image_height": 1528}, "IMG_1885.jpg": {"image_id": "smv_0213", "file_sha256": "9ccdba5af99882a3f6d2838f178a4ab1890af694727b841db4959b8684e0930a", "group_id": "G_1885", "image_width": 360, "image_height": 404}, "IMG_2318.jpg": {"image_id": "smv_0293", "file_sha256": "fb8c9339f37d103aa35d997661937aea26d3d2b9cdc77333d3338621eca4fcdb", "group_id": "G_2318", "image_width": 825, "image_height": 1108}, "IMG_1834.jpg": {"image_id": "smv_0205", "file_sha256": "633040344e89550d786ebf151088d913e155cd85081ef54b9d192b64d9d3c543", "group_id": "G_1834", "image_width": 577, "image_height": 754}, "IMG_2300.jpg": {"image_id": "smv_0280", "file_sha256": "197e2c7b0ce275149a1934f29d70cbd06b490bc9aff5ed407a751d3946510d8c", "group_id": "G_2300", "image_width": 954, "image_height": 716}, "IMG_2418.jpg": {"image_id": "smv_0320", "file_sha256": "74489b297ab884a7f4f88441c6d65bf3faa2f546c314b338217046ee92ef38cf", "group_id": "G_2418", "image_width": 1111, "image_height": 457}, "IMG_2372.jpg": {"image_id": "smv_0305", "file_sha256": "82caf82c5987d0ed8fc9f889f6066842aec105104650c7a9be12e581b7ad372d", "group_id": "G_2372", "image_width": 2568, "image_height": 1926}, "IMG_2424.jpg": {"image_id": "smv_0325", "file_sha256": "43ea6197e24c57beb9047765bd3aff58673fc688ed1ea2297b9e32b734cfac3e", "group_id": "G_2424", "image_width": 1391, "image_height": 1912}, "IMG_2304.jpg": {"image_id": "smv_0284", "file_sha256": "c7cc9db1ee64ffa87d1a06dafb495faa6c9cc792f86c80d6d005872dd140d484", "group_id": "G_2304", "image_width": 1499, "image_height": 1133}}
TAG_ORDER = ["P_IMPLICIT_K", "P_FORMAT", "P_INHERITED", "P_MULTI_UNLABELED", "P_UNIT_BUNDLE", "P_PROMO", "P_HANDWRITTEN_OVERRIDE", "P_MISSING", "N_NORMALIZE", "N_SECTION_COMPOSE", "N_FOREIGN", "L_VARIANT", "L_COMBO_SET", "L_ADDON", "L_ITEM_SPLIT", "S_DETACHED", "S_LAYOUT", "V_HANDWRITING", "V_DAMAGE", "V_QUALITY", "V_GEOMETRY", "D_DISTRACTOR"]
ITEM_DEFAULTS = {"item_code": "", "raw_name_2": "", "alt_norm_names": [], "description": "", "category": "", "is_add_on": False,
                 "is_combo": False, "combo_components": [], "serves": "", "option_groups": "", "attributes": [],
                 "availability": "", "item_status": "active", "legibility": "clear", "ambiguous": False, "ambig_reason": "", "notes": ""}
OFFER_DEFAULTS = {"size": "", "size_rank": None, "temp": "", "portion": "", "quantity": None, "qty_unit": "", "raw_price": "",
                  "raw_old_price": "", "price_vnd": None, "old_price_vnd": None, "price_kind": "fixed", "price_min_vnd": None,
                  "price_max_vnd": None, "unit_qty": None, "unit": "", "price_source": "item_line", "evidence": "printed",
                  "notes": "", "ambiguous": False, "ambig_reason": ""}

INHERIT_SRC = {'section_header', 'global_header', 'column_header', 'bracket', 'merged_cell'}
FIELD_TAGS = {'L_VARIANT', 'P_MULTI_UNLABELED', 'P_UNIT_BUNDLE', 'P_PROMO', 'P_HANDWRITTEN_OVERRIDE',
              'P_INHERITED', 'P_MISSING', 'L_ADDON'}
MARKET_RE = re.compile(r'\btuy\s*(loai|gia|size|thoi\s*gia|can)\b|\bthoi\s*gia\b|\bgia\s*thi\s*truong\b')


def unaccent(s):
    s = unicodedata.normalize('NFD', s.lower()).replace('đ', 'd')
    return ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')

changes = []



def log(fn, iid, field, before, after, rule):
    changes.append(dict(file_name=fn, item_id='' if iid is None else iid, field=field,
                        before=json.dumps(before, ensure_ascii=False), after=json.dumps(after, ensure_ascii=False), rule=rule))


def setf(fn, obj, iid, field, val, rule):
    if obj.get(field) != val:
        log(fn, iid, field, obj.get(field), val, rule)
        obj[field] = val


def clean_str(s):
    s2 = unicodedata.normalize('NFC', s)
    s2 = re.sub(r'[ \t ]+', ' ', s2).strip()
    return s2


def clean_tree(fn, o, path=''):
    """NFC + whitespace on every string; returns number of strings changed."""
    n = 0
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, str):
                c = clean_str(v)
                if c != v:
                    o[k] = c; n += 1
            else:
                n += clean_tree(fn, v, path + '.' + k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            if isinstance(v, str):
                c = clean_str(v)
                if c != v:
                    o[i] = c; n += 1
            else:
                n += clean_tree(fn, v, path)
    return n



def norm_default(raw: str, prefix: str = "") -> str:
    s = unicodedata.normalize("NFC", raw).strip().rstrip(":").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[\d①-⑳]+[.)]\s*", "", s)
    if prefix and not s.startswith(prefix):
        s = f"{prefix} {s}"
    return s




def rules(a):
    fn = a['file_name']
    n = clean_tree(fn, a)
    if n:
        log(fn, None, '*strings*', n, 'NFC+trim', 'auto:string_normalize')
    for it in a['items']:
        iid = it['item_id']
        if not (it.get('norm_name') or '').strip():
            setf(fn, it, iid, 'norm_name', norm_default(it['raw_name']), 'auto:fill_empty_norm')
        illeg = it.get('legibility') == 'illegible' or '???' in (it.get('raw_name') or '')
        for o in it['offers']:
            raw = (o.get('raw_price') or '')
            # market price ("tùy loại", "thời giá") -> price_kind market, not ambiguous
            if o.get('price_vnd') is None and MARKET_RE.search(unaccent(raw)):
                setf(fn, o, iid, 'price_kind', 'market', 'auto:market_price')
                setf(fn, o, iid, 'ambiguous', False, 'auto:market_price')
                setf(fn, o, iid, 'ambig_reason', '', 'auto:market_price')
            # R4: illegible / ??? -> ambiguous (excluded from main P/R)
            if illeg and not o.get('ambiguous'):
                setf(fn, o, iid, 'ambiguous', True, 'R4:illegible_item')
                setf(fn, o, iid, 'ambig_reason', o.get('ambig_reason') or 'occluded_char', 'R4:illegible_item')
            if o.get('ambiguous') and not o.get('ambig_reason'):
                setf(fn, o, iid, 'ambig_reason', 'other', 'auto:ambig_reason_required')
            if not o.get('ambiguous') and o.get('ambig_reason'):
                setf(fn, o, iid, 'ambig_reason', '', 'auto:ambig_reason_only_when_ambiguous')
            k = o.get('price_kind') or 'fixed'
            if k == 'range':
                if o.get('price_min_vnd') is not None and o.get('price_vnd') != o['price_min_vnd']:
                    setf(fn, o, iid, 'price_vnd', o['price_min_vnd'], 'R1:range_price_is_min')
            else:
                for f in ('price_min_vnd', 'price_max_vnd'):
                    if o.get(f) is not None:
                        setf(fn, o, iid, f, None, 'auto:minmax_only_for_range')
            if k == 'unit_price' and not (o.get('unit') or '').strip():
                setf(fn, o, iid, 'price_kind', 'fixed', 'auto:unit_price_needs_unit')
            if not (o.get('unit') or '').strip() and o.get('unit_qty') is not None and o.get('price_kind') == 'fixed':
                setf(fn, o, iid, 'unit_qty', None, 'auto:unit_qty_needs_unit')
            # R5: a price changed by hand (incl. increase) is an override, not a printed promo
            if o.get('old_price_vnd') is not None and o.get('evidence') == 'handwritten':
                setf(fn, o, iid, 'evidence', 'overwritten', 'R5:handwritten_change_is_override')
        if illeg and not it.get('ambiguous'):
            setf(fn, it, iid, 'ambiguous', True, 'R4:illegible_item')
        if it['offers']:
            amb = any(o.get('ambiguous') for o in it['offers'])
            if bool(it.get('ambiguous')) != amb:
                setf(fn, it, iid, 'ambiguous', amb, 'auto:item_ambiguous_follows_offers')
            if amb and not it.get('ambig_reason'):
                r = next(o['ambig_reason'] for o in it['offers'] if o.get('ambiguous'))
                setf(fn, it, iid, 'ambig_reason', r, 'auto:item_ambiguous_follows_offers')
            if not amb and it.get('ambig_reason'):
                setf(fn, it, iid, 'ambig_reason', '', 'auto:item_ambiguous_follows_offers')
    # price_unit_default: bare 1-3 digit prices read as thousands -> k_vnd
    if a.get('price_unit_default') == 'vnd':
        if any(o.get('price_vnd') is not None and re.fullmatch(r'(\d{1,3})(\s*/.*)?', (o.get('raw_price') or '').strip())
               and o['price_vnd'] == int(re.match(r'\d+', o['raw_price'].strip()).group()) * 1000
               for it in a['items'] for o in it['offers']):
            setf(fn, a, None, 'price_unit_default', 'k_vnd', 'auto:bare_numbers_are_thousands')
    priced = sum(o.get('price_vnd') is not None for it in a['items'] for o in it['offers'])
    setf(fn, a, None, 'has_prices', priced > 0, 'auto:has_prices_from_offers')



def derive_field_tags(a):
    t = set()
    if a.get('split') != 'ood' and not a.get('has_prices'):
        t.add('P_MISSING')
    for it in a['items']:
        offs = it['offers']
        if it.get('is_add_on'):
            t.add('L_ADDON')
        if len(offs) >= 2:
            if all(re.fullmatch(r'#\d+', o.get('size') or '') for o in offs):
                t.add('P_MULTI_UNLABELED')
            else:
                t.add('L_VARIANT')
        if not offs and it.get('item_status', 'active') == 'active':
            t.add('P_MISSING')
        for o in offs:
            k = o.get('price_kind') or 'fixed'
            if k in ('unit_price', 'bundle', 'starting_from') or (o.get('quantity') or 0) > 1:
                t.add('P_UNIT_BUNDLE')
            if k == 'range':
                t.add('P_MULTI_UNLABELED')
            ov, v = o.get('old_price_vnd'), o.get('price_vnd')
            if ov is not None:
                if o.get('evidence') == 'printed' and v is not None and ov > v:
                    t.add('P_PROMO')
                else:
                    t.add('P_HANDWRITTEN_OVERRIDE')
            if o.get('evidence') == 'overwritten' or o.get('price_source') == 'handwritten_override':
                t.add('P_HANDWRITTEN_OVERRIDE')
            if o.get('price_source') in INHERIT_SRC:
                t.add('P_INHERITED')
            if v is None and k != 'market' and it.get('item_status', 'active') == 'active':
                t.add('P_MISSING')
    return t



# auto_tags from 03_build_dataset (copied logic, add-only part)
def m6_auto_tags(a):
    tags = set(a.get('tags', []))
    for it in a['items']:
        raw_def = norm_default(it['raw_name'])
        if it['norm_name'] != raw_def:
            if raw_def in it.get('alt_norm_names', []) and it['norm_name'].endswith(raw_def):
                tags.add('N_SECTION_COMPOSE')
        if it.get('raw_name_2'):
            tags.add('N_FOREIGN')
        if it.get('is_combo'):
            tags.add('L_COMBO_SET')
        for o in it['offers']:
            raw = o.get('raw_price') or ''
            if o.get('price_source') == 'badge':
                tags.add('S_DETACHED')
            if o.get('evidence') == 'handwritten':
                tags.add('V_HANDWRITING')
            if o.get('price_vnd') is not None and raw:
                if re.fullmatch(r'\d{1,3}', raw.strip()):
                    tags.add('P_IMPLICIT_K')
                elif re.search(r'[kK]|tr\d|\d[.,]\d{3}|VND|VNĐ|đ', raw):
                    tags.add('P_FORMAT')
    if any(x != 'vi' for x in a.get('languages', [])):
        tags.add('N_FOREIGN')
    return tags



HIT4 = {"S_DETACHED", "S_LAYOUT", "V_GEOMETRY", "V_HANDWRITING", "P_HANDWRITTEN_OVERRIDE"}
HIT3 = {"P_INHERITED", "P_MULTI_UNLABELED", "P_PROMO", "L_VARIANT", "L_COMBO_SET", "L_ITEM_SPLIT"}


def ccl_of(a: dict):
    if a.get("split") == "ood":
        return None
    t = set(a.get("tags", []))
    if t & HIT4 or ("V_DAMAGE" in t and any(i.get("legibility", "clear") != "clear" for i in a["items"])):
        return 4
    delta = any(o.get("price_kind") == "delta" for i in a["items"] for o in i["offers"])
    unit = any(o.get("price_kind") in ("bundle", "starting_from")
               or (o.get("price_kind") == "unit_price" and (o.get("unit_qty") or 1) != 1)
               or re.search(r"\bk?g\b|ký|kg", o.get("unit") or "", re.I) for i in a["items"] for o in i["offers"])
    if t & HIT3 or delta or ("P_UNIT_BUNDLE" in t and unit):
        return 3
    norm = sum(1 for i in a["items"] if i.get("norm_name", "").lower() != i.get("raw_name", "").lower().rstrip(": "))
    return 2 if "N_SECTION_COMPOSE" in t or norm >= 3 else 1




def load_docs(src):
    src = Path(src)
    if src.is_dir():
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(src.glob("*.json")) if p.name != "schema_config.json"]
    docs = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            docs.append(json.loads(line))
    return docs


def main():
    if len(sys.argv) < 3:
        print(__doc__); return 1
    src, who = sys.argv[1], sys.argv[2]
    out = Path(sys.argv[3] if len(sys.argv) > 3 else "outputs/data/agreement_sets") / who
    out.mkdir(parents=True, exist_ok=True)
    by_sha = {v["file_sha256"]: k for k, v in MAPPING.items()}
    seen, warn = set(), []
    for d in load_docs(src):
        fn = d.get("file_name", "")
        if fn not in MAPPING:
            fn = by_sha.get(d.get("file_sha256", ""), fn)
        if fn not in MAPPING:
            warn.append(f"bỏ qua {d.get('file_name')}: không thuộc 30 ảnh gán mù"); continue
        if d.get("annotator") and d["annotator"] != who:
            if d.get("annotator") != who:
                warn.append(f"{fn}: annotator trong file là {d['annotator']!r}, ghi lại thành {who!r}")
        m = MAPPING[fn]
        if d.get("file_sha256") and d["file_sha256"] != m["file_sha256"]:
            warn.append(f"{fn}: sha256 ảnh khác bộ dữ liệu (ảnh bị sửa?)")
        if d.get("status") != "done":
            warn.append(f"{fn}: chưa bấm Xong (status={d.get('status')!r})")
        d.update(file_name=fn, image_id=m["image_id"], file_sha256=m["file_sha256"], group_id=m["group_id"], split="test",
                 annotator=who, label_origin="human", schema_version="3.1")
        if m.get("image_width"):
            d.setdefault("image_width", m["image_width"]); d.setdefault("image_height", m["image_height"])
        d.setdefault("domain", "food_menu")
        d.setdefault("sections", []); d.setdefault("tags", []); d.setdefault("languages", ["vi"])
        for i, it in enumerate(d.get("items", []), 1):
            for k, v in ITEM_DEFAULTS.items():
                it.setdefault(k, json.loads(json.dumps(v)))
            it.setdefault("item_id", i); it.setdefault("offers", [])
            for o in it["offers"]:
                for k, v in OFFER_DEFAULTS.items():
                    o.setdefault(k, v)
        rules(d)
        auto = m6_auto_tags(dict(d, tags=[t for t in d["tags"] if t not in FIELD_TAGS]))
        d["tags"] = sorted({t for t in (auto - FIELD_TAGS) | derive_field_tags(d) if t in TAG_ORDER}, key=TAG_ORDER.index)
        d["ccl"] = ccl_of(d)
        (out / f"{m['image_id']}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        seen.add(fn)
    print(f"Đã ghi {len(seen)} file vào {out}")
    for w in warn:
        print("  !", w)
    print("Các bước chuẩn hoá đã áp:", dict(Counter(c["rule"] for c in changes)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
