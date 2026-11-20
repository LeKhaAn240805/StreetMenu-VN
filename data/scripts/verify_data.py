#!/usr/bin/env python3
"""Kiểm tra nhanh thư mục dữ liệu StreetMenu-VN trên máy chủ (chỉ dùng thư viện chuẩn).

    python3 scripts/verify_data.py            # chạy trong thư mục StreetMenuVN_final
    python3 scripts/verify_data.py /data/StreetMenuVN_final

Kiểm: đủ ảnh cho mỗi nhãn, sha256 khớp, split khớp splits.csv, không nhóm nào nằm ở hai split,
enum theo schema v3.1, range có min/max, giá là số nguyên. In ra số ảnh/món/offer theo split.
Không mở hay hiển thị nội dung ảnh, chỉ đọc byte để tính sha256.
"""
import csv, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
sch = json.loads((root / 'schema/schema_config_v3.1.json').read_text(encoding='utf-8'))
enum = {}
for grp in ('image_fields', 'item_fields', 'offer_fields'):
    for f in sch[grp]:
        if f.get('options'):
            enum[(grp, f['key'])] = {o if isinstance(o, str) else o.get('v', o.get('value')) for o in f['options']}
tags = {t['code'] for g in sch['tag_groups'] for t in g['tags']}
splits = {r['file_name']: r for r in csv.DictReader(open(root / 'splits.csv', encoding='utf-8'))}
err, warn = [], []
count = defaultdict(Counter)
group_split = defaultdict(set)
labels = sorted((root / 'adjudicated').glob('*.json'))
for p in labels:
    a = json.loads(p.read_text(encoding='utf-8'))
    fn = a['file_name']
    img = root / 'images' / fn
    if not img.exists():
        err.append(f'{fn}: thiếu ảnh'); continue
    if hashlib.sha256(img.read_bytes()).hexdigest() != a['file_sha256']:
        err.append(f'{fn}: sha256 không khớp')
    s = splits.get(fn)
    if not s:
        err.append(f'{fn}: không có trong splits.csv')
    elif s['split'] != a['split']:
        err.append(f'{fn}: split nhãn {a["split"]} khác splits.csv {s["split"]}')
    else:
        group_split[s['group_id']].add(s['split'])
    for t in a['tags']:
        if t not in tags:
            err.append(f'{fn}: thẻ lạ {t}')
    for k in ('split', 'domain', 'venue_tier', 'price_unit_default'):
        if a.get(k) not in (None, '') and a[k] not in enum.get(('image_fields', k), {a[k]}):
            err.append(f'{fn}: {k}={a[k]!r} ngoài enum')
    if (a['ccl'] is None) != (a['split'] == 'ood'):
        err.append(f'{fn}: ccl={a["ccl"]} không hợp với split={a["split"]}')
    c = count[a['split']]
    c['images'] += 1; c['items'] += len(a['items'])
    for it in a['items']:
        for k in ('item_status', 'legibility'):
            if it.get(k) not in enum[('item_fields', k)]:
                err.append(f'{fn}#{it["item_id"]}: {k}={it.get(k)!r}')
        for o in it['offers']:
            c['offers'] += 1
            if o['ambiguous'] or it.get('item_status') != 'active' or it.get('legibility') == 'illegible':
                c['offers_excluded_main'] += 1
            if o['price_kind'] not in enum[('offer_fields', 'price_kind')]:
                err.append(f'{fn}#{it["item_id"]}: price_kind={o["price_kind"]!r}')
            v = o['price_vnd']
            if v is not None and not isinstance(v, int):
                err.append(f'{fn}#{it["item_id"]}: price_vnd không nguyên {v!r}')
            if o['price_kind'] == 'range' and (o['price_min_vnd'] is None or o['price_max_vnd'] is None):
                err.append(f'{fn}#{it["item_id"]}: range thiếu min/max')
            if v is not None and not o.get('raw_price') and o.get('price_source') == 'item_line':
                warn.append(f'{fn}#{it["item_id"]}: có giá nhưng thiếu raw_price')
leak = [g for g, s in group_split.items() if len(s) > 1]
if leak:
    err.append(f'nhóm nằm ở hai split: {leak[:5]}')
n_img = len([p for p in (root / 'images').iterdir() if p.is_file()])
print(f'Nhãn: {len(labels)}  Ảnh trong images/: {n_img}')
for k in ('dev', 'test', 'ood'):
    c = count[k]
    print(f'  {k:5s} ảnh {c["images"]:4d}  món {c["items"]:5d}  offer {c["offers"]:5d}  (không tính điểm chính: {c["offers_excluded_main"]})')
print(f'Lỗi: {len(err)}  Cảnh báo: {len(warn)}')
for e in err[:30]:
    print('  LỖI', e)
for w in warn[:10]:
    print('  cảnh báo', w)
sys.exit(1 if err else 0)
