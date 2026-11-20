#!/usr/bin/env python3
"""Gộp kết quả mã hóa lỗi của 4 người và tính độ đồng thuận (chỉ dùng thư viện chuẩn).

Cách dễ nhất: để file này, coding_sheet.csv và 4 file ketqua_*.csv (C1, C2, C3, C4) chung một thư mục rồi chạy
    python gop_ket_qua.py
Hoặc chỉ rõ:
    python gop_ket_qua.py coding_sheet.csv ketqua_Vinh.csv ketqua_Long.csv ketqua_An.csv ketqua_Lam.csv

Ghi vào thư mục ket_qua_gop/ (đổi bằng -o <thư mục>):
  coding_sheet_da_gop.csv   cùng cột, cùng thứ tự với coding_sheet.csv, đã điền theme_coder1/2 và note_coder1/2
                            → đem lên máy chủ chạy: python scripts/qual_coding.py score
  can_thong_nhat.csv        các ca hai người chọn khác nhau, cột theme_final để trống cho người thống nhất điền
  bao_cao.md, bao_cao.json  phân bố mã, đồng thuận (raw, Cohen κ, Scott π/Fleiss κ, Krippendorff α) theo từng cặp và gộp
"""
import csv, glob, json, os, re, sys
from collections import Counter
from pathlib import Path

CODES = ["read_print", "read_hand", "read_scale", "struct_inherit", "struct_variant", "struct_spatial", "name_norm", "halluc", "omit_dense", "omit_trunc", "other"]
PAIR = {"Q000": "Cặp 1", "Q001": "Cặp 1", "Q002": "Cặp 1", "Q003": "Cặp 2", "Q004": "Cặp 2", "Q005": "Cặp 2", "Q006": "Cặp 1", "Q007": "Cặp 2", "Q008": "Cặp 1", "Q009": "Cặp 2", "Q010": "Cặp 1", "Q011": "Cặp 2", "Q012": "Cặp 1", "Q013": "Cặp 1", "Q014": "Cặp 2", "Q015": "Cặp 1", "Q016": "Cặp 2", "Q017": "Cặp 2", "Q018": "Cặp 2", "Q019": "Cặp 2", "Q020": "Cặp 2", "Q021": "Cặp 1", "Q022": "Cặp 1", "Q023": "Cặp 2", "Q024": "Cặp 2", "Q025": "Cặp 1", "Q026": "Cặp 1", "Q027": "Cặp 1", "Q028": "Cặp 2", "Q029": "Cặp 1", "Q030": "Cặp 1", "Q031": "Cặp 1", "Q032": "Cặp 1", "Q033": "Cặp 2", "Q034": "Cặp 1", "Q035": "Cặp 1", "Q036": "Cặp 2", "Q037": "Cặp 2", "Q038": "Cặp 1", "Q039": "Cặp 2", "Q040": "Cặp 1", "Q041": "Cặp 2", "Q042": "Cặp 1", "Q043": "Cặp 2", "Q044": "Cặp 2", "Q045": "Cặp 2", "Q046": "Cặp 2", "Q047": "Cặp 2", "Q048": "Cặp 1", "Q049": "Cặp 1", "Q050": "Cặp 2", "Q051": "Cặp 2", "Q052": "Cặp 2", "Q053": "Cặp 2", "Q054": "Cặp 1", "Q055": "Cặp 1", "Q056": "Cặp 2", "Q057": "Cặp 1", "Q058": "Cặp 1", "Q059": "Cặp 1", "Q060": "Cặp 2", "Q061": "Cặp 2", "Q062": "Cặp 2", "Q063": "Cặp 2", "Q064": "Cặp 1", "Q065": "Cặp 1", "Q066": "Cặp 2", "Q067": "Cặp 2", "Q068": "Cặp 1", "Q069": "Cặp 1", "Q070": "Cặp 2", "Q071": "Cặp 2", "Q072": "Cặp 2", "Q073": "Cặp 1", "Q074": "Cặp 1", "Q075": "Cặp 2", "Q076": "Cặp 2", "Q077": "Cặp 1", "Q078": "Cặp 1", "Q079": "Cặp 2", "Q080": "Cặp 1", "Q081": "Cặp 2", "Q082": "Cặp 1", "Q083": "Cặp 2", "Q084": "Cặp 2", "Q085": "Cặp 2", "Q086": "Cặp 2", "Q087": "Cặp 2", "Q088": "Cặp 2", "Q089": "Cặp 1", "Q090": "Cặp 2", "Q091": "Cặp 2", "Q092": "Cặp 1", "Q093": "Cặp 1", "Q094": "Cặp 1", "Q095": "Cặp 1", "Q096": "Cặp 2", "Q097": "Cặp 1", "Q098": "Cặp 2", "Q099": "Cặp 2", "Q100": "Cặp 2", "Q101": "Cặp 1", "Q102": "Cặp 2", "Q103": "Cặp 2", "Q104": "Cặp 1", "Q105": "Cặp 1", "Q106": "Cặp 1", "Q107": "Cặp 2", "Q108": "Cặp 1", "Q109": "Cặp 2", "Q110": "Cặp 1", "Q111": "Cặp 1", "Q112": "Cặp 2", "Q113": "Cặp 2", "Q114": "Cặp 1", "Q115": "Cặp 2", "Q116": "Cặp 1", "Q117": "Cặp 1", "Q118": "Cặp 1", "Q119": "Cặp 2", "Q120": "Cặp 2", "Q121": "Cặp 1", "Q122": "Cặp 2", "Q123": "Cặp 2", "Q124": "Cặp 2", "Q125": "Cặp 2", "Q126": "Cặp 2", "Q127": "Cặp 1", "Q128": "Cặp 1", "Q129": "Cặp 1", "Q130": "Cặp 1", "Q131": "Cặp 2", "Q132": "Cặp 2", "Q133": "Cặp 1", "Q134": "Cặp 1", "Q135": "Cặp 2", "Q136": "Cặp 2", "Q137": "Cặp 1", "Q138": "Cặp 1", "Q139": "Cặp 1", "Q140": "Cặp 1", "Q141": "Cặp 2", "Q142": "Cặp 2", "Q143": "Cặp 1", "Q144": "Cặp 1", "Q145": "Cặp 1", "Q146": "Cặp 1", "Q147": "Cặp 1", "Q148": "Cặp 1", "Q149": "Cặp 1"}  # case_id -> "Cặp 1" / "Cặp 2"
ROLE = {("Cặp 1", "theme_coder1"): "C1", ("Cặp 1", "theme_coder2"): "C2",
        ("Cặp 2", "theme_coder1"): "C3", ("Cặp 2", "theme_coder2"): "C4"}


def read_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        r = list(csv.reader(f))
    return r[0], [x for x in r[1:] if x]


def cohen(a, b):
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return None if pe == 1 else (po - pe) / (1 - pe)


def scott_fleiss(a, b):  # Fleiss κ với 2 người/ca = Scott π
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    c = Counter(a) + Counter(b)
    pe = sum((v / (2 * n)) ** 2 for v in c.values())
    return None if pe == 1 else (po - pe) / (1 - pe)


def kripp_alpha(a, b):  # nominal, 2 người, không thiếu
    n = len(a)
    if not n:
        return None
    c = Counter(a) + Counter(b); N = 2 * n
    do = sum(x != y for x, y in zip(a, b)) * 2 / N
    de = sum(c[i] * c[j] for i in c for j in c if i != j) / (N * (N - 1))
    return None if de == 0 else 1 - do / de


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    here = Path(__file__).resolve().parent
    args = sys.argv[1:]
    out = None
    if "-o" in args:
        i = args.index("-o"); out = Path(args[i + 1]); del args[i:i + 2]
    if any(a in ("-h", "--help") for a in args):
        sys.exit(__doc__)
    files = []
    for a in args:
        if any(ch in a for ch in "*?["):
            files += sorted(glob.glob(a))
        elif Path(a).is_dir():
            files += [str(p) for p in sorted(Path(a).glob("*.csv"))]
        else:
            files.append(a)
    base = Path(files[0]).parent if files else here
    sheet = next((f for f in files if Path(f).name == "coding_sheet.csv"), None)
    if sheet is None:
        sheet = base / "coding_sheet.csv"
        if not sheet.exists():
            sheet = here / "coding_sheet.csv"
    res = [f for f in files if Path(f).name != "coding_sheet.csv"]
    if not res:
        res = [str(p) for p in sorted(base.glob("ketqua_*.csv"))]
    if out is None:
        out = base / "ket_qua_gop"
    if not Path(sheet).exists() or not res:
        sys.exit(f"Không thấy coding_sheet.csv hoặc file ketqua_*.csv trong {base}.\n" + __doc__)
    args = [str(sheet)] + res
    cols, rows = read_csv(args[0])
    ix = {c: i for i, c in enumerate(cols)}
    by = {r[ix["case_id"]]: r for r in rows}
    for c in ("theme_coder1", "note_coder1", "theme_coder2", "note_coder2"):
        assert c in ix, f"thiếu cột {c}"
    who = {}  # (case_id, cột) -> file
    problems = []
    for p in args[1:]:
        h, rr = read_csv(p)
        hx = {c: i for i, c in enumerate(h)}
        if "case_id" not in hx:
            problems.append(f"{Path(p).name}: không có cột case_id, bỏ qua"); continue
        m = re.match(r"ketqua_(C1|C2|C3|C4)(?![A-Za-z])", Path(p).name); frole = m.group(1) if m else None
        n = 0
        for r in rr:
            cid = r[hx["case_id"]]
            if cid not in by:
                problems.append(f"{Path(p).name}: ca lạ {cid}"); continue
            for t, nt in (("theme_coder1", "note_coder1"), ("theme_coder2", "note_coder2")):
                v = r[hx[t]].strip() if t in hx else ""
                if not v:
                    continue
                if v not in CODES:
                    problems.append(f"{Path(p).name}: {cid} mã lạ {v!r}"); continue
                if frole and ROLE.get((PAIR.get(cid), t)) != frole:
                    problems.append(f"{Path(p).name}: {cid} có mã ở {t} nhưng ca này không giao cho {frole}"); continue
                cur = by[cid][ix[t]].strip()
                if cur and cur != v:
                    problems.append(f"{cid} {t}: {who.get((cid, t))} ghi {cur!r}, {Path(p).name} ghi {v!r} (giữ bản đầu)"); continue
                by[cid][ix[t]] = v
                by[cid][ix[nt]] = r[hx[nt]] if nt in hx else ""
                who[(cid, t)] = Path(p).name; n += 1
        print(f"{Path(p).name}: {n} mã")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "coding_sheet_da_gop.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(cols); w.writerows(rows)

    T1, T2 = ix["theme_coder1"], ix["theme_coder2"]
    both = [r for r in rows if r[T1] and r[T2]]
    rep = {"n_cases": len(rows), "n_coder1": sum(1 for r in rows if r[T1]), "n_coder2": sum(1 for r in rows if r[T2]), "n_both": len(both)}
    groups = {"Gộp": both}
    for pr in sorted(set(PAIR.values())):
        groups[pr] = [r for r in both if PAIR.get(r[ix["case_id"]]) == pr]
    rep["agreement"] = {}
    for g, R in groups.items():
        a = [r[T1] for r in R]; b = [r[T2] for r in R]
        rep["agreement"][g] = dict(n=len(R), raw=round(sum(x == y for x, y in zip(a, b)) / len(R), 4) if R else None,
                                   cohen_kappa=None if cohen(a, b) is None else round(cohen(a, b), 4),
                                   fleiss_kappa=None if scott_fleiss(a, b) is None else round(scott_fleiss(a, b), 4),
                                   krippendorff_alpha=None if kripp_alpha(a, b) is None else round(kripp_alpha(a, b), 4))
    a = [r[T1] for r in both]; b = [r[T2] for r in both]
    rep["per_code"] = {}
    for c in CODES:
        n1, n2, nb = a.count(c), b.count(c), sum(1 for x, y in zip(a, b) if x == y == c)
        rep["per_code"][c] = dict(coder1=n1, coder2=n2, both=nb, specific_agreement=round(2 * nb / (n1 + n2), 3) if n1 + n2 else None)
    allc = [r[T1] for r in rows if r[T1]] + [r[T2] for r in rows if r[T2]]
    agreed = [r[T1] for r in both if r[T1] == r[T2]]
    rep["distribution"] = {c: dict(all_codes=allc.count(c), pct_all=round(100 * allc.count(c) / len(allc), 1) if allc else None,
                                   agreed=agreed.count(c)) for c in CODES}
    if "auto_theme" in ix:
        rep["vs_auto_theme"] = {t: round(sum(r[ix[t]] == r[ix["auto_theme"]] for r in rows if r[ix[t]]) / max(1, sum(1 for r in rows if r[ix[t]])), 4)
                                for t in ("theme_coder1", "theme_coder2")}
    rep["confusion"] = {f"{x} / {y}": n for (x, y), n in sorted(Counter(zip(a, b)).items()) if x != y}
    rep["problems"] = problems
    (out / "bao_cao.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")

    keep = ["case_id", "model", "file_name", "status", "gold_raw_name", "gold_raw_price", "pred_raw_name", "pred_raw_price", "truncated"]
    with open(out / "can_thong_nhat.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(keep + ["theme_coder1", "note_coder1", "theme_coder2", "note_coder2", "theme_final", "ly_do"])
        for r in both:
            if r[T1] != r[T2]:
                w.writerow([r[ix[k]] for k in keep if k in ix] + [r[T1], r[ix["note_coder1"]], r[T2], r[ix["note_coder2"]], "", ""])

    L = ["# Kết quả mã hóa lỗi", "",
         f"- Số ca: {rep['n_cases']}. Có mã ở cột theme_coder1: {rep['n_coder1']}, theme_coder2: {rep['n_coder2']}, cả hai cột: {rep['n_both']}.",
         "- Cột theme_coder1 là C1 (Cặp 1) và C3 (Cặp 2); cột theme_coder2 là C2 và C4.", "",
         "## Đồng thuận", "", "| Nhóm | Số ca | Trùng | Cohen κ | Fleiss κ | Krippendorff α |", "|---|---|---|---|---|---|"]
    for g, s in rep["agreement"].items():
        L.append(f"| {g} | {s['n']} | {s['raw']} | {s['cohen_kappa']} | {s['fleiss_kappa']} | {s['krippendorff_alpha']} |")
    L += ["", "Với 2 người mỗi ca, Fleiss κ trùng Scott π. Cặp 1 (C1, C2) và Cặp 2 (C3, C4) chấm hai nửa khác nhau của 150 ca.", "",
          "## Phân bố mã", "", "| Mã | Số lần chọn (cả hai người) | % | Hai người cùng chọn | Đồng thuận riêng mã |", "|---|---|---|---|---|"]
    for c in CODES:
        d = rep["distribution"][c]; s = rep["per_code"][c]
        L.append(f"| {c} | {d['all_codes']} | {d['pct_all']} | {d['agreed']} | {s['specific_agreement']} |")
    if "vs_auto_theme" in rep:
        L += ["", f"Trùng với gợi ý của máy (auto_theme): cột theme_coder1 {rep['vs_auto_theme']['theme_coder1']}, theme_coder2 {rep['vs_auto_theme']['theme_coder2']}."]
    if rep["confusion"]:
        L += ["", "## Cặp mã hay lệch nhau", "", "| theme_coder1 / theme_coder2 | Số ca |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in sorted(rep["confusion"].items(), key=lambda kv: -kv[1])[:12]]
    if problems:
        L += ["", "## Cần xem", ""] + [f"- {p}" for p in problems]
    (out / "bao_cao.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    miss = {g: sum(1 for r in rows if PAIR.get(r[ix["case_id"]]) == g and not (r[T1] and r[T2])) for g in sorted(set(PAIR.values()))}
    print("\nCa chưa đủ 2 mã: " + ", ".join(f"{g} {k}" for g, k in miss.items()))
    print(f"Đã ghi {out}{os.sep}coding_sheet_da_gop.csv, can_thong_nhat.csv, bao_cao.md, bao_cao.json")
    if not sys.argv[1:] and os.name == "nt":
        input("Nhấn Enter để đóng...")


if __name__ == "__main__":
    main()
