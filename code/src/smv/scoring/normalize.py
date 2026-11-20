"""String and price normalisation shared by the scorer, the OCR+regex baseline and the parser."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Any

_LEADING = re.compile(r"^\s*(?:[\(\[]?\s*(?:\d{1,3}|[a-zA-Z])\s*[\.\)\]:/-]\s+|[-–—•*·+#>~=]+\s*|[①-⑳]\s*)+")
_TRAILING_PUNCT = re.compile(r"[\s:;,.\-–—*·]+$")
_WS = re.compile(r"\s+")
_PUNCT_NOSPACE = re.compile(r"[\"“”'‘’`]")

# Vietnamese has letters (đ) that NFD does not decompose
_EXTRA_FOLD = str.maketrans({"đ": "d", "Đ": "D"})


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def strip_diacritics(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s.translate(_EXTRA_FOLD))


def _compile_lexicon(words: dict[str, str]) -> list[tuple[re.Pattern, str]]:
    out = []
    # longest keys first so multi-word entries win
    for k in sorted(words, key=len, reverse=True):
        v = words[k]
        key = nfc(str(k).lower())
        out.append((re.compile(r"(?<![\w])" + re.escape(key) + r"(?![\w])"), nfc(str(v).lower())))
    return out


@lru_cache(maxsize=4)
def _lexicon_patterns(items: tuple[tuple[str, str], ...]) -> list[tuple[re.Pattern, str]]:
    return _compile_lexicon(dict(items))


def normalize_name(s: Any, mode: str = "strict", lexicon: dict[str, str] | None = None) -> str:
    """NFC, lowercase, drop leading numbering/bullets and trailing punctuation, collapse spaces,
    apply the lexicon; lenient mode also strips diacritics."""
    if s is None:
        return ""
    s = nfc(str(s)).lower()
    s = _PUNCT_NOSPACE.sub("", s)
    s = _LEADING.sub("", s)
    s = _TRAILING_PUNCT.sub("", s)
    s = _WS.sub(" ", s).strip()
    if lexicon:
        for pat, rep in _lexicon_patterns(tuple(sorted(lexicon.items()))):
            s = pat.sub(rep, s)
        s = _WS.sub(" ", s).strip()
    if mode == "lenient":
        s = strip_diacritics(s)
    return s


def normalize_attr(s: Any) -> str:
    if s is None:
        return ""
    s = nfc(str(s)).lower().strip()
    s = re.sub(r"^(size|sz|cỡ|co)\s*[:\-]?\s*", "", s)
    return _WS.sub(" ", s).strip(" .:-")


def canonical_size(s: Any, sizes: dict[str, Any] | None = None) -> tuple[str, int | None]:
    """Return (canonical label, rank) for a size string. Unknown sizes keep their text, rank None.
    '#k' (unlabeled tiers) -> ('#k', k)."""
    t = normalize_attr(s)
    if not t:
        return "", None
    m = re.fullmatch(r"#\s*(\d+)", t)
    if m:
        return f"#{m.group(1)}", int(m.group(1))
    if sizes:
        for key in (t, strip_diacritics(t)):
            if key in sizes:
                v = sizes[key]
                return str(v.get("label", key)), v.get("rank")
    return t, None


def split_temp(s: Any, temps: dict[str, str] | None = None) -> list[str]:
    """'nóng|đá', 'nóng/đá', 'hot / iced' -> ['nóng', 'đá']; '' -> ['']."""
    t = normalize_attr(s)
    if not t:
        return [""]
    parts = [p.strip() for p in re.split(r"[|/,]| và | or | hoặc |&", t) if p.strip()]
    out = []
    for p in parts:
        canon = (temps or {}).get(p) or (temps or {}).get(strip_diacritics(p)) or p
        if canon not in out:
            out.append(canon)
    return out or [""]


# ----------------------------------------------------------------------------------- prices
_NUM = r"\d{1,3}(?:[.,\s]\d{3})+|\d+(?:[.,]\d+)?"
_PRICE_TOKEN = re.compile(
    r"(?<![\w.])(?P<pre>vnd|vnđ|đ)?\s*(?P<num>" + _NUM + r")\s*(?P<suf>k\b|k(?=[^a-zA-Z])|ngàn|nghìn|ng\b|tr\b|triệu|"
    r"đồng|vnd\b|vnđ\b|đ|d\b|000|⁰⁰⁰)?",
    re.IGNORECASE,
)


def _num_value(num: str) -> float | None:
    n = num.strip().replace(" ", "")
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", n):
        return float(re.sub(r"[.,]", "", n))
    if re.fullmatch(r"\d+[.,]\d+", n):
        return float(n.replace(",", "."))
    try:
        return float(n)
    except ValueError:
        return None


def parse_price_tokens(text: str, implicit_k: bool | None = None) -> list[int]:
    """Extract every price in a string as integer VND.

    implicit_k: bare numbers < 1000 mean thousands (True), are kept as-is (False) or decided per
    number (None: < 1000 -> thousands)."""
    if not text:
        return []
    out = []
    for m in _PRICE_TOKEN.finditer(nfc(text)):
        val = _num_value(m.group("num"))
        if val is None:
            continue
        suf = (m.group("suf") or "").lower()
        pre = (m.group("pre") or "").lower()
        if suf in ("k", "ngàn", "nghìn", "ng"):
            vnd = val * 1000
        elif suf in ("tr", "triệu"):
            vnd = val * 1_000_000
        elif suf in ("000", "⁰⁰⁰"):
            vnd = val * 1000
        elif suf in ("đồng", "vnd", "vnđ", "đ", "d") or pre:
            vnd = val if val >= 1000 else val * 1000
        else:
            if implicit_k is False:
                vnd = val
            else:
                vnd = val * 1000 if val < 1000 else val
        if vnd <= 0:
            continue
        out.append(int(round(vnd)))
    return out


def parse_price_value(v: Any) -> int | None:
    """Coerce a model-provided price_vnd value (int, float, '25.000', '25k', '25000đ') to int VND."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(round(v))
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in ("null", "none", "n/a", "-"):
            return None
        if re.fullmatch(r"\d+", s):
            return int(s)
        toks = parse_price_tokens(s, implicit_k=False)
        return toks[0] if toks else None
    if isinstance(v, dict):
        for k in ("value", "price", "amount"):
            if k in v:
                return parse_price_value(v[k])
    return None


def digits_of(s: Any) -> str:
    return re.sub(r"\D", "", str(s or ""))


def tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[^\w]+", nfc(s.lower())) if t]
