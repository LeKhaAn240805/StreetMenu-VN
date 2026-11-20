"""Parse raw model text: strip reasoning, strict JSON -> lenient (fences, {..} span, json-repair) -> failed."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from smv.schema.prediction import Prediction

THINK_PAIRS = [("<think>", "</think>"), ("<|channel>thought", "<channel|>"), ("<|begin_of_thought|>", "<|end_of_thought|>"),
               ("◁think▷", "◁/think▷"), ("<reasoning>", "</reasoning>")]
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


@dataclass
class ParseResult:
    status: str                       # strict | lenient | failed
    prediction: Prediction | None = None
    lines: list[str] = field(default_factory=list)
    obj: Any = None
    truncated: bool = False
    repetition: bool = False
    think_chars: int = 0
    error: str = ""

    @property
    def valid(self) -> bool:
        return self.status in ("strict", "lenient")


def strip_reasoning(text: str, pairs: list[tuple[str, str]] | None = None) -> tuple[str, int]:
    """Remove reasoning blocks. Returns (answer, number of reasoning characters removed)."""
    if not text:
        return "", 0
    removed = 0
    for start, end in (pairs or []) + THINK_PAIRS:
        while True:
            i = text.find(start)
            j = text.find(end, i + len(start)) if i >= 0 else -1
            if i >= 0 and j >= 0:
                removed += j + len(end) - i
                text = text[:i] + text[j + len(end):]
                continue
            if i < 0 and end in text:  # template already opened the block: drop everything before </end>
                j = text.rfind(end)
                removed += j + len(end)
                text = text[j + len(end):]
                continue
            if i >= 0 and j < 0:      # unterminated reasoning (truncated while thinking)
                removed += len(text) - i
                text = text[:i]
            break
    return text.strip(), removed


def _loads(s: str) -> Any:
    return json.loads(s)


def _lenient(text: str) -> Any:
    m = _FENCE.search(text)
    if m:
        try:
            return _loads(m.group(1).strip())
        except json.JSONDecodeError:
            text = m.group(1)
    a, b = text.find("{"), text.rfind("}")
    if 0 <= a < b:
        try:
            return _loads(text[a:b + 1])
        except json.JSONDecodeError:
            pass
    try:
        import json_repair

        obj = json_repair.loads(text)
        if obj in ("", None, [], {}):
            raise ValueError("json-repair returned nothing")
        return obj
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"unparseable: {e}") from e


def _as_items_obj(obj: Any) -> dict:
    if isinstance(obj, list):
        return {"items": obj}
    if isinstance(obj, dict):
        if "items" in obj:
            return obj
        for k in ("menu", "data", "menu_items", "results", "dishes"):
            if isinstance(obj.get(k), list):
                return {"items": obj[k]}
            if isinstance(obj.get(k), dict) and "items" in obj[k]:
                return obj[k]
        if "raw_name" in obj or "name" in obj:
            return {"items": [obj]}
    raise ValueError("no items list in output")


def _repetition(names: list[str], limit: int) -> bool:
    if not names:
        return False
    c = Counter(n for n in names if n)
    return bool(c) and c.most_common(1)[0][1] > limit


def parse_output(text: str | None, kind: str = "json", finish_reason: str = "",
                 strip_pairs: list[tuple[str, str]] | None = None, repetition_limit: int = 5) -> ParseResult:
    raw = text or ""
    answer, think_chars = strip_reasoning(raw, strip_pairs)
    truncated = finish_reason == "length"
    if kind == "free":
        lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
        return ParseResult("strict" if answer else "failed", lines=lines, obj=answer, truncated=truncated,
                           repetition=_repetition(lines, repetition_limit), think_chars=think_chars)
    status = "strict"
    try:
        obj = _loads(answer.strip())
    except (json.JSONDecodeError, ValueError):
        status = "lenient"
        try:
            obj = _lenient(answer)
        except ValueError as e:
            if kind == "lines":
                lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
                return ParseResult("failed", lines=lines, truncated=truncated, think_chars=think_chars,
                                   repetition=_repetition(lines, repetition_limit), error=str(e))
            return ParseResult("failed", truncated=truncated, think_chars=think_chars, error=str(e))
    if kind == "lines":
        lines: list[str] = []
        if isinstance(obj, dict) and isinstance(obj.get("lines"), list):
            lines = [str(x) for x in obj["lines"] if x is not None]
        elif isinstance(obj, list):
            lines = [str(x) if not isinstance(x, dict) else " ".join(map(str, x.values())) for x in obj]
        else:
            return ParseResult("failed", obj=obj, truncated=truncated, think_chars=think_chars,
                               error="no 'lines' list")
        return ParseResult(status, lines=lines, obj=obj, truncated=truncated, think_chars=think_chars,
                           repetition=_repetition(lines, repetition_limit))
    try:
        pred = Prediction.model_validate(_as_items_obj(obj))
    except (ValueError, ValidationError) as e:
        return ParseResult("failed", obj=obj, truncated=truncated, think_chars=think_chars, error=str(e)[:300])
    from smv.scoring.normalize import normalize_name

    names = [normalize_name(it.name or it.raw_name) for it in pred.items]
    return ParseResult(status, prediction=pred, obj=obj, truncated=truncated, think_chars=think_chars,
                       repetition=_repetition(names, repetition_limit))


def ocr_text_to_lines(text: str) -> list[str]:
    """Normalise the text output of an OCR-type VLM (markdown/HTML tables, layout JSON) into lines."""
    t, _ = strip_reasoning(text or "")
    # layout JSON (dots.ocr style): list of {"bbox":..., "text":...}
    s = t.strip()
    if s.startswith(("[", "{")):
        try:
            obj = json.loads(s)
            cells = obj if isinstance(obj, list) else obj.get("layout") or obj.get("blocks") or []
            items = []
            for c in cells:
                if isinstance(c, dict) and c.get("text"):
                    bb = c.get("bbox") or [0, 0, 0, 0]
                    items.append((bb, str(c["text"])))
            if items:
                items.sort(key=lambda x: (round((x[0][1] if len(x[0]) > 1 else 0) / 20), x[0][0] if x[0] else 0))
                t = "\n".join(txt for _, txt in items)
        except (json.JSONDecodeError, TypeError, IndexError, KeyError):
            pass
    t = re.sub(r"</?(?:td|th)[^>]*>", " | ", t, flags=re.I)
    t = re.sub(r"<br\s*/?>|</?(?:tr|p|div|li|table|tbody|thead)[^>]*>", "\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    out = []
    for ln in t.splitlines():
        ln = re.sub(r"^\s*(?:#{1,6}\s+|[-*]\s+)", "", ln)
        ln = re.sub(r"(?:\s*\|\s*)+", " | ", ln).strip(" |")
        ln = re.sub(r"\*\*|__", "", ln)
        ln = re.sub(r"\s+", " ", ln).strip()
        if ln and not re.fullmatch(r"[-:| ]+", ln):
            out.append(ln)
    return out
