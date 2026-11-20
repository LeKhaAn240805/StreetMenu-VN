"""Classic OCR worker (runs in the OCR venv; standalone).

Usage: python -I ocr_worker.py <cfg.json>
cfg.ocr.engine: paddleocr | easyocr | ppocr_vietocr. Writes per image: text (lines joined), ocr_lines
(sorted top-left -> bottom-right), ocr_boxes. Engines download their weights into cfg.models_dir, which the
orchestrator deletes afterwards."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback


def log(msg: str) -> None:
    print(f"[ocr_worker {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_lines(boxes: list[tuple[list[float], str]]) -> list[str]:
    """Group word/line boxes into reading-order lines: rows by vertical overlap, then left to right."""
    items = []
    for bb, txt in boxes:
        txt = (txt or "").strip()
        if not txt:
            continue
        xs, ys = bb[0::2], bb[1::2]
        items.append((min(ys), max(ys), min(xs), txt))
    if not items:
        return []
    heights = sorted(b - a for a, b, _, _ in items)
    tol = max(heights[len(heights) // 2] * 0.5, 4)
    items.sort(key=lambda t: ((t[0] + t[1]) / 2, t[2]))
    rows: list[list] = []
    for it in items:
        yc = (it[0] + it[1]) / 2
        if rows and abs(rows[-1][0] - yc) <= tol:
            rows[-1][1].append(it)
            n = len(rows[-1][1])
            rows[-1][0] = (rows[-1][0] * (n - 1) + yc) / n
        else:
            rows.append([yc, [it]])
    return [" ".join(t[3] for t in sorted(r[1], key=lambda t: t[2])) for r in rows]


def _poly_to_flat(p) -> list[float]:
    try:
        pts = [float(v) for pt in p for v in pt]
        return pts
    except TypeError:
        x0, y0, x1, y1 = [float(v) for v in p]
        return [x0, y0, x1, y0, x1, y1, x0, y1]


class PaddleEngine:
    def __init__(self, cfg: dict):
        from paddleocr import PaddleOCR

        o = cfg["ocr"].get("options", {})
        self.ocr = PaddleOCR(lang=cfg["ocr"].get("lang", "vi"), use_doc_orientation_classify=False,
                             use_doc_unwarping=False, use_textline_orientation=bool(o.get("textline_orientation", True)),
                             device=o.get("device", "gpu:0"),
                             **{k: v for k, v in o.items() if k in ("text_detection_model_name",
                                                                   "text_recognition_model_name",
                                                                   "text_recognition_model_dir")})

    def run(self, path: str) -> list[tuple[list[float], str]]:
        res = self.ocr.predict(path)
        out = []
        for r in res:
            d = r.json.get("res", r.json) if hasattr(r, "json") else r
            texts = d.get("rec_texts", [])
            polys = d.get("rec_polys") or d.get("dt_polys") or d.get("rec_boxes") or []
            for p, t in zip(polys, texts):
                out.append((_poly_to_flat(p), t))
        return out


class EasyEngine:
    def __init__(self, cfg: dict):
        import easyocr

        o = cfg["ocr"].get("options", {})
        d = os.path.join(cfg["models_dir"], "easyocr")
        os.makedirs(d, exist_ok=True)
        self.reader = easyocr.Reader(o.get("langs", ["vi", "en"]), gpu=bool(o.get("gpu", False)),
                                     model_storage_directory=d, user_network_directory=d, download_enabled=True)

    def run(self, path: str) -> list[tuple[list[float], str]]:
        res = self.reader.readtext(path, paragraph=False)
        return [(_poly_to_flat(bb), txt) for bb, txt, _ in res]


class VietOcrEngine:
    """PaddleOCR text detection + VietOCR (vgg_transformer) line recognition."""

    def __init__(self, cfg: dict):
        import urllib.request

        from paddleocr import TextDetection
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        o = cfg["ocr"].get("options", {})
        self.det = TextDetection(model_name=o.get("det_model", "PP-OCRv5_server_det"), device=o.get("device", "gpu:0"))
        name = o.get("vietocr_config", "vgg_transformer")
        vcfg = Cfg.load_config_from_name(name)
        wdir = os.path.join(cfg["models_dir"], "vietocr")
        os.makedirs(wdir, exist_ok=True)
        wpath = os.path.join(wdir, f"{name}.pth")
        url = o.get("weights_url") or vcfg.get("weights")
        if not os.path.isfile(wpath):
            log(f"downloading VietOCR weights {url}")
            urllib.request.urlretrieve(url, wpath)
        vcfg["weights"] = wpath
        vcfg["device"] = o.get("rec_device", "cpu")
        vcfg["cnn"]["pretrained"] = False
        vcfg["predictor"]["beamsearch"] = False
        self.rec = Predictor(vcfg)

    def run(self, path: str) -> list[tuple[list[float], str]]:
        from PIL import Image

        res = self.det.predict(path)
        out = []
        with Image.open(path) as im:
            im = im.convert("RGB")
            for r in res:
                d = r.json.get("res", r.json) if hasattr(r, "json") else r
                for p in d.get("dt_polys", []):
                    flat = _poly_to_flat(p)
                    xs, ys = flat[0::2], flat[1::2]
                    box = (max(min(xs) - 2, 0), max(min(ys) - 2, 0), min(max(xs) + 2, im.width), min(max(ys) + 2, im.height))
                    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                        continue
                    out.append((flat, self.rec.predict(im.crop(box))))
        return out


ENGINES = {"paddleocr": PaddleEngine, "easyocr": EasyEngine, "ppocr_vietocr": VietOcrEngine}


def main(cfg_path: str) -> int:
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    try:
        import torch

        torch.set_num_threads(int(cfg.get("num_threads", 10)))
    except ImportError:
        pass
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    eng_name = cfg["ocr"].get("engine", "paddleocr")
    log(f"engine {eng_name}")
    try:
        engine = ENGINES[eng_name](cfg)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        log(f"LOAD_FAILED {type(e).__name__}: {e}")
        return 2
    tasks = [json.loads(line) for line in open(cfg["tasks"], encoding="utf-8") if line.strip()]
    with open(cfg["out"], "a", encoding="utf-8") as out:
        for t in tasks:
            t0 = time.time()
            try:
                boxes = engine.run(t["image_path"])
                lines = to_lines(boxes)
                rec = {"id": t["id"], "status": "ok", "text": "\n".join(lines), "ocr_lines": lines,
                       "ocr_boxes": [[round(v, 1) for v in bb] + [txt] for bb, txt in boxes],
                       "finish_reason": "stop", "latency_s": round(time.time() - t0, 3)}
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                rec = {"id": t["id"], "status": "error", "text": "", "error": f"{type(e).__name__}: {str(e)[:500]}",
                       "latency_s": round(time.time() - t0, 3)}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
