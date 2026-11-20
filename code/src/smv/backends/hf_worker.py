"""Transformers backend worker (standalone: stdlib + torch + transformers + PIL).

Usage: python -I hf_worker.py <cfg.json>
cfg: model_dir, hf{adapter, model_class, batch_size, batch_ladder, dtype, max_num_tiles, attn_implementation},
     tasks (jsonl: id, image_path, body{messages, max_tokens, temperature, chat_template_kwargs...}), out (jsonl),
     num_threads, trust_remote_code.
Writes one JSON line per task to `out` as soon as it is done (same fields as the vLLM client)."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback


def log(msg: str) -> None:
    print(f"[hf_worker {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_tasks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def emit(out, rec: dict) -> None:
    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.flush()
    os.fsync(out.fileno())


def split_messages(body: dict) -> tuple[str | None, str, str | None]:
    """OpenAI messages -> (system, user_text, image_path)."""
    system, text, image = None, "", None
    for m in body.get("messages", []):
        c = m.get("content")
        if m.get("role") == "system":
            system = c if isinstance(c, str) else " ".join(p.get("text", "") for p in c)
            continue
        if isinstance(c, str):
            text += c
        else:
            for p in c:
                if p.get("type") == "text":
                    text += p["text"]
                elif p.get("type") == "image_url":
                    url = p["image_url"]["url"]
                    image = url[len("file://"):] if url.startswith("file://") else url
    return system, text, image


# --------------------------------------------------------------------------------------- generic
class Generic:
    def __init__(self, cfg: dict):
        import torch
        import transformers

        self.torch = torch
        hf = cfg["hf"]
        self.trc = bool(cfg.get("trust_remote_code"))
        dtype = getattr(torch, hf.get("dtype", "bfloat16"))
        cls = getattr(transformers, hf.get("model_class", "AutoModelForImageTextToText"))
        kw = {"trust_remote_code": self.trc, "device_map": "cuda:0"}
        if hf.get("attn_implementation"):
            kw["attn_implementation"] = hf["attn_implementation"]
        try:
            self.model = cls.from_pretrained(cfg["model_dir"], dtype=dtype, **kw).eval()
        except TypeError:
            self.model = cls.from_pretrained(cfg["model_dir"], torch_dtype=dtype, **kw).eval()
        self.text_only = hf.get("model_class") == "AutoModelForCausalLM"
        if self.text_only:
            self.proc = transformers.AutoTokenizer.from_pretrained(cfg["model_dir"], trust_remote_code=self.trc)
            self.tok = self.proc
        else:
            self.proc = transformers.AutoProcessor.from_pretrained(cfg["model_dir"], trust_remote_code=self.trc)
            self.tok = getattr(self.proc, "tokenizer", self.proc)
        if getattr(self.tok, "pad_token", None) is None and getattr(self.tok, "eos_token", None):
            self.tok.pad_token = self.tok.eos_token
        try:
            self.tok.padding_side = "left"
        except AttributeError:
            pass

    def _conv(self, body: dict) -> list[dict]:
        system, text, image = split_messages(body)
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system if self.text_only else [{"type": "text", "text": system}]})
        if self.text_only:
            msgs.append({"role": "user", "content": text})
        else:
            content = []
            if image:
                content.append({"type": "image", "url": image})
            content.append({"type": "text", "text": text})
            msgs.append({"role": "user", "content": content})
        return msgs

    def generate(self, batch: list[dict]) -> list[dict]:
        torch = self.torch
        convs = [self._conv(t["body"]) for t in batch]
        ctk = batch[0]["body"].get("chat_template_kwargs") or {}
        max_new = max(int(t["body"].get("max_tokens", 4096)) for t in batch)
        if self.text_only:
            texts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, **ctk) for c in convs]
            inputs = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        else:
            inputs = self.proc.apply_chat_template(convs, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt", padding=True, **ctk)
            inputs = inputs.to(self.model.device)
        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        dt = time.time() - t0
        n_in = inputs["input_ids"].shape[1]
        res = []
        eos = self.tok.eos_token_id
        for i in range(len(batch)):
            gen = out[i, n_in:]
            pad = self.tok.pad_token_id
            ids = [int(x) for x in gen.tolist() if x != pad]
            text = self.tok.decode(ids, skip_special_tokens=True)
            finish = "length" if len(gen) >= max_new and (not ids or ids[-1] != eos) else "stop"
            res.append({"status": "ok", "text": text, "finish_reason": finish,
                        "prompt_tokens": int(inputs["attention_mask"][i].sum()), "completion_tokens": len(ids),
                        "latency_s": round(dt, 3)})
        return res


# --------------------------------------------------------------------------------------- InternVL / Vintern
class InternVLChat:
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def __init__(self, cfg: dict):
        import torch
        import transformers

        self.torch = torch
        hf = cfg["hf"]
        self.size = int(hf.get("image_size", 448))
        self.max_tiles = int(hf.get("max_num_tiles") or 6)
        dtype = getattr(torch, hf.get("dtype", "bfloat16"))
        self.dtype = dtype
        self.model = transformers.AutoModel.from_pretrained(cfg["model_dir"], dtype=dtype, trust_remote_code=True,
                                                            low_cpu_mem_usage=True).eval().cuda()
        self.tok = transformers.AutoTokenizer.from_pretrained(cfg["model_dir"], trust_remote_code=True, use_fast=False)

    def _transform(self, img):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        return T.Compose([T.Resize((self.size, self.size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(),
                          T.Normalize(self.MEAN, self.STD)])(img)

    def _tiles(self, img):
        """Dynamic tiling: choose the grid (i x j, i*j <= max_tiles) closest to the image aspect ratio."""
        w, h = img.size
        ar = w / h
        ratios = sorted({(i, j) for n in range(1, self.max_tiles + 1) for i in range(1, n + 1)
                         for j in range(1, n + 1) if 1 <= i * j <= self.max_tiles}, key=lambda x: x[0] * x[1])
        best, best_diff = (1, 1), float("inf")
        for r in ratios:
            diff = abs(ar - r[0] / r[1])
            if diff < best_diff or (diff == best_diff and w * h > 0.5 * self.size * self.size * r[0] * r[1]):
                best, best_diff = r, diff
        tw, th = self.size * best[0], self.size * best[1]
        resized = img.resize((tw, th))
        tiles = [resized.crop(((k % best[0]) * self.size, (k // best[0]) * self.size,
                               (k % best[0] + 1) * self.size, (k // best[0] + 1) * self.size))
                 for k in range(best[0] * best[1])]
        if len(tiles) != 1:
            tiles.append(img.resize((self.size, self.size)))
        return tiles

    def generate(self, batch: list[dict]) -> list[dict]:
        from PIL import Image

        torch = self.torch
        pvs, nums, questions = [], [], []
        for t in batch:
            system, text, image = split_messages(t["body"])
            if image:
                with Image.open(image) as im:
                    tiles = self._tiles(im.convert("RGB"))
                pv = torch.stack([self._transform(x) for x in tiles])
                pvs.append(pv)
                nums.append(pv.shape[0])
                questions.append("<image>\n" + (system + "\n\n" if system else "") + text)
            else:
                questions.append((system + "\n\n" if system else "") + text)
        max_new = max(int(t["body"].get("max_tokens", 4096)) for t in batch)
        gen_cfg = {"max_new_tokens": max_new, "do_sample": False}
        t0 = time.time()
        with torch.inference_mode():
            if pvs:
                pixel = torch.cat(pvs).to(self.dtype).cuda()
                answers = self.model.batch_chat(self.tok, pixel, num_patches_list=nums, questions=questions,
                                                generation_config=gen_cfg)
            else:
                answers = [self.model.chat(self.tok, None, q, gen_cfg) for q in questions]
        dt = time.time() - t0
        out = []
        for a in answers:
            n = len(self.tok(a)["input_ids"])
            out.append({"status": "ok", "text": a, "finish_reason": "length" if n >= max_new - 2 else "stop",
                        "completion_tokens": n, "latency_s": round(dt, 3)})
        return out


# --------------------------------------------------------------------------------------- MiniCPM-V
class MiniCPMV:
    def __init__(self, cfg: dict):
        import torch
        import transformers

        self.torch = torch
        dtype = getattr(torch, cfg["hf"].get("dtype", "bfloat16"))
        self.model = transformers.AutoModel.from_pretrained(cfg["model_dir"], trust_remote_code=True, dtype=dtype,
                                                            attn_implementation="sdpa").eval().cuda()
        self.tok = transformers.AutoTokenizer.from_pretrained(cfg["model_dir"], trust_remote_code=True)

    def generate(self, batch: list[dict]) -> list[dict]:
        from PIL import Image

        out = []
        for t in batch:
            system, text, image = split_messages(t["body"])
            content = []
            if image:
                content.append(Image.open(image).convert("RGB"))
            content.append(((system + "\n\n") if system else "") + text)
            ctk = t["body"].get("chat_template_kwargs") or {}
            t0 = time.time()
            with self.torch.inference_mode():
                a = self.model.chat(msgs=[{"role": "user", "content": content}], tokenizer=self.tok,
                                    enable_thinking=bool(ctk.get("enable_thinking", False)), sampling=False,
                                    max_new_tokens=int(t["body"].get("max_tokens", 4096)))
            out.append({"status": "ok", "text": a if isinstance(a, str) else str(a), "finish_reason": "stop",
                        "latency_s": round(time.time() - t0, 3)})
        return out


ADAPTERS = {"generic": Generic, "internvl_chat": InternVLChat, "minicpmv": MiniCPMV}


# imports that remote code only makes inside optional code paths (Molmo tensorflow resize, MPT triton attention);
# transformers' regex import check would refuse to load the model. Same list as vllm_plugins/smv_vllm_patches.
OPTIONAL_REMOTE_IMPORTS = frozenset({"tensorflow", "triton_pre_mlir"})


def relax_remote_import_check() -> None:
    from transformers import dynamic_module_utils as dmu

    orig = dmu.get_imports
    if not getattr(orig, "_smv_relaxed", False):
        def get_imports(filename):
            return [m for m in orig(filename) if m.split(".")[0] not in OPTIONAL_REMOTE_IMPORTS]

        get_imports._smv_relaxed = True
        dmu.get_imports = get_imports


def main(cfg_path: str) -> int:
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    import torch

    relax_remote_import_check()

    torch.set_num_threads(int(cfg.get("num_threads", 10)))
    tasks = load_tasks(cfg["tasks"])
    hf = cfg["hf"]
    adapter = ADAPTERS.get(hf.get("adapter", "generic"), Generic)
    log(f"loading {cfg['model_dir']} with adapter {adapter.__name__}")
    t0 = time.time()
    try:
        engine = adapter(cfg)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        log(f"LOAD_FAILED {type(e).__name__}: {e}")
        return 2
    log(f"loaded in {time.time() - t0:.0f}s; {len(tasks)} tasks")
    ladder = [b for b in hf.get("batch_ladder", [4, 2, 1]) if b <= int(hf.get("batch_size", 4))] or [1]
    bi = 0
    i = 0
    with open(cfg["out"], "a", encoding="utf-8") as out:
        while i < len(tasks):
            bs = ladder[bi]
            batch = tasks[i:i + bs]
            try:
                res = engine.generate(batch)
                for t, r in zip(batch, res):
                    emit(out, {"id": t["id"], **r})
                i += len(batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bi + 1 < len(ladder):
                    bi += 1
                    log(f"OOM: batch size -> {ladder[bi]}")
                    continue
                for t in batch:
                    emit(out, {"id": t["id"], "status": "error", "text": "", "error": "CUDA out of memory at batch 1"})
                i += len(batch)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                for t in batch:
                    emit(out, {"id": t["id"], "status": "error", "text": "", "error": f"{type(e).__name__}: {str(e)[:500]}"})
                i += len(batch)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
