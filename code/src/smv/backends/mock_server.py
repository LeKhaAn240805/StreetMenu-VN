"""OpenAI-compatible mock server used by tests and the smoke plan (stdlib only).

It answers /health, /v1/models, /metrics and /v1/chat/completions. For image requests it looks the
image up by sha256 in a gold map (MOCK_GOLD_MAP: json {sha256: prediction_json}) and returns that
prediction, so the full pipeline can be scored end to end. Fault injection through environment:
  MOCK_OOM_STARTS=N      first N starts print a CUDA OOM message and exit (counter file MOCK_STATE_DIR)
  MOCK_DIE_AFTER=N       the server exits after N completions (once per state dir)
  MOCK_BAD_JSON_EVERY=N  every N-th answer is broken JSON
  MOCK_REPEAT_EVERY=N    every N-th answer repeats one item 8 times
  MOCK_TRUNCATE_EVERY=N  every N-th answer is cut and finish_reason=length
  MOCK_LATENCY=seconds   artificial latency per request
  MOCK_THINK=1           wrap answers in <think>...</think> when chat_template_kwargs.enable_thinking is true
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_lock = threading.Lock()
_count = {"n": 0}


def _state(name: str) -> Path | None:
    d = os.environ.get("MOCK_STATE_DIR")
    if not d:
        return None
    Path(d).mkdir(parents=True, exist_ok=True)
    return Path(d) / name


def _bump(name: str) -> int:
    """Process-safe counter in MOCK_STATE_DIR: one file per increment (O_EXCL), so concurrent threads and
    restarted servers never lose or double-count events."""
    p = _state(name)
    if p is None:
        return 0
    with _lock:
        k = 1
        while True:
            try:
                fd = os.open(f"{p}.{k}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return k
            except FileExistsError:
                k += 1


def _sha_of_url(url: str) -> str:
    if url.startswith("file://"):
        path = url[len("file://"):]
        name = Path(path).stem
        if len(name) == 64:
            return name
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()
    return ""



def _fake_logprobs(content: str) -> list[dict]:
    """Four characters per token, digits a little less certain (enough to test the mapping from text spans to tokens)."""
    out = []
    for i in range(0, len(content), 4):
        tok = content[i:i + 4]
        out.append({"token": tok, "bytes": list(tok.encode("utf-8")),
                    "logprob": -0.5 if any(c.isdigit() for c in tok) else -0.01})
    return out


class Handler(BaseHTTPRequestHandler):
    gold: dict[str, dict] = {}
    model_name = "mock"

    def log_message(self, fmt, *args):  # quiet
        return

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/health"):
            self._json(200, {"ok": True})
        elif self.path.startswith("/v1/models"):
            self._json(200, {"data": [{"id": self.model_name}]})
        elif self.path.startswith("/metrics"):
            body = b"vllm:num_preemptions_total 0\n"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if not self.path.startswith("/v1/chat/completions"):
            self._json(404, {"error": "not found"})
            return
        lat = float(os.environ.get("MOCK_LATENCY", "0"))
        if lat:
            time.sleep(lat)
        with _lock:
            _count["n"] += 1
            k = _count["n"]
        die_after = int(os.environ.get("MOCK_DIE_AFTER", "0"))
        if die_after and k > die_after and _bump("died") == 1:
            sys.stderr.write("mock: simulated crash\n")
            sys.stderr.flush()
            os._exit(17)
        msgs = req.get("messages", [])
        sha, text_parts = "", []
        for m in msgs:
            c = m.get("content")
            if isinstance(c, list):
                for part in c:
                    if part.get("type") == "image_url":
                        sha = _sha_of_url(part["image_url"]["url"])
                    elif part.get("type") == "text":
                        text_parts.append(part["text"])
            elif isinstance(c, str):
                text_parts.append(c)
        prompt = "\n".join(text_parts)
        pred = self.gold.get(sha, {"items": []}) if sha else {"items": []}
        if "Transcribe all text" in prompt:
            lines = []
            for it in pred.get("items", []):
                prices = " ".join(dict.fromkeys(o.get("raw_price") or "" for o in it.get("offers", [])))
                lines.append(f"{it.get('raw_name', '')} {prices}".strip())
            content = json.dumps({"lines": lines}, ensure_ascii=False)
        elif "List every item on this menu" in prompt:
            content = "\n".join(f"{it.get('raw_name')} — " + ", ".join(str(o.get("raw_price")) for o in it.get("offers", []))
                                for it in pred.get("items", []))
        elif prompt.strip().startswith(("OCR", "Text Recognition", "Free OCR")):
            content = "\n".join(f"{it.get('raw_name')} {(it.get('offers') or [{}])[0].get('raw_price') or ''}"
                                for it in pred.get("items", []))
        elif not sha:
            # text-only prompt (P-Oracle, P-NL stage 2): answer with the gold whose names occur in the prompt
            def hits(p: dict) -> int:
                return sum(1 for it in p.get("items", []) if it.get("raw_name") and it["raw_name"] in prompt)
            best = max(self.gold.values(), key=hits, default={"items": []})
            content = json.dumps(best if hits(best) else {"items": []}, ensure_ascii=False)
        else:
            content = json.dumps(pred, ensure_ascii=False)
        finish = "stop"
        every = int(os.environ.get("MOCK_BAD_JSON_EVERY", "0"))
        if every and k % every == 0:
            content = "```json\n" + content[: max(10, len(content) // 2)] + ",,"
        rep = int(os.environ.get("MOCK_REPEAT_EVERY", "0"))
        if rep and k % rep == 0 and pred.get("items"):
            content = json.dumps({"items": [pred["items"][0]] * 8}, ensure_ascii=False)
        tr = int(os.environ.get("MOCK_TRUNCATE_EVERY", "0"))
        if tr and k % tr == 0:
            content, finish = content[: len(content) // 2], "length"
        ctk = (req.get("chat_template_kwargs") or {})
        if os.environ.get("MOCK_THINK") and ctk.get("enable_thinking"):
            content = "<think>\nreasoning about the menu {not json}\n</think>\n" + content
        self._json(200, {
            "id": f"mock-{k}", "object": "chat.completion", "model": self.model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish,
                         **({"logprobs": {"content": _fake_logprobs(content)}} if req.get("logprobs") else {})}],
            "usage": {"prompt_tokens": 100 + len(prompt) // 4, "completion_tokens": len(content) // 4 + 1},
        })


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--served-model-name", default="mock")
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=16384)
    args, _ = ap.parse_known_args(argv)
    oom_starts = int(os.environ.get("MOCK_OOM_STARTS", "0"))
    if oom_starts and _bump("starts") <= oom_starts:
        print("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", flush=True)
        return 1
    if not Path(args.model).exists():
        print(f"mock: model path {args.model} missing", flush=True)
        return 1
    gm = os.environ.get("MOCK_GOLD_MAP")
    if gm and Path(gm).is_file():
        Handler.gold = json.loads(Path(gm).read_text(encoding="utf-8"))
    Handler.model_name = args.served_model_name
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock server on {args.host}:{args.port} max_num_seqs={args.max_num_seqs}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
