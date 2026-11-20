"""Async OpenAI-compatible client: many requests in flight (semaphore), retries with backoff,
context-length fallback, and server-death detection. Each result is handed to `on_result` immediately."""
from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

_CTX_RE = re.compile(r"(\d+)\s*(?:input\s+)?tokens?\s*(?:in the messages|from the input|in the prompt|input)", re.I)
_MAXLEN_RE = re.compile(r"maximum (?:context|model) length is (\d+)", re.I)


class ServerDiedError(RuntimeError):
    pass


@dataclass
class RequestTask:
    key: dict[str, Any]                      # model_key, prompt_id, image_sha, seed, thinking
    body: dict[str, Any]                     # OpenAI chat body without "model"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClientStats:
    sent: int = 0
    ok: int = 0
    errors: int = 0
    retries: int = 0
    died: bool = False
    paused: bool = False
    pending: list[RequestTask] = field(default_factory=list)


def _extract_message(data: dict) -> tuple[str, str, str]:
    ch = (data.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return content, reasoning, ch.get("finish_reason") or ""


def _extract_logprobs(data: dict, content: str) -> dict[str, Any]:
    """Token log-probabilities of the answer (request body `logprobs: true`): parallel lists of token text, UTF-8 byte
    length and log-probability, plus `lp_off`, the byte offset of the answer text inside the concatenated tokens (0 when
    the tokens tile the text exactly, -1 when the text cannot be found, e.g. a reasoning parser rewrote it), so that a span
    of the answer text can be mapped back to the tokens that produced it."""
    ch = (data.get("choices") or [{}])[0]
    items = (ch.get("logprobs") or {}).get("content") or []
    if not items:
        return {}
    toks, nbytes, lps, joined = [], [], [], bytearray()
    for it in items:
        tok = it.get("token") or ""
        b = it.get("bytes")
        raw = bytes(b) if isinstance(b, list) else tok.encode("utf-8")
        joined += raw
        toks.append(tok)
        nbytes.append(len(raw))
        lps.append(round(float(it.get("logprob") or 0.0), 4))
    off = bytes(joined).find(content.encode("utf-8")) if content else -1
    return {"lp_tok": toks, "lp_nb": nbytes, "lp": lps, "lp_off": off}


async def run_requests(base_url: str, model: str, tasks: list[RequestTask], concurrency: int,
                       on_result: Callable[[RequestTask, dict], None], *, timeout_s: float = 1200,
                       max_retries: int = 3, backoff_s: float = 5.0, max_model_len: int | None = None,
                       server_alive: Callable[[], bool] | None = None,
                       should_pause: Callable[[], bool] | None = None,
                       should_stop: Callable[[], bool] | None = None) -> ClientStats:
    stats = ClientStats()
    queue: asyncio.Queue[RequestTask] = asyncio.Queue()
    for t in tasks:
        queue.put_nowait(t)
    died = asyncio.Event()
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4)
    async with httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout_s, connect=30), limits=limits) as cli:

        async def one(task: RequestTask) -> None:
            body = dict(task.body)
            body["model"] = model
            attempt = 0
            ctx_shrinks = 0
            while True:
                t0 = time.time()
                rec: dict[str, Any] = {"attempt": attempt + 1}
                try:
                    r = await cli.post("/v1/chat/completions", json=body)
                    if r.status_code == 400 and ctx_shrinks < 3:
                        txt = r.text
                        m_in, m_max = _CTX_RE.search(txt), _MAXLEN_RE.search(txt)
                        if m_max or "context length" in txt.lower() or "max_tokens" in txt.lower():
                            limit = int(m_max.group(1)) if m_max else (max_model_len or 8192)
                            used = int(m_in.group(1)) if m_in else int(limit * 0.5)
                            new_max = max(256, limit - used - 32)
                            if new_max < int(body.get("max_tokens", 0) or 1 << 30):
                                body["max_tokens"] = new_max
                                ctx_shrinks += 1
                                continue
                    if r.status_code >= 400:
                        raise httpx.HTTPStatusError(f"HTTP {r.status_code}: {r.text[:500]}", request=r.request,
                                                    response=r)
                    data = r.json()
                    content, reasoning, finish = _extract_message(data)
                    usage = data.get("usage") or {}
                    rec.update({"status": "ok", "text": content, "reasoning": reasoning, "finish_reason": finish,
                                "prompt_tokens": usage.get("prompt_tokens"),
                                "completion_tokens": usage.get("completion_tokens"),
                                "latency_s": round(time.time() - t0, 3), "max_tokens_used": body.get("max_tokens")})
                    if body.get("logprobs"):
                        rec.update(_extract_logprobs(data, content))
                    stats.ok += 1
                    on_result(task, rec)
                    return
                except (httpx.TransportError, httpx.HTTPStatusError, ValueError) as e:
                    is_http = isinstance(e, httpx.HTTPStatusError)
                    code = e.response.status_code if is_http else None
                    alive = server_alive() if server_alive else True
                    if not alive:
                        died.set()
                        stats.pending.append(task)
                        return
                    attempt += 1
                    if attempt > max_retries or (code is not None and 400 <= code < 500 and code != 429):
                        rec.update({"status": "error", "text": "", "error": f"{type(e).__name__}: {str(e)[:800]}",
                                    "latency_s": round(time.time() - t0, 3)})
                        stats.errors += 1
                        on_result(task, rec)
                        return
                    stats.retries += 1
                    await asyncio.sleep(backoff_s * (2 ** (attempt - 1)))

        async def worker() -> None:
            while True:
                if died.is_set() or (should_stop and should_stop()):
                    return
                if should_pause and should_pause():
                    stats.paused = True
                    return
                try:
                    task = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                stats.sent += 1
                try:
                    await one(task)
                finally:
                    queue.task_done()

        await asyncio.gather(*[worker() for _ in range(max(1, concurrency))])
    while not queue.empty():
        stats.pending.append(queue.get_nowait())
    stats.died = died.is_set()
    return stats
