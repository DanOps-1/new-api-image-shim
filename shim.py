"""
Image-aware chat completions shim.

Sits in front of new-api. Transforms /v1/chat/completions and
/pg/chat/completions responses: the upstream returns generated images in
the non-standard `message.images[]` field (OpenRouter / Roo style). We
rewrite them into the standard `message.content` as markdown image links so
any OpenAI-compatible client renders them.

For non-streaming requests, we immediately send HTTP 200 with
Transfer-Encoding: chunked and emit periodic keep-alive whitespace bytes
while waiting for the upstream. This prevents Cloudflare's 100s edge
timeout (524) on slow image generations.

Everything else passes through unmodified.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

UPSTREAM = os.environ.get("UPSTREAM", "http://new-api:3000")
TIMEOUT = float(os.environ.get("TIMEOUT", "600"))
KEEPALIVE_INTERVAL = float(os.environ.get("KEEPALIVE_INTERVAL", "10"))

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [shim] %(message)s",
)
log = logging.getLogger("shim")

app = FastAPI()


def _extract_url(image_obj: Any) -> str | None:
    if not isinstance(image_obj, dict):
        return None
    iu = image_obj.get("image_url")
    if isinstance(iu, dict):
        return iu.get("url")
    if isinstance(iu, str):
        return iu
    return image_obj.get("url")


def _images_to_markdown(images: list[Any]) -> str:
    parts = []
    for img in images:
        url = _extract_url(img)
        if url:
            parts.append(f"![image]({url})")
    return "\n\n".join(parts)


def _merge(content: str | None, md: str) -> str:
    if not md:
        return content or ""
    if content:
        return f"{content}\n\n{md}"
    return md


def transform_message(msg: dict) -> dict:
    images = msg.get("images")
    if not images:
        return msg
    md = _images_to_markdown(images)
    if md:
        msg["content"] = _merge(msg.get("content"), md)
    msg.pop("images", None)
    return msg


def transform_response(data: dict) -> dict:
    for choice in data.get("choices", []) or []:
        msg = choice.get("message")
        if isinstance(msg, dict):
            transform_message(msg)
    return data


def transform_chunk(chunk: dict) -> dict:
    for choice in chunk.get("choices", []) or []:
        delta = choice.get("delta")
        if isinstance(delta, dict):
            transform_message(delta)
    return chunk


def _filter_request_headers(headers: dict[str, str]) -> dict[str, str]:
    skip = {"host", "content-length", "transfer-encoding"}
    return {k: v for k, v in headers.items() if k.lower() not in skip}


import re

# Strip inline base64 image data from message history before forwarding upstream.
# Without this, playground-style clients accumulate prior generated images as
# `![image](data:image/...;base64,...)` in conversation history, exploding the
# prompt token budget and triggering upstream errors / hangs.
_DATA_IMAGE_MD = re.compile(r"!\[[^\]]*\]\(data:image/[^;]+;base64,[^)]+\)")
_DATA_IMAGE_RAW = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+")


def _strip_inline_images_from_text(text: str) -> str:
    """Replace inline base64 images with a short placeholder."""
    if not text or "data:image/" not in text:
        return text
    text = _DATA_IMAGE_MD.sub("[image: omitted from history]", text)
    text = _DATA_IMAGE_RAW.sub("[image: omitted from history]", text)
    return text


def _strip_inline_images_from_messages(body: dict) -> tuple[dict, int]:
    """Mutate messages in body to remove inline base64 image payloads. Returns (body, bytes_saved)."""
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return body, 0
    saved = 0
    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        c = msg.get("content")
        if isinstance(c, str):
            new_c = _strip_inline_images_from_text(c)
            if new_c != c:
                saved += len(c) - len(new_c)
                msg["content"] = new_c
        elif isinstance(c, list):
            for item in c:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image_url":
                    iu = item.get("image_url")
                    url = iu.get("url") if isinstance(iu, dict) else iu
                    if isinstance(url, str) and url.startswith("data:image/"):
                        saved += len(url)
                        if isinstance(iu, dict):
                            iu["url"] = "[image: omitted from history]"
                        else:
                            item["image_url"] = "[image: omitted from history]"
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    new_t = _strip_inline_images_from_text(item["text"])
                    if new_t != item["text"]:
                        saved += len(item["text"]) - len(new_t)
                        item["text"] = new_t
    return body, saved


async def handle_chat_completions(request: Request, path: str):
    """Shared handler for /v1/chat/completions and /pg/chat/completions."""
    raw = await request.body()
    headers = _filter_request_headers(dict(request.headers))

    try:
        body = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        body = {}
    is_stream = bool(body.get("stream"))
    model = body.get("model", "?")
    original_size = len(raw)

    # Strip inline base64 images from message history to prevent prompt-token explosion.
    # Only the LAST message keeps its content untouched (so a fresh image-input request
    # still works); earlier messages get their inline images replaced with a placeholder.
    body, saved = _strip_inline_images_from_messages(body)
    if saved > 0:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        log.info(f"-> {path} model={model} stream={is_stream} body={original_size}B "
                 f"-> {len(raw)}B (stripped {saved}B of inline images from history)")
    else:
        log.info(f"-> {path} model={model} stream={is_stream} body={original_size}B")

    upstream_url = f"{UPSTREAM}/{path}"

    if is_stream:
        return await _proxy_stream(upstream_url, raw, headers, path, model)
    return await _proxy_blocking_with_keepalive(upstream_url, raw, headers, path, model)


async def _proxy_stream(
    upstream_url: str, raw: bytes, headers: dict, path: str, model: str
) -> StreamingResponse:
    """
    SSE pass-through with image transformation per chunk.

    Uses two-coroutine pattern: a producer reads upstream lines into a queue,
    a consumer yields them to the client and emits SSE comment heartbeats
    every KEEPALIVE_INTERVAL seconds while the queue is idle.
    """
    client = httpx.AsyncClient(timeout=TIMEOUT)
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()
    started = time.monotonic()

    async def producer():
        try:
            async with client.stream("POST", upstream_url, content=raw, headers=headers) as upstream:
                async for line in upstream.aiter_lines():
                    await queue.put(line)
        except Exception as e:
            await queue.put(("__error__", str(e)))
        finally:
            await queue.put(SENTINEL)

    # Optional debug dump: set SHIM_DUMP=1 in env to capture last response.
    import os as _os
    dump_path = "/tmp/shim_last_response.bin" if _os.environ.get("SHIM_DUMP") == "1" else None
    dump_fp = open(dump_path, "wb") if dump_path else None

    async def streamer() -> AsyncIterator[bytes]:
        # Flush an empty OpenAI-format chunk immediately so client sees TTFB ~0.
        # We use a real (but no-op) data frame instead of an SSE comment line
        # because sse.js v2's parseEventChunk coerces null data to "" via
        # `i.data = n.data || ""`, which then fires the 'message' listener with
        # an empty string — breaking JSON.parse(e.data) downstream.
        first_bytes = (
            b'data: {"id":"shim-start","object":"chat.completion.chunk",'
            b'"choices":[{"index":0,"delta":{}}]}\n\n'
        )
        if dump_fp: dump_fp.write(first_bytes); dump_fp.flush()
        yield first_bytes
        producer_task = asyncio.create_task(producer())
        chunks_out = 0
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_INTERVAL)
                except asyncio.TimeoutError:
                    yield (b'data: {"id":"shim-keepalive","object":"chat.completion.chunk",'
                           b'"choices":[{"index":0,"delta":{}}]}\n\n')
                    continue
                if item is SENTINEL:
                    break
                if isinstance(item, tuple) and item and item[0] == "__error__":
                    err = item[1]
                    log.warning(f"   stream upstream error: {err}")
                    payload = json.dumps({"error": {"message": err, "type": "shim_upstream_error"}})
                    yield f"data: {payload}\n\n".encode("utf-8")
                    yield b"data: [DONE]\n\n"
                    break
                line = item
                if not line:
                    yield b"\n"
                    continue
                # SSE protocol: each non-empty line ends with \n, empty line = event boundary.
                # We must NOT add \n\n to data lines, because the upstream's blank line
                # already supplies the terminator. Doubling it produces \n\n\n which
                # parsers like sse.js mis-interpret as an empty event.
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        out = (line + "\n").encode("utf-8")
                        if dump_fp: dump_fp.write(out); dump_fp.flush()
                        yield out
                        continue
                    try:
                        chunk = json.loads(payload)
                        chunk = transform_chunk(chunk)
                        out = f"data: {json.dumps(chunk, ensure_ascii=False)}\n".encode("utf-8")
                        if dump_fp: dump_fp.write(out); dump_fp.flush()
                        yield out
                        chunks_out += 1
                    except json.JSONDecodeError:
                        out = (line + "\n").encode("utf-8")
                        if dump_fp: dump_fp.write(out); dump_fp.flush()
                        yield out
                else:
                    # Empty line or non-data line — preserve as-is (single \n terminator)
                    out = (line + "\n").encode("utf-8")
                    if dump_fp: dump_fp.write(out); dump_fp.flush()
                    yield out
        finally:
            elapsed = time.monotonic() - started
            log.info(f"<- {path} stream model={model} chunks={chunks_out} took={elapsed:.2f}s")
            if dump_fp:
                dump_fp.close()
            if not producer_task.done():
                producer_task.cancel()
            try:
                await producer_task
            except (asyncio.CancelledError, Exception):
                pass
            await client.aclose()

    return StreamingResponse(streamer(), media_type="text/event-stream")


async def _proxy_blocking_with_keepalive(
    upstream_url: str, raw: bytes, headers: dict, path: str = "", model: str = "?"
) -> StreamingResponse:
    """
    Non-streaming JSON response, but emitted via Transfer-Encoding: chunked
    so we can flush HTTP 200 immediately and trickle whitespace until the
    upstream finishes. Defeats Cloudflare's 100s 524 timeout.

    JSON parsers ignore leading whitespace, so prepending newlines is safe.
    """
    client = httpx.AsyncClient(timeout=TIMEOUT)
    started = time.monotonic()

    async def runner() -> AsyncIterator[bytes]:
        # Flush headers immediately so CF/nginx see TTFB ~0
        yield b"\n"

        upstream_task = asyncio.create_task(
            client.post(upstream_url, content=raw, headers=headers)
        )
        try:
            while True:
                try:
                    upstream = await asyncio.wait_for(
                        asyncio.shield(upstream_task), timeout=KEEPALIVE_INTERVAL
                    )
                    break
                except asyncio.TimeoutError:
                    yield b" "  # keep CF/nginx connection warm
        except Exception:
            upstream_task.cancel()
            try:
                await upstream_task
            except Exception:
                pass
            raise

        try:
            data = upstream.json()
            data = transform_response(data)
            yield json.dumps(data, ensure_ascii=False).encode("utf-8")
        except json.JSONDecodeError:
            yield upstream.content
        finally:
            elapsed = time.monotonic() - started
            log.info(f"<- {path} blocking model={model} took={elapsed:.2f}s")
            await client.aclose()

    return StreamingResponse(runner(), media_type="application/json")


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: Request):
    return await handle_chat_completions(request, "v1/chat/completions")


@app.post("/pg/chat/completions")
async def pg_chat_completions(request: Request):
    return await handle_chat_completions(request, "pg/chat/completions")


@app.get("/healthz")
async def healthz():
    return {"ok": True}
