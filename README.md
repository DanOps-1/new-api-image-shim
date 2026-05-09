# new-api-image-shim

A small reverse proxy that sits in front of an OpenAI-compatible
`/chat/completions` endpoint and fixes two practical problems:

1. **Image responses don't render.** Some upstreams (OpenRouter / Roo
   style) return generated images in a non-standard
   `message.images[]` field. OpenAI clients only render images that
   are inlined into `message.content` as markdown. The shim rewrites
   `message.images[]` into `![image](url)` markdown links inside
   `content`, on both blocking and SSE responses.
2. **Slow image generations get killed by Cloudflare 524.** Cloudflare
   closes idle edge connections after 100 s. For non-streaming
   requests the shim flushes `HTTP 200` + `Transfer-Encoding:
   chunked` immediately and trickles whitespace until the upstream
   replies. For streaming requests it injects empty SSE
   `chat.completion.chunk` heartbeats while the upstream is silent.

It also strips inline `data:image/*;base64,...` payloads out of
**prior** messages in the history before forwarding upstream
(playground-style clients otherwise accumulate every previously
generated image and blow up the prompt token budget). The latest
message is left untouched, so a fresh image-input request still
works.

Everything else passes through unchanged.

## Run

Docker:

```bash
docker build -t new-api-image-shim .
docker run --rm -p 3100:3100 \
  -e UPSTREAM=http://your-openai-compatible-host:port \
  new-api-image-shim
```

Direct:

```bash
pip install fastapi==0.115.0 'uvicorn[standard]==0.32.0' httpx==0.27.2
UPSTREAM=http://localhost:3000 \
  uvicorn shim:app --host 0.0.0.0 --port 3100 --no-access-log
```

Then point your OpenAI-compatible client at `http://<shim-host>:3100`
instead of the upstream directly.

## Endpoints

| Method | Path                    | Behavior                                |
|--------|-------------------------|-----------------------------------------|
| POST   | `/v1/chat/completions`  | Transformed (streaming + non-streaming) |
| POST   | `/pg/chat/completions`  | Transformed (streaming + non-streaming) |
| GET    | `/healthz`              | `{"ok": true}`                          |

Any other path returns 404. Add routes in `shim.py` if you need them.

## Configuration

| Env var              | Default                  | Meaning                                                            |
|----------------------|--------------------------|--------------------------------------------------------------------|
| `UPSTREAM`           | `http://new-api:3000`    | Upstream OpenAI-compatible base URL (no trailing slash).           |
| `TIMEOUT`            | `600`                    | httpx total timeout for upstream requests, in seconds.             |
| `KEEPALIVE_INTERVAL` | `10`                     | Seconds between keep-alive bytes (blocking) / SSE heartbeats.      |
| `SHIM_DUMP`          | unset                    | If `1`, dumps the last streamed response to `/tmp/shim_last_response.bin` for debugging. |

## Notes

- Designed for, and tested against, [new-api](https://github.com/QuantumNous/new-api), but
  works in front of any OpenAI-compatible upstream that speaks
  `/chat/completions`.
- The keep-alive heartbeat is a real (no-op) `chat.completion.chunk`
  rather than an SSE comment line, because some browser SSE parsers
  (e.g. `sse.js` v2) coerce missing data to an empty string and then
  fail `JSON.parse('')`.
- The shim is stateless; scale horizontally if you need to.
