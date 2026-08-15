# API reference

Base URL `http://localhost:8787` in development.

Every response carries an `x-request-id` header, which is also echoed in error bodies and appears
in the server logs — so one request can be traced end to end.

## Error shape

```json
{
  "code": "INDEX_EMPTY",
  "message": "The vector index is empty. Run `npm run index` to download and index the dataset.",
  "requestId": "0e2f1a9c-…"
}
```

`details` is included in development only; production responses never carry stack traces or
internal state.

| Code | HTTP | Meaning |
| --- | --- | --- |
| `VALIDATION_ERROR` | 400 | Request failed Zod validation. `details` lists the offending paths. |
| `GUARDRAIL_BLOCKED` | 422 | Blocked by an input guardrail. |
| `RATE_LIMITED` | 429 | Too many requests in the window. |
| `PAYLOAD_TOO_LARGE` | 413 | Audio exceeds `MAX_AUDIO_UPLOAD_MB`. |
| `UNSUPPORTED_MEDIA_TYPE` | 415 | Audio MIME type not in the allowlist. |
| `INDEX_EMPTY` | 503 | Nothing has been indexed yet. |
| `VECTOR_STORE_UNAVAILABLE` | 503 | Qdrant unreachable. |
| `STT_FAILED` | 502 | ElevenLabs rejected or failed the request. |
| `EMBEDDING_FAILED` | 502 | Embedding provider failed. |
| `LLM_FAILED` | 502 | The LLM gateway failed. |
| `LLM_TIMEOUT` | 504 | Generation exceeded `LLM_TIMEOUT_MS`. |
| `INTERNAL_ERROR` | 500 | Unhandled. |

## Rate limits

| Scope | Default | Env |
| --- | --- | --- |
| All `/api/*` | 60 / min | `RATE_LIMIT_MAX` |
| `/query`, `/benchmark` | 20 / min | `RATE_LIMIT_QUERY_MAX` |
| `/transcribe`, `/voice-query` | 12 / min | `RATE_LIMIT_TRANSCRIBE_MAX` |

Standard `RateLimit-*` headers (draft-7) are returned.

---

## `POST /api/query`

Ask a question as text.

### Request

```jsonc
{
  "query": "What is a corporation?",   // required, 2–2000 chars
  "history": [                          // optional, max 20 turns
    { "role": "user", "content": "…" },
    { "role": "assistant", "content": "…" }
  ],
  "options": {                          // all optional
    "topK": 10,                         // candidates fused before rerank (1–50)
    "rerankTopN": 5,                    // chunks sent to the model (1–20)
    "languages": ["hin_Deva"],          // restrict retrieval by language
    "enableRerank": true,
    "enableMmr": true,
    "enableParentExpansion": true,
    "enableThinking": false,            // reasoning mode — much slower
    "retrievalOnly": false              // skip generation, return chunks only
  },
  "stream": false
}
```

### Response (`stream: false`)

```jsonc
{
  "requestId": "…",
  "query": "What is a corporation?",
  "transcription": null,
  "answer": "A corporation is a company or group of people authorized to act as a single entity [1].",
  "status": "answered",                 // answered | insufficient_context | low_confidence | blocked
  "reasoning": null,
  "citations": [
    {
      "index": 1,
      "chunkId": "…",
      "documentId": "1102432-0-eng_Latn",
      "source": "ai4bharat/MSMARCO-XI/validation#1102432",
      "topic": "what is a corporation",
      "language": "eng_Latn",
      "score": 0.91,
      "snippet": "A corporation is a company…"
    }
  ],
  "chunks": [ /* RetrievedChunk[] — see below */ ],
  "confidence": {
    "overall": 0.82, "topScore": 0.91, "meanScore": 0.7,
    "retrievalAgreement": 0.6, "groundedness": 0.9, "contextCoverage": 0.7,
    "sufficient": true, "threshold": 0.42
  },
  "guardrails": {
    "passed": true, "blocked": false, "blockedBy": null,
    "results": [ { "id": "prompt_injection", "stage": "pre_retrieval", "verdict": "pass", "score": 0, "threshold": 0.8, "reason": "…", "evidence": [], "durationMs": 0.2 } ],
    "totalDurationMs": 1.4
  },
  "latency": {
    "transcription": null, "guardrailsPre": 0.4, "embedding": 180,
    "denseRetrieval": 12, "sparseRetrieval": 3, "fusion": 0.2,
    "reranking": 1400, "promptBuilding": 0.3, "generation": 2600,
    "guardrailsPost": 1.1, "total": 4210, "timeToFirstToken": null
  },
  "usage": { "promptTokens": 900, "completionTokens": 64, "reasoningTokens": 0, "totalTokens": 964 },
  "model": "openai/gpt-oss-120b",
  "providers": { "embedding": "local:onnx:BAAI/bge-m3", "vectorStore": "qdrant", "reranker": "local:cross-encoder", "llm": "…", "stt": null },
  "createdAt": "2026-08-13T…"
}
```

### `RetrievedChunk`

```jsonc
{
  "id": "…",
  "text": "A corporation is a company…",
  "metadata": {
    "documentId": "…", "source": "…", "language": "eng_Latn", "passageId": "0",
    "chunkIndex": 0, "parentChunk": null, "topic": "…", "strategy": "semantic",
    "tokenCount": 108, "charStart": 0, "charEnd": 432,
    "isSelected": true,               // dataset ground-truth relevance label
    "queryId": "1102432", "queryText": "…", "indexedAt": "…"
  },
  "denseScore": 0.776,                // cosine, null if sparse-only hit
  "sparseScore": 4.21,                // BM25, null if dense-only hit
  "fusedScore": 0.0328,               // reciprocal rank fusion
  "rerankScore": 0.913,               // cross-encoder, null if rerank disabled
  "score": 0.913,                     // final ordering score
  "rankBeforeRerank": 3,
  "rankAfterRerank": 1,               // compare with the above to see rerank's effect
  "parentText": "…",                  // wider span given to the model, if any
  "matchedBy": ["dense", "sparse"]
}
```

### Streaming (`stream: true`)

`Content-Type: text/event-stream`. Each frame is `event: <type>` plus a `data:` JSON line.

| Event | Payload |
| --- | --- |
| `start` | `{ requestId, createdAt }` |
| `stage` | `{ stage, status: 'started' \| 'completed', durationMs?, detail? }` |
| `transcript` | `{ transcription }` — voice queries only |
| `guardrails` | `{ report }` — emitted per stage as checks complete |
| `chunks` | `{ chunks }` — sent before generation begins |
| `reasoning` | `{ delta }` — thinking mode only |
| `token` | `{ delta }` — answer text |
| `done` | `{ result }` — the full `QueryResult` |
| `error` | `{ error }` |

```bash
curl -N -X POST http://localhost:8787/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is a corporation?","stream":true}'
```

The `done` event is authoritative: if the confidence gate fired, its `answer` differs from the
concatenated `token` deltas. Render `done.result.answer` as the final text.

---

## `POST /api/transcribe`

`multipart/form-data`.

| Field | Type | Notes |
| --- | --- | --- |
| `file` | file | **Required.** webm, ogg, wav, mp3, mp4, m4a, flac. |
| `languageCode` | string | Optional ISO-639-3 hint, e.g. `hin`. Omit to auto-detect. |
| `diarize` | boolean | Tag distinct speakers. |

```bash
curl -X POST http://localhost:8787/api/transcribe \
  -F 'file=@question.webm' -F 'languageCode=hin'
```

```jsonc
{
  "text": "कॉर्पोरेशन क्या है?",
  "languageCode": "hin",
  "languageProbability": 0.98,
  "durationSeconds": 2.4,
  "words": [ { "text": "कॉर्पोरेशन", "start": 0.1, "end": 0.9, "type": "word", "logprob": -0.08 } ],
  "confidence": 0.94,        // derived from per-word log-probabilities
  "provider": "elevenlabs",
  "model": "scribe_v1",
  "latencyMs": 820
}
```

---

## `POST /api/voice-query`

Audio in, grounded answer out. Same form fields as `/transcribe`, plus:

| Field | Type | Notes |
| --- | --- | --- |
| `history` | string | JSON array of conversation turns. |
| `options` | string | JSON object, same shape as `/query`'s `options`. |
| `stream` | boolean | Stream as SSE. |

With `stream=true` the `transcript` event fires as soon as STT returns — the UI shows what was
heard while retrieval is still running.

```bash
curl -X POST http://localhost:8787/api/voice-query \
  -F 'file=@question.webm' \
  -F 'stream=false' \
  -F 'options={"topK":10,"rerankTopN":5}'
```

---

## `POST /api/speak`

Reads an answer aloud, closing the voice loop. Also available at `POST /api/tts` (alias).

```jsonc
{
  "text": "A corporation is a company recognised in law as a single entity.",
  "voiceId": "priya",          // optional — Sarvam speaker or ElevenLabs voice id
  "languageCode": "hin_Deva"   // optional, defaults to SARVAM_TTS_LANGUAGE
}
```

Two providers, tried in order:

1. **Sarvam AI** (`bulbul:v3`) when `SARVAM_API_KEY` is set and the language is one Sarvam
   speaks — the 22 Indic languages plus `en-IN`. Returns `audio/wav`.
2. **ElevenLabs** for every other language, and whenever Sarvam fails. Returns `audio/mpeg`.

`languageCode` accepts what the STT reports (`hin_Deva`), a bare ISO code (`hi`/`hin`) or BCP-47
(`hi-IN`). Omitting it means the answer is spoken with `SARVAM_TTS_LANGUAGE` phonetics, which
mispronounces anything else — pass the language of the turn.

`voiceId` is matched against the configured Sarvam model's speaker list; a value that is not one
of them (an ElevenLabs voice id, or a retired `bulbul:v2` name) falls back to `SARVAM_TTS_SPEAKER`
for Sarvam and is still used verbatim for ElevenLabs.

Returns raw audio (not base64 JSON) so the browser can stream it straight into an `<audio>`
element. Response headers carry `X-TTS-Provider`, `X-TTS-Model` and `X-TTS-Latency-Ms`.

Citation markers, Markdown emphasis and code fences are stripped before synthesis — read aloud
they become noise that obscures the answer. Text over `2500` characters is truncated there.

**Latency.** Sarvam caps a request at 500 characters, so longer text is split on sentence
boundaries and the pieces are synthesised concurrently, then joined into one WAV. Wall clock is
the slowest piece rather than their sum: roughly 1s up to ~650 characters and under 3s at 1300,
against ~10.6s for a single 500-character `bulbul:v3` request. Identical text is served from an
in-memory cache (48 MB, least-recently-used) in single-digit milliseconds, so replaying an answer
costs nothing.

> Sarvam returns an **array** of clips, not one, once the input passes roughly 220 characters.
> Every entry has to be joined — reading only the first silently drops the rest of the answer.

```bash
curl -X POST http://localhost:8787/api/speak \
  -H 'Content-Type: application/json' \
  -d '{"text":"A corporation is a legal entity.","languageCode":"en-IN"}' --output answer.wav
```

> **Free ElevenLabs accounts cannot use TTS.** The API returns HTTP 402, which this endpoint
> surfaces as a typed, non-retryable error. The frontend treats that as a signal to fall back to
> the browser's built-in `speechSynthesis` rather than showing an error the user cannot fix.
> Speech-to-text is unaffected — it works on the free tier. When Sarvam is also configured, the
> 402 body carries `details.sarvamFailure` explaining why the preferred provider was skipped.

---

## `GET /api/benchmark`

| Param | Default | Notes |
| --- | --- | --- |
| `sampleSize` | 10 | 1–100 queries. |
| `generation` | false | Include the LLM. Slow. |
| `language` | — | Restrict to one dataset language. |
| `concurrency` | 2 | 1–8 parallel cases. |

```bash
curl 'http://localhost:8787/api/benchmark?sampleSize=20&generation=false' | jq .quality
```

```jsonc
{
  "recallAt5": 0.71, "recallAt10": 0.83, "precisionAt5": 0.34,
  "mrr": 0.68, "ndcgAt5": 0.72, "hitRate": 0.85,
  "evaluatedQueries": 20
}
```

Scored against the dataset's `is_selected` labels. See
[ARCHITECTURE.md §8](./ARCHITECTURE.md#8-evaluation).

---

## `GET /api/health`

`?deep=true` additionally probes Groq and the voice providers, which costs real API calls. The default is
cheap and suitable for a load-balancer probe.

Returns **503** when any component is `down`; `degraded` still returns 200 because the service can
serve traffic.

```jsonc
{
  "status": "ok",                  // ok | degraded | down
  "version": "1.0.0",
  "uptimeSeconds": 412,
  "timestamp": "…",
  "components": [
    { "name": "vector_store", "status": "ok", "detail": "qdrant · 7088 points", "latencyMs": 4 },
    { "name": "index", "status": "ok", "detail": "7,088 vectors · 5,992 documents", "latencyMs": null },
    { "name": "embeddings", "status": "ok", "detail": "local:onnx · BAAI/bge-m3", "latencyMs": null },
    { "name": "speech_to_text", "status": "ok", "detail": "elevenlabs · scribe_v1", "latencyMs": null }
  ]
}
```

---

## `GET /api/stats`

| Param | Default | Notes |
| --- | --- | --- |
| `recentLimit` | 25 | 0–200 recent requests to include. |

```jsonc
{
  "index": {
    "documents": 5992, "vectors": 7088, "chunks": 6800,
    "averageChunkSizeChars": 380, "averageChunkTokens": 96,
    "languages": [ { "language": "eng_Latn", "count": 3544 } ],
    "strategies": [ { "strategy": "semantic", "count": 4100 } ],
    "collection": "goarag_msmarco_xi", "vectorStore": "qdrant",
    "embeddingModel": "BAAI/bge-m3", "embeddingDimensions": 1024,
    "lastIndexedAt": "…", "indexed": true
  },
  "analytics": {
    "totalRequests": 42, "successfulRequests": 38, "failedRequests": 0,
    "blockedRequests": 2, "lowConfidenceRequests": 2,
    "averageLatencyMs": 3820, "averageConfidence": 0.71,
    "tokensUsed": { "promptTokens": 38000, "completionTokens": 2400, "reasoningTokens": 0, "totalTokens": 40400 },
    "latency": { "embedding": { "p50": 180, "p70": 190, "p95": 240, "p99": 310, "p100": 320, "mean": 188, "min": 160, "count": 42 } },
    "throughput": [ { "bucket": "2026-08-13T09:00", "count": 12, "averageLatencyMs": 3600 } ],
    "recent": [ /* RequestLogEntry[] */ ],
    "guardrailEvents": { "off_topic": 3 },
    "uptimeSeconds": 412
  }
}
```

Analytics are in-process and per-instance — see
[ARCHITECTURE.md §10](./ARCHITECTURE.md#10-known-limitations).
