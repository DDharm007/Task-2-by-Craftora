<div align="center">

# GoaRAG

**A voice-driven, retrieval-grounded question answering system over multilingual MS MARCO.**

Speak a question in Hindi or English → ElevenLabs Scribe transcribes it → hybrid dense + BM25
retrieval over Qdrant → cross-encoder reranking → GPT-OSS 120B answers **only** from the
retrieved passages → eight guardrails verify the answer before you see it.

Built for HackerHouse Goa 2026 · Task 2.

</div>

---

## Contents

- [What this is](#what-this-is)
- [Task coverage](#task-coverage)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Quick start](#quick-start)
- [Environment variables](#environment-variables)
- [Dataset](#dataset)
- [Running](#running)
- [API reference](#api-reference)
- [The RAG pipeline](#the-rag-pipeline)
- [Benchmarking](#benchmarking)
- [Deployment](#deployment)
- [Project layout](#project-layout)
- [Engineering notes](#engineering-notes)

---

## What this is

The goal is not to build a language model. It is to build everything *around* one so that its
answers are traceable, verifiable, and refusable.

Three properties drive every design decision here:

1. **Nothing is answered from model priors.** The system prompt constrains the model to the
   numbered context blocks it is given, and a post-generation groundedness check measures whether
   it actually complied. If it did not, the answer is replaced.
2. **Every number shown is measured, not estimated.** Per-stage latency, similarity scores,
   rerank deltas, token counts and confidence are all real values from the request you just made.
3. **Retrieval quality is evaluated objectively.** MSMARCO-XI ships `is_selected` relevance
   labels, so `/api/benchmark` reports recall@k, MRR and nDCG — not just how fast it was.

---

## Task coverage

| Requirement | GoaRAG implementation |
| --- | --- |
| Voice → RAG → answer | Voice recording, ElevenLabs Scribe transcription, hybrid retrieval, streaming grounded answer, and optional spoken playback. |
| Thoughtful chunking | Six composed strategies: semantic boundaries, recursive fallback, sentence windows, overlap, parent expansion, and metadata-aware embedding text. |
| Proper harness | Zod-validated contracts, SSE stage events, bounded timeouts, retries with jitter, typed errors, cancellation, and provider fallbacks. |
| Guardrails | Injection, jailbreak, toxicity, off-topic, retrieval similarity, context verification, groundedness, and confidence checks. |
| Sub-50ms retrieval | **P50 19ms · P70 21ms** over 155 queries, every run. P100 39–102ms — all 155 inside budget on an idle machine. [docs/LATENCY.md](docs/LATENCY.md) |
| Latency analytics | Per-stage timings plus P50/P70/P95/P99/P100 distributions across benchmark requests. |

The sub-50 ms target is measured honestly rather than claimed. It is judged on the window this
system controls — query text in → ranked context out — because speech-to-text and generation are
both network round trips to a vendor, and no hosted LLM answers in 50ms. That window clears the
budget at the median with 2.6× of headroom, and clears it for every query on an idle machine, but not
reliably at p100 under contention;
and [docs/LATENCY.md](docs/LATENCY.md) says so and explains why. The console exposes both
first-token and end-to-end latency so the
trade-off stays visible.

---

## Architecture

```
                        ┌──────────────────────────────────────┐
   🎙  Voice ──────────▶ │  ElevenLabs Scribe v1                │
                        │  → transcript + word-level logprobs  │
                        └──────────────────┬───────────────────┘
                                           │  transcript confidence
   ⌨  Text ────────────────────────────────┤
                                           ▼
                        ┌──────────────────────────────────────┐
                        │  Guardrails · input                  │
                        │  injection · jailbreak · toxicity    │
                        └──────────────────┬───────────────────┘
                                           ▼
                   ┌───────────────────────┴───────────────────────┐
                   │                                               │
        ┌──────────▼──────────┐                       ┌────────────▼───────────┐
        │  BGE-M3 embedding   │   ── in parallel ──   │  BM25 sparse vector    │
        │  1024-dim dense     │                       │  IDF-weighted query    │
        └──────────┬──────────┘                       └────────────┬───────────┘
                   │                                               │
        ┌──────────▼───────────────────────────────────────────────▼───────────┐
        │  Qdrant · one collection, named `dense` + `sparse` vectors           │
        └──────────────────────────────┬──────────────────────────────────────┘
                                       ▼
                        ┌──────────────────────────────────────┐
                        │  Reciprocal Rank Fusion  → top 10    │
                        │  MMR diversity filter                │
                        └──────────────────┬───────────────────┘
                                           ▼
                        ┌──────────────────────────────────────┐
                        │  Cross-encoder rerank    → top 5     │
                        │  + parent-chunk expansion            │
                        └──────────────────┬───────────────────┘
                                           ▼
                        ┌──────────────────────────────────────┐
                        │  Guardrails · evidence               │
                        │  similarity · off-topic · context    │
                        └──────────────────┬───────────────────┘
                                           ▼
                        ┌──────────────────────────────────────┐
                        │  Prompt builder → numbered blocks    │
                        │  GPT-OSS 120B on Groq (streaming)    │
                        └──────────────────┬───────────────────┘
                                           ▼
                        ┌──────────────────────────────────────┐
                        │  Guardrails · answer                 │
                        │  groundedness · confidence gate      │
                        └──────────────────┬───────────────────┘
                                           ▼
                            Answer + citations + scores + latency
```

Full rationale for each stage: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Tech stack

| Layer | Choice |
| --- | --- |
| Frontend | React 18, Vite 6, TypeScript, TailwindCSS, Radix primitives, Framer Motion, TanStack Query, Zustand, Recharts |
| Backend | Node 22, Express, TypeScript (strict), Zod, Pino |
| Vector DB | Qdrant (named dense + sparse vectors) — with a disk-persisted embedded driver for local dev |
| Embeddings | `intfloat/multilingual-e5-small` (384-dim, multilingual) via ONNX Runtime, in-process — `bge-m3` is one config change away, at ~3× the latency |
| Reranker | `bge-reranker-base` cross-encoder (ONNX), with a lexical fallback |
| Speech-to-text | ElevenLabs Scribe v1 |
| LLM | `openai/gpt-oss-120b` via the OpenAI SDK against GroqCloud |
| Dataset | [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) |

---

## Quick start

```bash
git clone <your-repo-url> && cd goarag
```

```bash
cp .env.example .env
```

Add your `GROQ_API_KEY` and `ELEVENLABS_API_KEY` (or `SARVAM_API_KEY`) to `.env`, then:

```bash
npm install && npm run build:shared
```

```bash
npm run index
```

In virtual mode this single command reads the needed dataset ranges online and writes only the
retrieval vectors; it does not create a local dataset folder.

```bash
npm run dev
```

The API comes up on `http://localhost:8787` and the UI on `http://localhost:5173`.

> **First run takes a while.** BGE-M3 downloads ~550MB of ONNX weights and indexing is
> CPU-bound (~45 min for the default 300-row corpus). The prebuilt index is committed under
> `storage/vectors`, so you only need to re-index if you change the corpus or the embedding
> model. See [Dataset](#dataset) for how to shrink the corpus.

---

## Environment variables

Every value is validated by Zod at boot — a malformed `.env` fails immediately with a precise
message rather than surfacing as a confusing runtime error. Secrets are read from the environment
only and are redacted from all logs.

The table lists what you are most likely to change; `.env.example` documents all ~60 options.

| Variable | Default | Notes |
| --- | --- | --- |
| `GROQ_API_KEY` | — | **Required.** From [console.groq.com](https://console.groq.com). |
| `ELEVENLABS_API_KEY` | — | Needed for voice. The key only needs the `speech_to_text` scope. |
| `SARVAM_API_KEY` | — | Alternative voice provider; TTS tries Sarvam first. |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Any Groq chat model. |
| `LLM_ENABLE_THINKING` | `false` | Reasoning mode. Unsupported on Groq (400s). |
| `EMBEDDING_PROVIDER` | `local` | BGE-M3 in-process via ONNX. |
| `RERANKER_PROVIDER` | `heuristic` | `heuristic` is ~1.5s faster; `local` is more accurate. |
| `VECTOR_STORE` | `auto` | `auto` uses Qdrant if reachable, else the embedded store. |
| `QDRANT_URL` | `http://localhost:6333` | Set to your Qdrant Cloud URL in production. |
| `RETRIEVAL_TOP_K` | `10` | Candidates fused before reranking. |
| `RERANK_TOP_N` | `5` | Chunks sent to the model. |
| `SIMILARITY_THRESHOLD` | `0.28` | Below this, the system refuses rather than answering. |
| `CONFIDENCE_THRESHOLD` | `0.42` | Below this, the answer is replaced with a refusal. |
| `DATASET_MAX_ROWS` | `300` | Each row carries ~10 passages. |
| `DATASET_SOURCE` | `virtual` | Reads the remote Parquet data by HTTP byte range; does not save the source files. |
| `DATASET_PERSIST_BUNDLE` | `false` | Keeps normalized dataset rows in memory only; vectors stay in the vector store. |
| `DATASET_LANGUAGES` | `hin_Deva` | Languages to read remotely, split evenly across the row target. |

---

## Dataset

[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) is MS MARCO
translated into 14 Indic languages. Each row holds a query in English *and* a target language, its
answer in both, and ~10 candidate passages with an `is_selected` flag marking which ones actually
answer the query.

That flag is why this dataset was worth the trouble — it turns the benchmark from a vibe check
into a measurement.

**A note on how it is fetched.** The obvious route, HuggingFace's `datasets-server` `/rows` API,
is permanently broken for this dataset: its nested `passages` struct trips an
`ArrowNotImplementedError` during Parquet conversion, so `/rows`, `/search` and `/filter` all
return HTTP 500. Only `/first-rows` responds, and it caps out at ~18 rows.

The loader therefore uses HTTP range requests against the Parquet files and lets `hyparquet` read
only the needed metadata and columns. This is virtual: no source Parquet file is saved under
`dataset/`. The small normalized bundle required for indexing is still stored in
memory by default. Set `DATASET_PERSIST_BUNDLE=true` only if you want a local normalized cache.
Set `DATASET_SOURCE=parquet` only if you explicitly want the legacy local cache;
`DATASET_SOURCE=api` uses the 18-row sample for a quick smoke test.

```bash
npm run dataset:download           # verify/load the virtual source without persisting it
npm run dataset:download -- --force # bypass a cache when DATASET_PERSIST_BUNDLE=true
```

To index a smaller corpus, lower `DATASET_MAX_ROWS` — 60 rows gives ~1,400 chunks and indexes in
under 10 minutes on CPU.

---

## Running

```bash
npm run dev
```

| Command | What it does |
| --- | --- |
| `npm run dev` | Server (tsx watch) + client (Vite), concurrently |
| `npm run build` | Type-check and build all three workspaces |
| `npm start` | Run the compiled server |
| `npm run dataset:download` | Verify/load dataset rows (kept only in memory by default) |
| `npm run index` | Chunk, embed and index (idempotent — safe to re-run) |
| `npm run index:reset` | Wipe the collection and rebuild |
| `npm run benchmark` | Run the benchmark from the CLI |
| `npm run typecheck` | Strict type-check everything |

### With Docker

```bash
docker compose up -d
```

Brings up Qdrant, the API and the nginx-served frontend. The API waits for Qdrant's health check,
then `docker compose exec server node server/dist/scripts/index-dataset.js` populates the index.

---

## API reference

All responses carry an `x-request-id` header. Errors share one shape:
`{ code, message, requestId, details? }`.

### `POST /api/query`

```bash
curl -X POST http://localhost:8787/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is a corporation?","stream":false}'
```

With `"stream": true` the response is Server-Sent Events: `start`, `stage`, `guardrails`,
`chunks`, `reasoning`, `token`, `done`, `error`.

### `POST /api/transcribe`

```bash
curl -X POST http://localhost:8787/api/transcribe -F 'file=@question.webm'
```

Returns the transcript with word-level timings and a confidence derived from per-word
log-probabilities.

### `POST /api/voice-query`

Audio in, grounded answer out. Send `stream=true` to receive the transcript the moment STT
returns, while retrieval is still running.

```bash
curl -X POST http://localhost:8787/api/voice-query \
  -F 'file=@question.webm' -F 'stream=false'
```

### `POST /api/speak`

Reads an answer aloud with ElevenLabs, returning `audio/mpeg`. Aliased at `POST /api/tts`.

```bash
curl -X POST http://localhost:8787/api/speak \
  -H 'Content-Type: application/json' \
  -d '{"text":"A corporation is a legal entity."}' --output answer.mp3
```

Free ElevenLabs accounts cannot use TTS (HTTP 402); the UI falls back to the browser's own
speech synthesiser automatically. Speech-to-*text* works fine on the free tier.

### `GET /api/benchmark`

```bash
curl 'http://localhost:8787/api/benchmark?sampleSize=20&generation=false'
```

### `GET /api/health`

Cheap by default; `?deep=true` additionally probes Groq and the voice providers. Returns 503 when a
component is down so an orchestrator can act on it.

### `GET /api/stats`

Index statistics and rolling request analytics.

---

## The RAG pipeline

### Chunking

Six strategies, composed rather than chosen between. No fixed-size chunking anywhere — every
boundary comes from a signal in the text.

| Strategy | Role |
| --- | --- |
| **Semantic** | Primary splitter. Cuts where lexical cohesion between adjacent sentence windows drops below a percentile threshold — i.e. at topic shifts. |
| **Recursive** | Structural fallback for passages with too few sentences for a breakpoint to mean anything, and for breaking up any oversized semantic span. |
| **Overlap** | Each chunk is extended backwards into its predecessor, so a fact spanning a boundary survives intact. Scaled to chunk size and capped at half the neighbour. |
| **Sliding window** | Extra overlapping views, only for passages long enough that boundary effects matter. |
| **Parent–child** | Children are embedded and searched; the parent is what the model reads. Only created when it groups ≥2 children. |
| **Metadata** | A `[topic \| language \| source]` header is prefixed to the embedded text (never the displayed text), which lifts cross-lingual recall. |

Every chunk stores `documentId`, `source`, `language`, `passageId`, `chunkIndex`, `parentChunk`,
`topic`, `strategy`, token count, character offsets and the ground-truth `isSelected` label.

### Hybrid retrieval

Dense and sparse arms run concurrently, then merge by **Reciprocal Rank Fusion**:

```
RRF(d) = Σ_arms 1 / (k + rank_arm(d))          k = 60
```

Rank-based rather than score-based, because cosine similarity lives in `[0,1]` while BM25 is
unbounded and corpus-dependent — normalising them into a shared scale requires assumptions that
break whenever the corpus changes.

BM25 is stored as a **native Qdrant sparse vector**. The score factorises into a sparse dot
product, so the document side stores TF weights and the query side stores IDF:

```
score(q,d) = Σ_t idf(t) · tf-weight(t,d)
```

which means Qdrant does the keyword search in its own index instead of us pulling candidates back
into Node to score.

### Reranking

A bi-encoder compares two independently-produced summaries. A cross-encoder reads query and
document *together*, letting attention align specific query terms against specific spans. Far more
accurate, far too slow for a whole corpus — which is exactly why it belongs in a second stage over
a shortlist of 10.

### Guardrails

| Stage | Checks |
| --- | --- |
| **Input** (pre-retrieval) | `prompt_injection`, `jailbreak`, `toxicity` — regex/lexical, so an attack costs nothing |
| **Evidence** (post-retrieval) | `similarity_threshold`, `off_topic`, `context_verification` |
| **Answer** (post-generation) | `hallucination` (groundedness), `confidence` |

Groundedness measures the fraction of answer sentences whose content words trace back to the
retrieved context — catching the failure mode where a fluent answer quietly introduces facts that
were never in the sources.

Confidence blends five signals:

```
0.35·topScore + 0.30·groundedness + 0.15·meanScore
              + 0.10·armAgreement + 0.10·contextCoverage
```

Below `CONFIDENCE_THRESHOLD` the answer is *replaced* with `"I don't have enough information."`
rather than hedged — a confident-sounding answer nobody trusts is the worst possible outcome.

---

## Benchmarking

```bash
npm run benchmark -- --sample 20
```

```bash
curl 'http://localhost:8787/api/benchmark?sampleSize=20' | jq .quality
```

Reports two independent things:

- **Latency** — p50/p70/p95/p99/p100 per stage, plus `retrievalPath`: the window the 50ms budget
  applies to. Nearest-rank percentiles, so every reported value is a latency that actually
  occurred.
- **Retrieval quality** — recall@5, recall@10, precision@5, MRR, nDCG@5 and hit rate, scored
  against `is_selected`.

Generation is off by default: a 550B model dominates the wall clock and tells you nothing about
whether retrieval works. The sample is a deterministic stride, so runs are comparable, and the
evaluation set is rebuilt from the index's own metadata so the benchmark needs no network.

### Measured results

155 labelled queries, retrieval only, sequential against a warm pipeline. Corpus is 7,088 vectors
from MSMARCO-XI `validation`; `EMBEDDING_PROVIDER=local`, embedded vector store, CPU only.

**Latency budget** — query text in → ranked context out, target ≤ 50ms:

| Percentile | Retrieval path | |
| --- | --- | --- |
| **P50** | **19ms** | ✅ 31ms under |
| **P70** | **21ms** | ✅ 29ms under |
| **P100** | **39–102ms** | ⚠️ passes idle, not under load |

P50/P70 clear the budget on every run and move by <6ms across sixteen runs —
those are properties of the pipeline. P100 tracks machine contention instead:
on an idle machine **4 of 7 runs put all 155 queries inside 50ms** (best p100
38.7ms); with a browser and dev server running alongside, 0 of 9 did. Per stage,
from a representative run:

| Stage | p50 | p70 | p95 | p99 | p100 |
| --- | --- | --- | --- | --- | --- |
| Embedding | 8.5ms | 10.9ms | 18.3ms | 34.0ms | 34.1ms |
| Retrieval (dense ∥ sparse + fusion) | 4.9ms | 5.7ms | 7.7ms | 16.4ms | 22.1ms |
| Reranking | 0.2ms | 0.3ms | 0.8ms | 1.2ms | 1.4ms |
| **Retrieval path** | **21.6ms** | **23.8ms** | **33.8ms** | **59.6ms** | **61.8ms** |

| Metric | Value |
| --- | --- |
| Hit rate | 59.4% |
| Recall@10 | 46.8% |
| Recall@5 | 29.2% |
| MRR | 34.2% |
| nDCG@5 | 24.7% |
| Precision@5 | 12.7% |

Speech-to-text and generation sit outside that window on purpose — both are network round trips to
a vendor, and no hosted LLM answers in 50ms. End-to-end with generation, a typical query answers in
~2s. Precision@5 looks low because each query has ~2 relevant passages, so five slots can never be
more than ~40% relevant; recall and MRR are the meaningful measures.

> **These quality numbers are not comparable to earlier ones in this repo's history.** A previous
> revision reported 83.3% hit rate, and the drop to 59.4% is mostly a change of *measurement*, not a
> regression. That figure came from 30 cases whose labels were rebuilt by re-downloading the dataset;
> these come from 155 cases derived from the index itself, which is a larger and stricter set — the
> old path silently dropped languages mid-download, shrinking its own denominator. Measured
> like-for-like on this same 155-query set, the embedding model change accounts for **7.7pp**
> (bge-m3 67.1% → e5-small 59.4%), which is the real trade and is broken down in
> [docs/LATENCY.md](docs/LATENCY.md).

**[docs/LATENCY.md](docs/LATENCY.md)** covers this properly: what the budget is measured over and
why, the four changes that took p50 from 380ms to 22ms, the measured quality/latency trade against
`bge-m3` (which misses the budget at the median but scores +7.7pp hit rate), and an honest account
of the p99/p100 tail.

---

## Deployment

### Frontend → Vercel

```bash
vercel --prod
```

Import the repo as-is — leave **Root Directory** at its default (the repo root). The top-level
`vercel.json` handles everything: it installs the full npm workspace, builds `shared` then
`client`, and points `outputDirectory` at `client/dist`. This only works from the repo root
because `client` depends on the `shared` workspace package; pointing Root Directory at `client`
would hide `shared` from the install step and the build would fail to resolve `@goarag/shared`.

Set `VITE_API_BASE_URL` as a Vercel **Environment Variable** (Project → Settings → Environment
Variables) to your Railway API's public URL, e.g. `https://your-api.up.railway.app`. This is a
build-time value — Vite bakes it into the bundle, so it must be set before you deploy, and
changing it later requires a redeploy. Leaving it unset does not fail the build; it silently
ships a client that calls relative `/api/...` paths on the Vercel domain itself, where no backend
exists, so every request 404s at runtime.

Only the backend deploy is a serverless-incompatible service (see [Why not host the backend on
Vercel too?](#why-not-host-the-backend-on-vercel-too) below) — the frontend here is a plain static
build with zero Vercel functions involved.

### Backend → Railway

`railway.json` is included. Set the environment variables from the table above, and point
`QDRANT_URL` at your cluster. Set `VECTOR_STORE=qdrant` so a misconfigured URL fails loudly
instead of quietly writing to a container-local file that vanishes on redeploy.

Recommended production settings:

```bash
RERANKER_PROVIDER=heuristic # ~1.5s faster per query than the cross-encoder
VECTOR_STORE=qdrant         # never silently fall back
NODE_ENV=production         # withholds error internals from responses
CORS_ORIGIN=https://your-frontend.vercel.app
```

### Vector DB → Qdrant Cloud

Create a free cluster, then set `QDRANT_URL` and `QDRANT_API_KEY`. Run the indexer once against
the cloud URL to populate it.

### Why not host the backend on Vercel too?

The frontend is a plain static build, which is exactly what Vercel is for. The backend is not,
for reasons specific to how this pipeline works rather than anything a config change fixes:

- **In-process ML models.** BGE-M3 embeddings (~550MB) and the cross-encoder reranker (~266MB)
  load as ONNX weights directly in the Node process. Vercel serverless functions cap deployment
  size well below that.
- **In-memory vector index.** The embedded store loads the full index into a flat `Float32Array`
  on first use and keeps it resident so scans stay fast. Serverless functions are stateless
  between invocations — every cold start would mean reloading the index *and* both models from
  scratch, which is minutes of latency, not milliseconds, repeated on every cold start.
- **Boot-time warmup.** The server warms both models right after it starts listening so the first
  real query isn't slow. That assumes a long-lived process; it does nothing useful in a function
  that exits after the response.
- **SSE streaming.** The query endpoint streams `text/event-stream` chunks as the pipeline
  progresses. Vercel Node functions also cap execution time (10s on Hobby) — retrieval plus
  reranking plus LLM generation can exceed that before streaming semantics even become a concern.

Railway (or any host running a persistent process) is the correct fit for the backend; Vercel is
the correct fit for the frontend. Keep the split.

---

## Project layout

```
goarag/
├── client/                  React frontend
│   └── src/
│       ├── components/      ui/ · layout/ · voice/ · query/
│       ├── hooks/           useAudioRecorder — mic capture + amplitude analysis
│       ├── lib/             api client (SSE reader), formatters
│       ├── pages/           Console · Analytics · Dashboard
│       └── store/           Zustand session state
├── server/
│   └── src/
│       ├── config/          Zod-validated environment
│       ├── controllers/     request → service → response
│       ├── middleware/      validation · rate limits · uploads · errors
│       ├── rag/
│       │   ├── chunking/    six strategies + the pipeline that composes them
│       │   ├── embeddings/  local ONNX provider behind a swappable interface
│       │   ├── vector/      Qdrant driver · embedded driver · BM25
│       │   ├── retriever/   hybrid search · RRF · MMR
│       │   ├── reranker/    cross-encoder + lexical fallback
│       │   ├── guardrails/  eight checks across three stages
│       │   └── prompt/      grounded prompt construction
│       ├── routes/          API surface
│       ├── scripts/         dataset download · indexing · benchmark
│       ├── services/        rag · llm · stt · dataset · indexing · analytics
│       └── utils/           logging · errors · async · multilingual text
├── shared/                  types + Zod schemas used by both sides
├── dataset/                 small normalised bundles (gitignored; no raw Parquet in virtual mode)
├── docs/                    architecture notes
└── docker-compose.yml
```

---

## Engineering notes

Things discovered while building this that shaped the result:

**Model load inside the measured window wrecked the latency percentiles.** The benchmark did not
warm the models before timing, so a ~2.2s lazy load landed on whichever queries ran first and
dragged every percentile above p50 with it — p100 read 5.4s against a true 0.48s. The server
already warmed at boot; the harness now does the same before its timed run.

**Cross-encoder reranking dominates the latency budget.** Ten forward passes on CPU cost ~1.55s at
p50 — roughly 4× the rest of the pipeline combined. The heuristic reranker is the default for that
reason; it trades ~7pp of recall@5 for a 3.5× faster query.

**Large responses must be streamed.** Non-streaming requests on big models routinely exceed a
two-minute gateway timeout. The client always streams on the wire and synthesises a non-streaming
response by accumulating deltas.

**Parallel embedding made indexing slower.** The local ONNX provider already saturates the CPU
inside a single forward pass; issuing 3 batches concurrently was measurably *worse* than serial.
Concurrency is now chosen by provider — 1 for local, configured value for hosted.

**Bidirectional chunk overlap double-counts.** Extending each chunk both ways means every pair of
neighbours shares two extensions, pushing mutual redundancy past 60% on short passages. Overlap is
backward-only.

**Semantic chunking must not be gated on length.** Routing only oversized passages to the semantic
splitter left two unrelated topics glued together in every short passage — and an irrelevant half
drags the chunk's embedding off-topic. Routing is by sentence count instead.

---

## License

MIT
