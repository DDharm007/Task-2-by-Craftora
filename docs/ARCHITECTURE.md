# Architecture

Why each part of VoxRAG is built the way it is. The README covers *what* the system does; this
document covers *why*, including the decisions that went the other way first.

---

## 1. The central constraint

A retrieval-augmented system has exactly one job that a bare LLM cannot do: **be checkable**. If a
user cannot tell which passage produced a claim, or whether the model invented it, the retrieval
layer has added latency and nothing else.

Every design decision below follows from that. Where a choice traded accuracy for
inspectability, inspectability won.

---

## 2. Provider abstraction

Three subsystems have two implementations each, behind one interface:

| Subsystem | Primary | Alternative | Selected by |
| --- | --- | --- | --- |
| Embeddings | `BAAI/bge-m3` via ONNX, in-process | NVIDIA NIM hosted | `EMBEDDING_PROVIDER` |
| Vector store | Qdrant | Disk-persisted embedded store | `VECTOR_STORE` |
| Reranker | `bge-reranker-base` cross-encoder | Lexical coverage + proximity | `RERANKER_PROVIDER` |

This is not indecision. Each pair covers a real deployment split:

- **Embeddings.** NVIDIA lists `baai/bge-m3` in `/v1/models`, but the NIM returns HTTP 500 for
  every request shape tried (array input, string input, with and without `input_type`/`truncate`).
  Running BGE-M3 locally is therefore both the spec-faithful option *and* the working one. The
  hosted provider uses `nvidia/nv-embedqa-e5-v5`, which is also 1024-dimensional and so requires
  no schema change to swap.

- **Vector store.** Qdrant is the production target and the only one used when
  `VECTOR_STORE=qdrant`. The embedded driver exists so the full pipeline runs with no external
  services — `npm run dev` works on a laptop without Docker, and CI can index and query. It
  implements the same interface, so nothing above it changes.

- **Reranker.** The cross-encoder is a ~266MB download. The lexical fallback keeps the pipeline
  working offline and is what a reranker failure degrades *to*, rather than failing the request.

The embedded vector store deserves a note on its dense search: it is an **exact brute-force cosine
scan**, not an ANN index. At the corpus sizes it targets (tens of thousands of vectors) exact
search is both faster and more accurate than building an HNSW graph — and it means the benchmark
measures retrieval quality with no recall loss from approximate search muddying the numbers.

---

## 3. Chunking

### Why not fixed-size

Fixed-size chunking cuts mid-sentence and mid-fact. The resulting embedding is a blend of two
unrelated ideas, which is precisely the vector that matches nothing well.

### How the strategies compose

Chunking is one pass per passage that routes through several strategies rather than picking one:

```
passage
  │
  ├─ < 3 sentences ──▶ recursive        (no breakpoint would be meaningful)
  │
  └─ ≥ 3 sentences ──▶ semantic         (cut at topic shifts)
                          │
                          └─ oversized span ──▶ recursive (structural split)
                          │
                          ▼
                       overlap          (extend backwards into predecessor)
                          │
                          ▼
                    parent grouping     (only when it groups ≥2 children)
                          │
                          ▼
                    sliding windows     (only if passage > 2× window)
                          │
                          ▼
                    metadata header     (prefixed to embedded text only)
```

### Two decisions that were wrong first

**Routing on token count.** The first version sent a passage to the semantic splitter only when it
exceeded the target size. MS MARCO passages usually sit *under* the budget while still covering two
unrelated ideas — a passage about corporate law that ends with four sentences about hurricanes
stayed as one chunk, and the irrelevant half dragged its embedding off-topic. Routing is now by
sentence count, so semantic chunking is the default path rather than an exception.

**Bidirectional overlap.** Extending each chunk both forwards and backwards means every pair of
neighbours shares *two* extensions. On short passages that pushed mutual redundancy past 60% —
double the configured budget — inflating the index and letting one passage occupy every top-k
slot. Overlap is now backward-only, scaled to the chunk's own size and capped at half the
neighbour.

### Semantic breakpoints without a second model

A textbook semantic chunker embeds every sentence, which is an extra forward pass per sentence
across the whole corpus. Instead we measure **lexical cohesion**: Jaccard distance over content
tokens between adjacent ±1 sentence windows, cut at a percentile threshold. Same breakpoint rule
as the embedding-based approach, O(n) per document, and it tracks topic shifts in MS MARCO
passages well in practice.

### The metadata header

Each chunk is embedded as `[topic | language | source]\n<text>` but *displayed* and sent to the
LLM as plain text. The header lifts cross-lingual recall — a Hindi query can reach an English
passage through the shared topic line — without polluting the answer context or the BM25 term
statistics (which index the display text only).

---

## 4. Hybrid retrieval

### Why both arms

Dense embeddings are strong at paraphrase and cross-lingual matching, weak at rare literal tokens
— model numbers, proper nouns, transliterated names. BM25 is the mirror image. Neither is
sufficient; their *agreement* is a stronger signal than either alone, and we use exactly that as a
confidence input.

Measured on this corpus, BGE-M3 places a Hindi translation of a passage at 0.61 cosine against an
English query, versus 0.39 for an unrelated English passage — the cross-lingual signal is real and
well-separated.

### Why fuse on rank, not score

Cosine similarity is bounded in `[0,1]`. BM25 is unbounded and depends on corpus statistics.
Normalising them onto a shared scale means picking a normalisation, and every choice
(min-max, z-score, sigmoid) breaks when the corpus or query distribution shifts.

Reciprocal Rank Fusion sidesteps this entirely by discarding magnitudes:

```
RRF(d) = Σ_arms 1 / (k + rank_arm(d))          k = 60
```

`k = 60` (Cormack et al.) damps the influence of the very top ranks so one arm cannot dominate. A
document found by both arms accumulates two contributions — agreement is rewarded structurally
rather than by a hand-tuned weight.

### BM25 as a sparse vector

The BM25 score factorises into a sparse dot product:

```
score(q,d) = Σ_t  idf(t) · [ tf·(k₁+1) / (tf + k₁·(1 − b + b·|d|/avgdl)) ]
             └── query side ──┘  └──────────── document side ────────────┘
```

So the document vector stores TF weights and the query vector stores IDF, and Qdrant computes BM25
natively in its sparse index. No pulling candidates back into Node to rescore.

The cost is that BM25 needs corpus-global statistics (document frequency, average length) before
any vector can be encoded — which forces indexing to be two-pass: chunk everything and fit the
model, then embed and upsert. The fitted model is persisted alongside the index and rebuilt by
scrolling the store if the snapshot is ever missing, so the vector store stays authoritative.

Tokenisation is Unicode-aware throughout. Nothing assumes whitespace word boundaries or ASCII
punctuation, because the corpus spans Devanagari, Bengali, Tamil, Telugu and Arabic script.
Sentence splitting is hand-rolled rather than using `Intl.Segmenter`, which does not break on the
Devanagari danda (`।`) — the primary sentence terminator across most of this dataset.

### MMR before reranking, not after

Overlapping chunk strategies mean the top-10 can easily be five near-copies of one passage.
Applying MMR *before* the cross-encoder means its (expensive) budget is spent on distinct
passages rather than re-scoring duplicates.

---

## 5. Reranking

A bi-encoder embeds query and document independently, so it can only ever compare two summaries
produced in isolation. A cross-encoder reads both in one forward pass, letting attention align
specific query terms against specific document spans.

That is far more accurate and far too slow to run over a corpus — which is exactly the shape of
problem a two-stage pipeline solves: cheap recall-oriented retrieval to 10, expensive
precision-oriented rescoring to 5.

`bge-reranker-base` is XLM-RoBERTa based, so it scores Hindi queries against English passages
correctly rather than collapsing to lexical overlap.

The UI surfaces the rank delta per chunk. Watching a chunk move from rank 7 to rank 1 is the
clearest available evidence that the second stage earns its latency.

---

## 6. Guardrails

Eight checks across three stages. The staging matters as much as the checks.

### Input (pre-retrieval)

`prompt_injection`, `jailbreak`, `toxicity`. Pure regex and lexical scoring, run before anything
is embedded — so a hostile query costs no model time and no API spend.

Patterns are deliberately **high-precision rather than high-recall**. A false positive blocks a
legitimate question, which for a retrieval system is a worse failure than letting an odd phrasing
through into a pipeline that is already constrained to answer only from retrieved context. The
rules require verb+object structure rather than matching bare keywords, so a question that merely
contains the word "ignore" or "system" does not trip.

### Evidence (post-retrieval)

`similarity_threshold`, `off_topic`, `context_verification`.

The single most effective defence against hallucination is not asking the question when the
evidence is thin. If the best reranked chunk scores below `SIMILARITY_THRESHOLD`, the pipeline
refuses before spending a token on generation.

`off_topic` deliberately requires *two* signals to agree — an explicit out-of-scope phrasing hint
**and** low lexical coverage of the context. Either alone is unreliable: a genuine corpus question
can use generative phrasing, and a cross-lingual question legitimately shares few tokens with its
English source.

### Answer (post-generation)

`hallucination`, `confidence`.

Groundedness is computed per sentence: strip citation markers, take content words, and measure
what fraction appear in the retrieved context. A sentence clearing 60% coverage is very likely
drawn from the evidence rather than from model priors. The answer's score is the fraction of its
sentences that clear that bar.

This catches the characteristic RAG failure — a fluent, well-cited answer that quietly introduces
one fact nobody supplied — without a second model call.

### The confidence gate

```
overall = 0.35·topScore + 0.30·groundedness + 0.15·meanScore
        + 0.10·armAgreement + 0.10·contextCoverage
```

Weights favour the two most direct measures: whether the evidence answers the question
(`topScore`) and whether the answer used it (`groundedness`).

Below threshold, the answer is **replaced**, not hedged. A hedged answer still puts unverified
claims in front of the user in a form they will read and remember. `"I don't have enough
information."` does not.

---

## 7. Streaming and latency

### Why the wire is always streamed

Non-streaming requests to a 550B reasoning model routinely exceed a two-minute gateway timeout.
`generateCompletion` therefore streams on the wire regardless and accumulates deltas into a single
result — the non-streaming API is a convenience over a streaming transport, not a separate path.

Reasoning mode is **off by default**. It adds 30–120 seconds and, on grounded RAG where the answer
must come from five supplied passages, the quality difference is small. It remains available
per-request, and the reasoning stream is surfaced separately from the answer.

### What runs in parallel

The dense and sparse arms are genuinely concurrent — one is bounded by an embedding forward pass,
the other by an inverted-index walk, and neither consumes the other's output. Reported "retrieval"
latency is therefore `max(dense, sparse) + fusion`, not the sum.

Not everything parallelises. Indexing embeddings **serially** is measurably faster than
concurrently for the local provider: ONNX already saturates the available cores inside a single
forward pass, so issuing three batches at once only adds contention. Measured on this corpus,
`INDEX_CONCURRENCY=3` was more than 3× slower end-to-end than serial. Concurrency is now chosen by
provider — 1 for local, the configured value for hosted, where the bottleneck is network round
trips.

Percentiles use the **nearest-rank** method rather than linear interpolation, so every reported
value is a latency that actually occurred. An interpolated p99 corresponds to no real request,
which makes it useless for debugging a tail.

---

## 8. Evaluation

MSMARCO-XI's `is_selected` flag marks which passages genuinely answer each query. That turns
`/api/benchmark` into a measurement rather than a demo:

- **recall@5 / recall@10** — did we find the relevant passages at all?
- **precision@5** — how much of what we sent the model was actually relevant?
- **MRR** — how high did the first relevant passage rank?
- **nDCG@5** — full ranking quality, rewarding relevant passages placed *first* rather than
  merely present. This is the metric that matters when only five chunks reach the context window.

Documents are deduplicated before scoring, since several chunks can come from one passage and a
document should be credited once, at its best rank.

The sample is a **deterministic stride** through the evaluation set, not a random draw — a
benchmark you cannot reproduce is not a benchmark.

---

## 9. Security

- Secrets are read from the environment only, validated by Zod at boot, and redacted from every
  log line and from `redactedConfig()`.
- Helmet with a restrictive CSP; the API serves only JSON and SSE.
- CORS is an explicit allowlist. Requests with no `Origin` (curl, server-to-server) pass.
- Rate limiting is tiered: a global limit, plus tighter limits on the endpoints that cost real
  money (`/query`, `/transcribe`, `/voice-query`, `/benchmark`).
- Every request body and query string is Zod-validated at the edge; the parsed value replaces the
  raw input so handlers never see unvalidated data.
- Audio uploads are memory-only with a MIME allowlist and size cap enforced before buffering. They
  are never written to disk.
- In production, error responses carry a code and message but no stack traces or internal detail.
- `trust proxy` is set so per-IP rate limiting sees the real client address behind a platform
  proxy rather than counting every request against the proxy.
- Retrieved passages are fenced in clearly delimited blocks and the system prompt states that
  block content is data, never instruction — so a passage containing imperative text is not
  followed.

---

## 10. Known limitations

Stated plainly, because a system that hides these is harder to trust than one that does not.

1. **Groundedness is lexical.** Content-word overlap approximates entailment. A correctly grounded
   paraphrase using entirely different vocabulary scores lower than it deserves. A proper NLI
   model would be better and would cost another forward pass per sentence.

2. **Semantic chunking uses lexical cohesion, not embeddings.** Deliberate — see §3 — but a
   passage that shifts topic while reusing vocabulary will not be split.

3. **Guardrail patterns are English-centric.** The injection and jailbreak rules will not catch an
   attack written in Hindi. Multilingual coverage would need either translated patterns or a
   classifier.

4. **Analytics are in-process and per-instance.** Fine for a single service; a multi-replica
   deployment should export to Prometheus instead. The structured logs already carry the same
   fields.

5. **`GET /api/stats` walks the whole collection** to aggregate, because Qdrant has no aggregation
   API. Cached for 15s, which is adequate at this corpus size and would not be at 10M vectors.

6. **The embedded store loads all vectors into memory.** Correct for the dev corpus it targets,
   wrong as a production store — which is what Qdrant is for.

7. **One reranker forward pass per candidate.** Reranking 10 candidates on CPU is the second
   largest latency contributor after generation. A GPU or a hosted reranking NIM would remove it.
