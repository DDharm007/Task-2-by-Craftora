/**
 * Hybrid retriever.
 *
 * Pipeline:
 *
 *   query
 *     ├── embed (dense) ─┐
 *     └── BM25 (sparse) ─┤   both arms run concurrently
 *                        ▼
 *                  RRF fusion  → top-K (default 10)
 *                        ▼
 *                       MMR    → drop near-duplicate chunks
 *                        ▼
 *                  cross-encoder rerank → top-N (default 5)
 *                        ▼
 *                  parent expansion → wider context for the LLM
 *
 * The two retrieval arms are genuinely parallel: the dense arm is bounded by
 * the embedding forward pass, the sparse arm by an inverted-index walk, and
 * neither depends on the other's output.
 */
import type { RetrievedChunk } from '@goarag/shared';
import type { RetrievalOptions } from '@goarag/shared';
import { config } from '../../config/env.js';
import { logger } from '../../utils/logger.js';
import { StageTimer, timed } from '../../utils/async.js';
import { getEmbeddingProvider } from '../embeddings/index.js';
import { getBm25Model } from '../vector/bm25-service.js';
import { getVectorStore } from '../vector/index.js';
import type { SearchFilter, SearchHit } from '../vector/types.js';
import { rerankWithFallback, type RerankCandidate } from '../reranker/index.js';
import { maximalMarginalRelevance, reciprocalRankFusion, retrievalAgreement } from './fusion.js';
import { normalizeQuery, retrievalCache, type CacheHit, type CacheTier } from './cache.js';

export interface RetrievalRequest {
  query: string;
  options: RetrievalOptions;
  timer: StageTimer;
}

/**
 * Stages that a cache hit skips, zeroed so the breakdown reports what actually
 * ran rather than carrying stale values from a previous request.
 */
const CACHED_STAGES = [
  'embedding',
  'denseRetrieval',
  'sparseRetrieval',
  'fusion',
  'diversity',
  'reranking',
  'expansion',
] as const;

/** Rebuild a full outcome from a cache hit, attributing only the cost paid. */
function fromCache(hit: CacheHit, embeddingMs: number): RetrievalOutcome {
  return {
    chunks: hit.chunks,
    candidates: hit.candidates,
    agreement: hit.agreement,
    queryVector: hit.queryVector,
    timings: {
      embedding: embeddingMs,
      dense: 0,
      sparse: 0,
      fusion: 0,
      mmr: 0,
      rerank: 0,
      expansion: 0,
    },
    rerankerProvider: hit.rerankerProvider,
    degraded: hit.degraded,
    cache: { tier: hit.tier, similarity: hit.similarity },
  };
}

/**
 * Trim a query to the configured bound, cutting on a word boundary where one
 * is available so the tail term isn't left as a fragment that matches nothing.
 */
function clampQuery(query: string): string {
  const trimmed = query.trim();
  const limit = config.retrieval.maxQueryChars;
  if (trimmed.length <= limit) return trimmed;

  const head = trimmed.slice(0, limit);
  const lastSpace = head.lastIndexOf(' ');
  const clamped = lastSpace > limit * 0.6 ? head.slice(0, lastSpace) : head;
  logger.debug(
    { from: trimmed.length, to: clamped.length },
    'Query exceeded the retrieval bound and was truncated',
  );
  return clamped;
}

export interface RetrievalOutcome {
  chunks: RetrievedChunk[];
  /** Every candidate considered before reranking, for the inspector. */
  candidates: RetrievedChunk[];
  agreement: number;
  queryVector: Float32Array;
  timings: {
    embedding: number;
    dense: number;
    sparse: number;
    fusion: number;
    /** MMR diversity selection. */
    mmr: number;
    rerank: number;
    /** Fetching parent chunks for the surviving children. */
    expansion: number;
  };
  rerankerProvider: string;
  degraded: boolean;
  /** Present only when the result came from the cache. */
  cache?: { tier: CacheTier; similarity: number };
}

export async function retrieve(request: RetrievalRequest): Promise<RetrievalOutcome> {
  const { options, timer } = request;

  // Bound the query before anything touches it. Both arms scale with its
  // length — the embedder tokenizes the whole string, and BM25 walks one
  // posting list per distinct term — so an unbounded query is an unbounded
  // latency budget. Applied here rather than only in the HTTP schema so the
  // bound holds for every caller.
  const query = clampQuery(request.query);
  const normalized = normalizeQuery(query);

  // ── L1: exact match ───────────────────────────────────────────────────────
  // The only tier that can skip the encoder, because it never needs a vector
  // to establish identity. Every stage stays at zero in the breakdown, which
  // is honest: none of them ran.
  const exact = retrievalCache.lookupExact(normalized, options);
  if (exact) {
    for (const stage of CACHED_STAGES) timer.set(stage, 0);
    logger.debug({ query: query.slice(0, 80) }, 'Retrieval served from L1 cache');
    return fromCache(exact, 0);
  }

  const topK = options.topK ?? config.retrieval.topK;
  const rerankTopN = options.rerankTopN ?? config.retrieval.rerankTopN;
  const enableRerank = options.enableRerank ?? true;
  const enableMmr = options.enableMmr ?? config.retrieval.enableMmr;
  const enableParents = options.enableParentExpansion ?? config.retrieval.enableParentExpansion;

  const filter: SearchFilter = {
    includeParents: false,
    ...(options.languages?.length ? { languages: options.languages } : {}),
  };

  const store = await getVectorStore();
  const bm25 = await getBm25Model();

  // Over-fetch each arm: fusion needs deeper lists than the final K so that a
  // document ranked 15th by one arm and 2nd by the other can still surface.
  const armLimit = Math.max(topK * 2, 20);

  let embeddingMs = 0;
  let denseMs = 0;
  let sparseMs = 0;

  // ── the two retrieval arms, concurrently ─────────────────────────────────
  // The sparse arm starts first and speculatively: it depends only on the
  // query text, not on the embedding, so it overlaps the encoder either way.
  // On an L2 cache hit below its result is simply dropped — but because it was
  // running in parallel the whole time, that waste costs no wall-clock on the
  // miss path, which is the one the latency budget is judged on.
  const sparsePromise = (async () => {
    const searched = await timed(async () => {
      const queryVector = bm25.queryVector(query);
      if (queryVector.indices.length === 0) return [] as SearchHit[];
      return store.searchSparse(queryVector, armLimit, filter);
    });
    sparseMs = searched.durationMs;
    return searched.value;
  })();
  // An abandoned arm must not surface as an unhandled rejection.
  sparsePromise.catch(() => undefined);

  const embedded = await timed(() => getEmbeddingProvider().embed([query], 'query'));
  embeddingMs = embedded.durationMs;
  const vector = embedded.value[0];
  if (!vector) throw new Error('Query embedding failed');

  // ── L2: semantic match ────────────────────────────────────────────────────
  // Only reachable once the vector exists, which is the whole reason this is a
  // separate tier from L1 rather than one lookup: similarity is a property of
  // the embedding, so this can never avoid paying for it.
  const semantic = retrievalCache.lookupSemantic(vector, options);
  if (semantic) {
    timer.set('embedding', embeddingMs);
    for (const stage of CACHED_STAGES) {
      if (stage !== 'embedding') timer.set(stage, 0);
    }
    logger.debug(
      { query: query.slice(0, 80), similarity: Number(semantic.similarity.toFixed(4)) },
      'Retrieval served from L2 cache',
    );
    return fromCache(semantic, embeddingMs);
  }

  const searchedDense = await timed(() => store.searchDense(vector, armLimit, filter));
  denseMs = searchedDense.durationMs;
  const dense = { vector, hits: searchedDense.value };
  const sparseHits = await sparsePromise;

  retrievalCache.recordMiss();

  timer.set('embedding', embeddingMs);
  timer.set('denseRetrieval', denseMs);
  timer.set('sparseRetrieval', sparseMs);

  // ── fusion ────────────────────────────────────────────────────────────────
  const fusion = await timed(() => {
    const fused = reciprocalRankFusion(dense.hits, sparseHits, config.retrieval.rrfK);
    const agreement = retrievalAgreement(dense.hits, sparseHits);
    return { fused, agreement };
  });
  timer.set('fusion', fusion.durationMs);

  let candidates = fusion.value.fused;

  // ── diversity ─────────────────────────────────────────────────────────────
  // Applied before reranking so the cross-encoder spends its budget on
  // distinct passages rather than re-scoring near-duplicates.
  //
  // Timed separately: MMR sits inside the latency budget but used to be
  // invisible in the breakdown, which let ~70ms/query hide between the stages
  // that *were* measured. Anything on the critical path gets a number.
  const diversity = await timed(() => {
    if (enableMmr && candidates.length > topK) {
      return maximalMarginalRelevance(
        candidates,
        topK,
        config.retrieval.mmrLambda,
        (candidate) => candidate.fusedScore,
        () => null, // candidate vectors are not fetched back; trigram similarity is used
        (candidate) => candidate.hit.text,
      );
    }
    return candidates.slice(0, topK);
  });
  candidates = diversity.value;
  const mmrMs = diversity.durationMs;
  timer.set('diversity', mmrMs);

  const beforeRerank: RetrievedChunk[] = candidates.map((candidate, index) => ({
    id: candidate.hit.id,
    text: candidate.hit.text,
    metadata: candidate.hit.metadata,
    denseScore: candidate.denseScore,
    sparseScore: candidate.sparseScore,
    fusedScore: candidate.fusedScore,
    rerankScore: null,
    score: candidate.fusedScore,
    rankBeforeRerank: index + 1,
    rankAfterRerank: null,
    parentText: null,
    matchedBy: candidate.matchedBy,
  }));

  // ── reranking ─────────────────────────────────────────────────────────────
  let finalChunks = beforeRerank;
  let rerankMs = 0;
  let rerankerProvider = 'disabled';
  let degraded = false;

  if (enableRerank && beforeRerank.length > 0) {
    const rerankCandidates: RerankCandidate[] = beforeRerank.map((chunk) => ({
      id: chunk.id,
      text: chunk.text,
      priorScore: chunk.fusedScore,
    }));

    const reranked = await timed(() => rerankWithFallback(query, rerankCandidates));
    rerankMs = reranked.durationMs;
    rerankerProvider = reranked.value.provider;
    degraded = reranked.value.degraded;

    const scoreById = new Map(reranked.value.results.map((result) => [result.id, result.score]));
    finalChunks = [...beforeRerank]
      .map((chunk) => ({ ...chunk, rerankScore: scoreById.get(chunk.id) ?? 0 }))
      .sort((a, b) => (b.rerankScore ?? 0) - (a.rerankScore ?? 0))
      .slice(0, rerankTopN)
      .map((chunk, index) => ({
        ...chunk,
        score: chunk.rerankScore ?? chunk.fusedScore,
        rankAfterRerank: index + 1,
      }));
  } else {
    finalChunks = beforeRerank.slice(0, rerankTopN).map((chunk, index) => ({
      ...chunk,
      rankAfterRerank: index + 1,
    }));
  }
  timer.set('reranking', rerankMs);

  // ── parent expansion ──────────────────────────────────────────────────────
  // The matched child is precise but often too narrow to answer from; its
  // parent carries the surrounding sentences the model needs.
  const expansion = await timed(async () => {
    if (!enableParents || finalChunks.length === 0) return;
    const parentIds = [
      ...new Set(
        finalChunks
          .map((chunk) => chunk.metadata.parentChunk)
          .filter((id): id is string => typeof id === 'string' && id.length > 0),
      ),
    ];
    if (parentIds.length === 0) return;
    try {
      const parents = await store.fetchByIds(parentIds);
      const parentById = new Map(parents.map((parent) => [parent.id, parent.text]));
      finalChunks = finalChunks.map((chunk) => ({
        ...chunk,
        parentText: chunk.metadata.parentChunk
          ? (parentById.get(chunk.metadata.parentChunk) ?? null)
          : null,
      }));
    } catch (error) {
      // Expansion is an enhancement; losing it must not fail the query.
      logger.warn({ error: (error as Error).message }, 'Parent chunk expansion failed');
    }
  });
  const expansionMs = expansion.durationMs;
  timer.set('expansion', expansionMs);

  logger.debug(
    {
      query: query.slice(0, 80),
      dense: dense.hits.length,
      sparse: sparseHits.length,
      fused: fusion.value.fused.length,
      final: finalChunks.length,
      agreement: Number(fusion.value.agreement.toFixed(3)),
    },
    'Retrieval complete',
  );

  // Cached after the fact, so a request that threw partway never leaves a
  // half-built result behind for the next caller to be served.
  retrievalCache.store(normalized, options, {
    chunks: finalChunks,
    candidates: beforeRerank,
    agreement: fusion.value.agreement,
    queryVector: dense.vector,
    rerankerProvider,
    degraded,
  });

  return {
    chunks: finalChunks,
    candidates: beforeRerank,
    agreement: fusion.value.agreement,
    queryVector: dense.vector,
    timings: {
      embedding: embeddingMs,
      dense: denseMs,
      sparse: sparseMs,
      fusion: fusion.durationMs,
      mmr: mmrMs,
      rerank: rerankMs,
      expansion: expansionMs,
    },
    rerankerProvider,
    degraded,
  };
}
