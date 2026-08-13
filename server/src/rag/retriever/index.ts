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
import type { RetrievedChunk } from '@voxrag/shared';
import type { RetrievalOptions } from '@voxrag/shared';
import { config } from '../../config/env.js';
import { logger } from '../../utils/logger.js';
import { StageTimer, timed } from '../../utils/async.js';
import { getEmbeddingProvider } from '../embeddings/index.js';
import { getBm25Model } from '../vector/bm25-service.js';
import { getVectorStore } from '../vector/index.js';
import type { SearchFilter, SearchHit } from '../vector/types.js';
import { rerankWithFallback, type RerankCandidate } from '../reranker/index.js';
import { maximalMarginalRelevance, reciprocalRankFusion, retrievalAgreement } from './fusion.js';

export interface RetrievalRequest {
  query: string;
  options: RetrievalOptions;
  timer: StageTimer;
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
    rerank: number;
  };
  rerankerProvider: string;
  degraded: boolean;
}

export async function retrieve(request: RetrievalRequest): Promise<RetrievalOutcome> {
  const { query, options, timer } = request;

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
  const densePromise = (async () => {
    const embedded = await timed(() => getEmbeddingProvider().embed([query], 'query'));
    embeddingMs = embedded.durationMs;
    const vector = embedded.value[0];
    if (!vector) throw new Error('Query embedding failed');

    const searched = await timed(() => store.searchDense(vector, armLimit, filter));
    denseMs = searched.durationMs;
    return { vector, hits: searched.value };
  })();

  const sparsePromise = (async () => {
    const searched = await timed(async () => {
      const queryVector = bm25.queryVector(query);
      if (queryVector.indices.length === 0) return [] as SearchHit[];
      return store.searchSparse(queryVector, armLimit, filter);
    });
    sparseMs = searched.durationMs;
    return searched.value;
  })();

  const [dense, sparseHits] = await Promise.all([densePromise, sparsePromise]);

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
  if (enableMmr && candidates.length > topK) {
    candidates = maximalMarginalRelevance(
      candidates,
      topK,
      config.retrieval.mmrLambda,
      (candidate) => candidate.fusedScore,
      () => null, // candidate vectors are not fetched back; trigram similarity is used
      (candidate) => candidate.hit.text,
    );
  } else {
    candidates = candidates.slice(0, topK);
  }

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
  if (enableParents && finalChunks.length > 0) {
    const parentIds = [
      ...new Set(
        finalChunks
          .map((chunk) => chunk.metadata.parentChunk)
          .filter((id): id is string => typeof id === 'string' && id.length > 0),
      ),
    ];
    if (parentIds.length > 0) {
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
    }
  }

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
      rerank: rerankMs,
    },
    rerankerProvider,
    degraded,
  };
}
