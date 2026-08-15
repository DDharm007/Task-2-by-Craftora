/**
 * Benchmarking.
 *
 * Measures two things that are usually conflated:
 *
 *   1. Latency — per-stage percentiles (p50/p70/p95/p99/p100).
 *   2. Retrieval quality — recall@k, precision@k, MRR and nDCG against the
 *      dataset's own `is_selected` labels.
 *
 * The second is the one that actually says whether the RAG pipeline works.
 * Because MSMARCO-XI marks which passages answer each query, we can score
 * retrieval objectively instead of eyeballing outputs.
 *
 * Generation is off by default: a 550B model dominates the wall clock and
 * tells you nothing about retrieval. Turn it on to measure the full pipeline.
 */
import type {
  BenchmarkCase,
  BenchmarkResult,
  RetrievalQualityMetrics,
  StageLatencyStats,
  TokenUsage,
} from '@goarag/shared';
import { config } from '../config/env.js';
import { logger } from '../utils/logger.js';
import { StageTimer, mapWithConcurrency, now } from '../utils/async.js';
import { retrieve } from '../rag/retriever/index.js';
import { percentiles } from './analytics.service.js';
import { loadDataset, toEvaluationCases, type EvaluationCase } from './dataset.service.js';
import { runQuery } from './rag.service.js';
import { getIndexStats } from './indexing.service.js';
import { getEmbeddingProvider } from '../rag/embeddings/index.js';
import { getReranker } from '../rag/reranker/index.js';
import { getVectorStore } from '../rag/vector/index.js';
import { errors } from '../utils/errors.js';

export interface BenchmarkOptions {
  sampleSize: number;
  generation: boolean;
  language?: string;
  concurrency: number;
}

interface CaseOutcome {
  benchmarkCase: BenchmarkCase;
  latencies: {
    embedding: number;
    retrieval: number;
    reranking: number;
    generation: number;
    total: number;
  };
  /** Final reranked document ids, best first — used for the @5 metrics. */
  ranked: string[];
  /**
   * Fused candidate document ids before reranking truncates to top-N.
   * Recall@10 must be measured here: the final list holds only `rerankTopN`
   * items, so scoring @10 against it would just restate recall@5.
   */
  candidates: string[];
  relevant: Set<string>;
  usage: TokenUsage;
  confidence: number;
}

/**
 * Deterministic sample.
 *
 * A fixed stride rather than random selection so repeated runs are
 * comparable — a benchmark you cannot reproduce is not a benchmark.
 */
function sample(cases: EvaluationCase[], size: number): EvaluationCase[] {
  if (cases.length <= size) return cases;
  const stride = cases.length / size;
  return Array.from({ length: size }, (_, i) => cases[Math.floor(i * stride)] as EvaluationCase);
}

export async function runBenchmark(options: BenchmarkOptions): Promise<BenchmarkResult> {
  const startedAt = new Date().toISOString();
  const started = now();

  const stats = await getIndexStats();
  if (!stats.indexed) throw errors.indexEmpty();

  const bundle = await loadDataset().catch(() => {
    throw errors.validation(
      'No dataset found. Run `npm run dataset:download` before benchmarking.',
    );
  });

  let cases = toEvaluationCases(bundle);
  if (options.language) {
    cases = cases.filter((item) => item.language === options.language);
  }
  if (cases.length === 0) {
    throw errors.validation(
      options.language
        ? `No labelled evaluation cases for language "${options.language}".`
        : 'The dataset contains no labelled evaluation cases (is_selected is all zero).',
    );
  }

  const selected = sample(cases, options.sampleSize);
  logger.info(
    { cases: selected.length, generation: options.generation, concurrency: options.concurrency },
    'Running benchmark',
  );

  // Warm the models before timing anything. The server does this at boot
  // (see index.ts), so a lazy load inside the measured window is an artefact of
  // the harness, not latency a user would ever see: it put a ~2.2s model load
  // on whichever queries happened to run first and dragged every percentile
  // above p50 up with it.
  const warmStarted = now();
  await Promise.all([getEmbeddingProvider().warmup(), getReranker().warmup()]);
  logger.info({ ms: Math.round(now() - warmStarted) }, 'Models warm — starting timed run');

  const outcomes = await mapWithConcurrency(selected, options.concurrency, async (item) =>
    runCase(item, options.generation),
  );

  const finished = now();

  return {
    startedAt,
    finishedAt: new Date().toISOString(),
    durationMs: Math.round(finished - started),
    sampleSize: outcomes.length,
    generationEnabled: options.generation,
    latency: aggregateLatency(outcomes),
    quality: computeQuality(outcomes),
    averageConfidence:
      outcomes.length > 0
        ? Number(
            (outcomes.reduce((sum, o) => sum + o.confidence, 0) / outcomes.length).toFixed(4),
          )
        : 0,
    tokensUsed: outcomes.reduce<TokenUsage>(
      (sum, o) => ({
        promptTokens: sum.promptTokens + o.usage.promptTokens,
        completionTokens: sum.completionTokens + o.usage.completionTokens,
        reasoningTokens: sum.reasoningTokens + o.usage.reasoningTokens,
        totalTokens: sum.totalTokens + o.usage.totalTokens,
      }),
      { promptTokens: 0, completionTokens: 0, reasoningTokens: 0, totalTokens: 0 },
    ),
    cases: outcomes.map((o) => o.benchmarkCase),
    config: {
      embeddingProvider: getEmbeddingProvider().name,
      embeddingModel: getEmbeddingProvider().model,
      vectorStore: (await getVectorStore()).name,
      rerankerProvider: getReranker().name,
      llmModel: config.llm.model,
      topK: config.retrieval.topK,
      rerankTopN: config.retrieval.rerankTopN,
    },
  };
}

async function runCase(item: EvaluationCase, withGeneration: boolean): Promise<CaseOutcome> {
  const relevant = new Set(item.expectedDocumentIds);
  const caseStarted = now();

  if (withGeneration) {
    const result = await runQuery({ query: item.query, options: {} });
    // Deduplicate: several chunks can come from one passage, and a document
    // should be credited once, at its best rank.
    const ranked = dedupeDocuments(result.chunks.map((chunk) => chunk.metadata.documentId));
    const hit = ranked.some((id) => relevant.has(id));
    // The full-pipeline path does not expose the pre-rerank candidates, so the
    // @10 window is the final list. Retrieval-only mode measures it properly.
    const candidates = ranked;

    return {
      benchmarkCase: {
        query: item.query,
        language: item.language,
        expectedDocumentIds: item.expectedDocumentIds,
        retrievedDocumentIds: ranked,
        hit,
        reciprocalRank: reciprocalRank(ranked, relevant),
        latencyMs: Math.round(result.latency.total),
        confidence: result.confidence.overall,
        status: result.status,
      },
      latencies: {
        embedding: result.latency.embedding,
        retrieval:
          Math.max(result.latency.denseRetrieval, result.latency.sparseRetrieval) + result.latency.fusion,
        reranking: result.latency.reranking,
        generation: result.latency.generation,
        total: result.latency.total,
      },
      ranked,
      candidates,
      relevant,
      usage: result.usage,
      confidence: result.confidence.overall,
    };
  }

  // Retrieval-only: skip guardrails and the LLM entirely.
  const timer = new StageTimer();
  const retrieval = await retrieve({ query: item.query, options: {}, timer });
  const totalMs = now() - caseStarted;
  const ranked = dedupeDocuments(retrieval.chunks.map((chunk) => chunk.metadata.documentId));
  // `candidates` is the fused top-K before reranking cut it down to top-N.
  const candidates = dedupeDocuments(
    retrieval.candidates.map((chunk) => chunk.metadata.documentId),
  );
  const hit = ranked.some((id) => relevant.has(id));
  const topScore = retrieval.chunks[0]?.score ?? 0;

  return {
    benchmarkCase: {
      query: item.query,
      language: item.language,
      expectedDocumentIds: item.expectedDocumentIds,
      retrievedDocumentIds: ranked,
      hit,
      reciprocalRank: reciprocalRank(ranked, relevant),
      latencyMs: Math.round(totalMs),
      confidence: topScore,
      status: hit ? 'answered' : 'insufficient_context',
    },
    latencies: {
      embedding: retrieval.timings.embedding,
      retrieval: Math.max(retrieval.timings.dense, retrieval.timings.sparse) + retrieval.timings.fusion,
      reranking: retrieval.timings.rerank,
      generation: 0,
      total: totalMs,
    },
    ranked,
    candidates,
    relevant,
    usage: { promptTokens: 0, completionTokens: 0, reasoningTokens: 0, totalTokens: 0 },
    confidence: topScore,
  };
}

/** Keep the first (best-ranked) occurrence of each document id. */
function dedupeDocuments(ids: readonly string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const id of ids) {
    if (seen.has(id)) continue;
    seen.add(id);
    out.push(id);
  }
  return out;
}

/** 1/rank of the first relevant document, 0 if none was retrieved. */
function reciprocalRank(ranked: readonly string[], relevant: ReadonlySet<string>): number {
  for (let i = 0; i < ranked.length; i += 1) {
    if (relevant.has(ranked[i] as string)) return 1 / (i + 1);
  }
  return 0;
}

/**
 * Normalised discounted cumulative gain at k.
 *
 * Unlike recall, nDCG rewards ranking the relevant document *first* rather
 * than merely somewhere in the list — which is what matters when only the top
 * few chunks reach the model's context window.
 */
function ndcgAt(ranked: readonly string[], relevant: ReadonlySet<string>, k: number): number {
  let dcg = 0;
  for (let i = 0; i < Math.min(k, ranked.length); i += 1) {
    if (relevant.has(ranked[i] as string)) dcg += 1 / Math.log2(i + 2);
  }
  // Ideal ranking puts every relevant document at the top.
  let idcg = 0;
  const ideal = Math.min(k, relevant.size);
  for (let i = 0; i < ideal; i += 1) idcg += 1 / Math.log2(i + 2);
  return idcg > 0 ? dcg / idcg : 0;
}

function recallAt(ranked: readonly string[], relevant: ReadonlySet<string>, k: number): number {
  if (relevant.size === 0) return 0;
  let found = 0;
  for (let i = 0; i < Math.min(k, ranked.length); i += 1) {
    if (relevant.has(ranked[i] as string)) found += 1;
  }
  return found / relevant.size;
}

function precisionAt(ranked: readonly string[], relevant: ReadonlySet<string>, k: number): number {
  const window = Math.min(k, ranked.length);
  if (window === 0) return 0;
  let found = 0;
  for (let i = 0; i < window; i += 1) {
    if (relevant.has(ranked[i] as string)) found += 1;
  }
  return found / window;
}

function computeQuality(outcomes: readonly CaseOutcome[]): RetrievalQualityMetrics {
  if (outcomes.length === 0) {
    return {
      recallAt5: 0,
      recallAt10: 0,
      precisionAt5: 0,
      mrr: 0,
      ndcgAt5: 0,
      hitRate: 0,
      evaluatedQueries: 0,
    };
  }

  const mean = (fn: (outcome: CaseOutcome) => number): number =>
    Number((outcomes.reduce((sum, outcome) => sum + fn(outcome), 0) / outcomes.length).toFixed(4));

  return {
    recallAt5: mean((o) => recallAt(o.ranked, o.relevant, 5)),
    // Measured over the pre-rerank candidates — see `CaseOutcome.candidates`.
    recallAt10: mean((o) => recallAt(o.candidates, o.relevant, 10)),
    precisionAt5: mean((o) => precisionAt(o.ranked, o.relevant, 5)),
    mrr: mean((o) => o.benchmarkCase.reciprocalRank),
    ndcgAt5: mean((o) => ndcgAt(o.ranked, o.relevant, 5)),
    hitRate: mean((o) => (o.benchmarkCase.hit ? 1 : 0)),
    evaluatedQueries: outcomes.length,
  };
}

function aggregateLatency(outcomes: readonly CaseOutcome[]): StageLatencyStats {
  return {
    embedding: percentiles(outcomes.map((o) => o.latencies.embedding)),
    retrieval: percentiles(outcomes.map((o) => o.latencies.retrieval)),
    reranking: percentiles(outcomes.map((o) => o.latencies.reranking)),
    generation: percentiles(outcomes.map((o) => o.latencies.generation).filter((v) => v > 0)),
    transcription: percentiles([]),
    total: percentiles(outcomes.map((o) => o.latencies.total)),
  };
}
