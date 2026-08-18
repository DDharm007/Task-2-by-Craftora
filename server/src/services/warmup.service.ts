/**
 * Pipeline warm-up.
 *
 * Every expensive component here loads lazily on first use: ONNX weights are
 * read and compiled, the embedded store pulls 7k vectors off disk into one
 * flat Float32Array, and the BM25 model rebuilds a ~28k-term inverted index.
 * Together that is seconds of work, and whichever query arrives first pays all
 * of it.
 *
 * That is worth eliminating twice over:
 *
 *   · **In production**, it is the difference between a first user waiting
 *     several seconds and waiting 20ms. The server calls this at boot, off the
 *     critical path, so the cost is paid before anyone asks for anything.
 *   · **In the benchmark**, a cold load inside the timed window is an artefact
 *     of the harness rather than latency any user would see — it lands
 *     entirely on whichever query happens to run first and drags p100 with it.
 *     Measured here: p100 887ms warm-store-less against 30ms warm.
 *
 * The two callers share this function deliberately. If the benchmark warmed
 * *more* than the server does, its numbers would flatter a pipeline that is
 * genuinely slow on its first real request — so the honest arrangement is for
 * both to warm exactly the same set.
 */
import { logger } from '../utils/logger.js';
import { StageTimer, now } from '../utils/async.js';
import { getEmbeddingProvider } from '../rag/embeddings/index.js';
import { getReranker } from '../rag/reranker/index.js';
import { getBm25Model } from '../rag/vector/bm25-service.js';
import { getVectorStore } from '../rag/vector/index.js';
import { retrieve } from '../rag/retriever/index.js';

/**
 * A throwaway query, run once to force every lazy path to resolve.
 *
 * Deliberately generic: it must not be one of the benchmark's evaluation
 * queries, or its result would sit in the embedding cache and that query would
 * then be measured as a cache hit while every other query pays full cost.
 */
const PROBE_QUERY = 'warm up the retrieval pipeline';

export interface WarmupResult {
  ms: number;
  /** False when the index is empty, so the probe retrieval was skipped. */
  probed: boolean;
}

export async function warmPipeline(): Promise<WarmupResult> {
  const started = now();

  const store = await getVectorStore();
  const stats = await store.stats();

  // Both models compile ONNX graphs on first call; the BM25 model rebuilds its
  // inverted index. None of these depend on each other.
  await Promise.all([
    getEmbeddingProvider().warmup(),
    getReranker().warmup(),
    getBm25Model(),
  ]);

  // One full pass through the real code path. Warming the components
  // individually still leaves per-call allocations — ONNX arenas sized for a
  // real sequence length, the store's scratch buffers — to the first query.
  let probed = false;
  if (stats.vectors > 0) {
    try {
      await retrieve({ query: PROBE_QUERY, options: {}, timer: new StageTimer() });
      probed = true;
    } catch (error) {
      // A failed probe is not fatal: the pipeline is merely less warm.
      logger.warn({ error: (error as Error).message }, 'Warm-up probe query failed');
    }
  }

  const ms = Math.round(now() - started);
  logger.info({ ms, vectors: stats.vectors, probed }, 'Pipeline warm');
  return { ms, probed };
}
