/**
 * Indexing pipeline: dataset → chunks → embeddings + BM25 → vector store.
 *
 * BM25 forces a two-pass structure. Document frequency and average document
 * length are corpus-global, so every chunk must exist before any sparse vector
 * can be computed. Pass one chunks everything and fits the model; pass two
 * embeds and upserts.
 *
 * Embedding is the bottleneck (a CPU forward pass per batch), so batches are
 * pipelined with bounded concurrency: while one batch is in the model, the
 * previous batch is being written to the store.
 */
import type { ChunkStrategy, IndexStats } from '@goarag/shared';
import { config } from '../config/env.js';
import { logger } from '../utils/logger.js';
import { chunkArray, mapWithConcurrency, now } from '../utils/async.js';
import { chunkDocuments, type PreparedChunk } from '../rag/chunking/index.js';
import { getEmbeddingProvider } from '../rag/embeddings/index.js';
import { Bm25Model } from '../rag/vector/bm25.js';
import { setBm25Model } from '../rag/vector/bm25-service.js';
import { bm25ModelPath, getVectorStore } from '../rag/vector/index.js';
import type { VectorRecord } from '../rag/vector/types.js';
import { EmbeddedVectorStore } from '../rag/vector/embedded-store.js';
import { downloadDataset, loadDataset, toSourceDocuments } from './dataset.service.js';

export interface IndexProgress {
  phase: 'downloading' | 'chunking' | 'fitting' | 'embedding' | 'finalising' | 'done';
  processed: number;
  total: number;
  message: string;
}

export interface IndexResult {
  documents: number;
  chunks: number;
  parents: number;
  vectors: number;
  durationMs: number;
  embeddingMs: number;
  averageChunkTokens: number;
  strategies: Array<{ strategy: ChunkStrategy; count: number }>;
  languages: Array<{ language: string; count: number }>;
}

export interface IndexOptions {
  reset?: boolean;
  forceDownload?: boolean;
  onProgress?: (progress: IndexProgress) => void;
}

export async function runIndexing(options: IndexOptions = {}): Promise<IndexResult> {
  const started = now();
  const report = (progress: IndexProgress) => options.onProgress?.(progress);

  // ── 1. dataset ────────────────────────────────────────────────────────────
  report({ phase: 'downloading', processed: 0, total: 0, message: 'Loading dataset' });
  const bundle = options.forceDownload
    ? await downloadDataset({ force: true })
    : await loadDataset().catch(() => downloadDataset());

  const documents = toSourceDocuments(bundle);
  logger.info({ records: bundle.records.length, documents: documents.length }, 'Dataset ready');

  // ── 2. chunking ───────────────────────────────────────────────────────────
  report({
    phase: 'chunking',
    processed: 0,
    total: documents.length,
    message: `Chunking ${documents.length} passages`,
  });

  const chunks = chunkDocuments(documents, {
    targetTokens: config.chunking.targetTokens,
    minTokens: config.chunking.minTokens,
    overlapTokens: config.chunking.overlapTokens,
    parentTokens: config.chunking.parentTokens,
    semanticPercentile: config.chunking.semanticPercentile,
    windowTokens: config.chunking.windowTokens,
    windowStride: config.chunking.windowStride,
  });

  if (chunks.length === 0) {
    throw new Error('Chunking produced no chunks — is the dataset empty?');
  }
  logger.info({ chunks: chunks.length }, 'Chunking complete');

  // ── 3. fit BM25 ───────────────────────────────────────────────────────────
  // Must see every chunk before any sparse vector can be encoded.
  report({
    phase: 'fitting',
    processed: 0,
    total: chunks.length,
    message: 'Fitting BM25 over the corpus',
  });

  const bm25 = new Bm25Model();
  bm25.fit(chunks.map((chunk) => chunk.text));
  logger.info({ terms: bm25.size, documents: bm25.docCount }, 'BM25 model fitted');

  // ── 4. store setup ────────────────────────────────────────────────────────
  const store = await getVectorStore();
  const embedder = getEmbeddingProvider();

  if (options.reset) {
    logger.warn({ collection: store.collection }, 'Resetting collection');
    await store.reset();
  }
  await store.init(embedder.dimensions);

  // ── 5. embed + upsert ─────────────────────────────────────────────────────
  const batches = chunkArray(chunks, config.dataset.batchSize);
  let processed = 0;
  let embeddingMs = 0;

  // Concurrency depends on where the bottleneck is. The local ONNX provider is
  // CPU-bound and already saturates the cores inside a single forward pass, so
  // issuing parallel batches only adds contention — measured on this corpus,
  // concurrency 3 was >3x slower end-to-end than serial. A hosted provider is
  // network-bound, where parallelism is exactly what hides the round trip.
  const embedConcurrency =
    config.embedding.provider === 'local' ? 1 : config.dataset.concurrency;

  report({
    phase: 'embedding',
    processed: 0,
    total: chunks.length,
    message: `Embedding ${chunks.length} chunks with ${embedder.model}`,
  });

  await mapWithConcurrency(batches, embedConcurrency, async (batch) => {
    const embedStarted = now();
    // `passage` input type — the asymmetric counterpart to the query-side
    // embedding used at search time.
    const vectors = await embedder.embed(
      batch.map((chunk) => chunk.embedText),
      'passage',
    );
    embeddingMs += now() - embedStarted;

    const records: VectorRecord[] = batch.map((chunk, i) => ({
      id: chunk.id,
      chunkKey: chunk.chunkKey,
      text: chunk.text,
      embedText: chunk.embedText,
      vector: vectors[i] as Float32Array,
      // BM25 indexes the display text, not the metadata-prefixed embed text:
      // the header's tokens would otherwise pollute term statistics.
      sparse: bm25.documentVector(chunk.text),
      metadata: chunk.metadata,
      isParent: chunk.isParent,
    }));

    await store.upsert(records);

    processed += batch.length;
    report({
      phase: 'embedding',
      processed,
      total: chunks.length,
      message: `Embedded ${processed}/${chunks.length} chunks`,
    });
    if (processed % (config.dataset.batchSize * 5) < config.dataset.batchSize) {
      logger.info({ processed, total: chunks.length }, 'Indexing progress');
    }
  });

  // ── 6. persist ────────────────────────────────────────────────────────────
  report({ phase: 'finalising', processed: chunks.length, total: chunks.length, message: 'Persisting index' });

  await bm25.save(bm25ModelPath());
  setBm25Model(bm25);

  // The embedded driver buffers writes in memory; force them to disk.
  if (store instanceof EmbeddedVectorStore) await store.flush();

  const summary = summarize(chunks);
  const durationMs = now() - started;

  report({ phase: 'done', processed: chunks.length, total: chunks.length, message: 'Indexing complete' });

  logger.info(
    {
      documents: documents.length,
      chunks: summary.children,
      parents: summary.parents,
      durationMs: Math.round(durationMs),
      embeddingMs: Math.round(embeddingMs),
    },
    'Indexing complete',
  );

  return {
    documents: documents.length,
    chunks: summary.children,
    parents: summary.parents,
    vectors: chunks.length,
    durationMs,
    embeddingMs,
    averageChunkTokens: summary.averageTokens,
    strategies: summary.strategies,
    languages: summary.languages,
  };
}

function summarize(chunks: readonly PreparedChunk[]) {
  const strategies = new Map<ChunkStrategy, number>();
  const languages = new Map<string, number>();
  let parents = 0;
  let children = 0;
  let totalTokens = 0;

  for (const chunk of chunks) {
    strategies.set(chunk.metadata.strategy, (strategies.get(chunk.metadata.strategy) ?? 0) + 1);
    languages.set(chunk.metadata.language, (languages.get(chunk.metadata.language) ?? 0) + 1);
    if (chunk.isParent) {
      parents += 1;
    } else {
      children += 1;
      totalTokens += chunk.metadata.tokenCount;
    }
  }

  return {
    parents,
    children,
    averageTokens: children ? Math.round(totalTokens / children) : 0,
    strategies: [...strategies.entries()].map(([strategy, count]) => ({ strategy, count })),
    languages: [...languages.entries()]
      .map(([language, count]) => ({ language, count }))
      .sort((a, b) => b.count - a.count),
  };
}

/** Index statistics for /api/stats, cached briefly since it walks the store. */
let statsCache: { value: IndexStats; expiresAt: number } | null = null;
const STATS_TTL_MS = 15_000;

export async function getIndexStats(force = false): Promise<IndexStats> {
  if (!force && statsCache && statsCache.expiresAt > Date.now()) return statsCache.value;

  const store = await getVectorStore();
  const embedder = getEmbeddingProvider();
  const stats = await store.stats();

  const value: IndexStats = {
    documents: stats.documents,
    vectors: stats.vectors,
    chunks: stats.chunks,
    averageChunkSizeChars: stats.averageChunkChars,
    averageChunkTokens: stats.averageChunkTokens,
    languages: stats.languages,
    strategies: stats.strategies,
    collection: store.collection,
    vectorStore: store.name,
    embeddingModel: embedder.model,
    embeddingDimensions: embedder.dimensions,
    lastIndexedAt: stats.lastIndexedAt,
    indexed: stats.vectors > 0,
  };

  statsCache = { value, expiresAt: Date.now() + STATS_TTL_MS };
  return value;
}

export function invalidateStatsCache(): void {
  statsCache = null;
}
