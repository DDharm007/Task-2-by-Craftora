/**
 * Qdrant driver — the production vector store.
 *
 * The collection carries two named vectors per point:
 *   • `dense`  — 1024-dim BGE-M3 embedding, cosine distance
 *   • `sparse` — BM25 term weights, stored as a native Qdrant sparse vector
 *
 * Keeping both on one point means hybrid retrieval is two indexed queries
 * against the same collection rather than a join across two systems, and
 * filters (language, strategy) apply identically to both arms.
 *
 * Payload indexes are created for every field we filter on; without them
 * Qdrant falls back to a full scan once the collection grows.
 */
import { QdrantClient } from '@qdrant/js-client-rest';
import type { ChunkMetadata, ChunkStrategy } from '@goarag/shared';
import { config } from '../../config/env.js';
import { logger } from '../../utils/logger.js';
import { chunkArray, retry, withTimeout } from '../../utils/async.js';
import { errors } from '../../utils/errors.js';
import type { SparseVector } from './bm25.js';
import type { SearchFilter, SearchHit, StoreStats, VectorRecord, VectorStore } from './types.js';

const DENSE_VECTOR = 'dense';
const SPARSE_VECTOR = 'sparse';

/** Payload fields we filter or aggregate on. */
const INDEXED_PAYLOAD_FIELDS: Array<{ field: string; schema: 'keyword' | 'bool' | 'integer' }> = [
  { field: 'language', schema: 'keyword' },
  { field: 'strategy', schema: 'keyword' },
  { field: 'documentId', schema: 'keyword' },
  { field: 'passageId', schema: 'keyword' },
  { field: 'queryId', schema: 'keyword' },
  { field: 'isParent', schema: 'bool' },
  { field: 'isSelected', schema: 'bool' },
];

interface QdrantPayload extends Record<string, unknown> {
  text: string;
  chunkKey: string;
  isParent: boolean;
  metadata: ChunkMetadata;
  // Flattened copies so Qdrant can index/filter them directly.
  language: string;
  strategy: ChunkStrategy;
  documentId: string;
  passageId: string;
  queryId: string | null;
  isSelected: boolean;
}

export class QdrantVectorStore implements VectorStore {
  readonly name = 'qdrant';
  readonly collection: string;
  private readonly client: QdrantClient;
  private initialised = false;

  constructor() {
    this.collection = config.vectorStore.collection;
    this.client = new QdrantClient({
      url: config.vectorStore.qdrantUrl,
      ...(config.vectorStore.qdrantApiKey ? { apiKey: config.vectorStore.qdrantApiKey } : {}),
      checkCompatibility: false,
    });
  }

  async init(dimensions: number): Promise<void> {
    if (this.initialised) return;

    const existing = await this.client.getCollections();
    const found = existing.collections.some((c) => c.name === this.collection);

    if (!found) {
      logger.info({ collection: this.collection, dimensions }, 'Creating Qdrant collection');
      await this.client.createCollection(this.collection, {
        vectors: {
          [DENSE_VECTOR]: { size: dimensions, distance: 'Cosine', on_disk: true },
        },
        sparse_vectors: {
          [SPARSE_VECTOR]: { index: { on_disk: true } },
        },
        // Defer indexing during bulk load, then build once — much faster than
        // maintaining the HNSW graph incrementally across every batch.
        optimizers_config: { default_segment_number: 2, indexing_threshold: 20_000 },
        hnsw_config: { m: 16, ef_construct: 128, on_disk: false },
      });
    }

    for (const { field, schema } of INDEXED_PAYLOAD_FIELDS) {
      try {
        await this.client.createPayloadIndex(this.collection, {
          field_name: field,
          field_schema: schema,
          wait: false,
        });
      } catch {
        // Index already exists — Qdrant has no idempotent create for these.
      }
    }

    this.initialised = true;
  }

  async reset(): Promise<void> {
    try {
      await this.client.deleteCollection(this.collection);
    } catch {
      // Nothing to delete.
    }
    this.initialised = false;
  }

  async upsert(records: readonly VectorRecord[]): Promise<void> {
    if (records.length === 0) return;

    // Qdrant accepts large batches but very large bodies stall the HTTP
    // connection; 128 points keeps each request comfortably small.
    for (const batch of chunkArray([...records], 128)) {
      const points = batch.map((record) => ({
        id: record.id,
        vector: {
          [DENSE_VECTOR]: Array.from(record.vector),
          ...(record.sparse.indices.length > 0
            ? { [SPARSE_VECTOR]: { indices: record.sparse.indices, values: record.sparse.values } }
            : {}),
        },
        payload: {
          text: record.text,
          chunkKey: record.chunkKey,
          isParent: record.isParent,
          metadata: record.metadata,
          language: record.metadata.language,
          strategy: record.metadata.strategy,
          documentId: record.metadata.documentId,
          passageId: record.metadata.passageId,
          queryId: record.metadata.queryId,
          isSelected: record.metadata.isSelected,
        } satisfies QdrantPayload,
      }));

      await retry(
        () => this.client.upsert(this.collection, { wait: true, points }),
        {
          retries: 2,
          baseDelayMs: 500,
          onRetry: (error, attempt, delayMs) =>
            logger.warn({ attempt, delayMs, error: (error as Error).message }, 'Retrying Qdrant upsert'),
        },
      ).catch((error) => {
        throw errors.vectorStore(`Qdrant upsert failed: ${(error as Error).message}`, error);
      });
    }
  }

  async searchDense(vector: Float32Array, limit: number, filter?: SearchFilter): Promise<SearchHit[]> {
    try {
      const response = await this.client.query(this.collection, {
        query: Array.from(vector),
        using: DENSE_VECTOR,
        limit,
        with_payload: true,
        filter: buildFilter(filter),
      });
      return (response.points ?? []).map(toHit);
    } catch (error) {
      throw errors.vectorStore(`Qdrant dense search failed: ${(error as Error).message}`, error);
    }
  }

  async searchSparse(vector: SparseVector, limit: number, filter?: SearchFilter): Promise<SearchHit[]> {
    if (vector.indices.length === 0) return [];
    try {
      const response = await this.client.query(this.collection, {
        query: { indices: vector.indices, values: vector.values },
        using: SPARSE_VECTOR,
        limit,
        with_payload: true,
        filter: buildFilter(filter),
      });
      return (response.points ?? []).map(toHit);
    } catch (error) {
      throw errors.vectorStore(`Qdrant sparse search failed: ${(error as Error).message}`, error);
    }
  }

  async fetchByIds(ids: readonly string[]): Promise<SearchHit[]> {
    if (ids.length === 0) return [];
    try {
      const points = await this.client.retrieve(this.collection, {
        ids: [...ids],
        with_payload: true,
      });
      return points.map((point) => toHit({ id: point.id, score: 1, payload: point.payload }));
    } catch (error) {
      throw errors.vectorStore(`Qdrant retrieve failed: ${(error as Error).message}`, error);
    }
  }

  async *scrollAll(batchSize: number): AsyncGenerator<SearchHit[], void, unknown> {
    let offset: string | number | undefined;
    for (;;) {
      const response = await this.client.scroll(this.collection, {
        limit: batchSize,
        with_payload: true,
        with_vector: false,
        ...(offset !== undefined ? { offset } : {}),
      });
      const points = response.points ?? [];
      if (points.length === 0) return;
      yield points.map((point) => toHit({ id: point.id, score: 0, payload: point.payload }));
      const next = response.next_page_offset;
      if (next === null || next === undefined) return;
      offset = next as string | number;
    }
  }

  async stats(): Promise<StoreStats> {
    const empty: StoreStats = {
      vectors: 0,
      documents: 0,
      chunks: 0,
      parents: 0,
      averageChunkChars: 0,
      averageChunkTokens: 0,
      languages: [],
      strategies: [],
      lastIndexedAt: null,
    };

    let info: Awaited<ReturnType<QdrantClient['getCollection']>>;
    try {
      info = await this.client.getCollection(this.collection);
    } catch {
      return empty;
    }

    const vectors = info.points_count ?? 0;
    if (vectors === 0) return empty;

    // Qdrant has no aggregation API, so walk the payloads once. The corpus is
    // small enough for this to be cheap, and the dashboard caches the result.
    const documents = new Set<string>();
    const languages = new Map<string, number>();
    const strategies = new Map<ChunkStrategy, number>();
    let parents = 0;
    let chunks = 0;
    let totalChars = 0;
    let totalTokens = 0;
    let lastIndexedAt: string | null = null;

    for await (const batch of this.scrollAll(512)) {
      for (const hit of batch) {
        documents.add(hit.metadata.documentId);
        languages.set(hit.metadata.language, (languages.get(hit.metadata.language) ?? 0) + 1);
        strategies.set(hit.metadata.strategy, (strategies.get(hit.metadata.strategy) ?? 0) + 1);
        if (hit.isParent) {
          parents += 1;
        } else {
          chunks += 1;
          totalChars += hit.text.length;
          totalTokens += hit.metadata.tokenCount;
        }
        if (!lastIndexedAt || hit.metadata.indexedAt > lastIndexedAt) {
          lastIndexedAt = hit.metadata.indexedAt;
        }
      }
    }

    return {
      vectors,
      documents: documents.size,
      chunks,
      parents,
      averageChunkChars: chunks ? Math.round(totalChars / chunks) : 0,
      averageChunkTokens: chunks ? Math.round(totalTokens / chunks) : 0,
      languages: [...languages.entries()]
        .map(([language, count]) => ({ language, count }))
        .sort((a, b) => b.count - a.count),
      strategies: [...strategies.entries()].map(([strategy, count]) => ({ strategy, count })),
      lastIndexedAt,
    };
  }

  async healthCheck(): Promise<{ ok: boolean; detail: string }> {
    try {
      const info = await withTimeout(this.client.getCollection(this.collection), 5_000);
      return {
        ok: true,
        detail: `${config.vectorStore.qdrantUrl} · ${this.collection} · ${info.points_count ?? 0} points`,
      };
    } catch (error) {
      // Reachable server but missing collection is still "up" — just unindexed.
      try {
        await withTimeout(this.client.getCollections(), 5_000);
        return { ok: true, detail: `${config.vectorStore.qdrantUrl} · collection not created yet` };
      } catch {
        return { ok: false, detail: `unreachable at ${config.vectorStore.qdrantUrl}: ${(error as Error).message}` };
      }
    }
  }

  /** Probe used by the factory to decide whether Qdrant is usable. */
  async ping(timeoutMs = 2_500): Promise<boolean> {
    try {
      await withTimeout(this.client.getCollections(), timeoutMs);
      return true;
    } catch {
      return false;
    }
  }

  async close(): Promise<void> {
    // The REST client holds no persistent connection.
  }
}

/** Translate our filter shape into a Qdrant filter. */
function buildFilter(filter?: SearchFilter): Record<string, unknown> | undefined {
  const must: Array<Record<string, unknown>> = [];

  // Parent spans are context carriers, not retrieval targets — exclude by default.
  if (!filter?.includeParents) {
    must.push({ key: 'isParent', match: { value: false } });
  }
  if (filter?.languages?.length) {
    must.push({ key: 'language', match: { any: filter.languages } });
  }
  if (filter?.strategies?.length) {
    must.push({ key: 'strategy', match: { any: filter.strategies } });
  }
  if (filter?.documentIds?.length) {
    must.push({ key: 'documentId', match: { any: filter.documentIds } });
  }

  return must.length > 0 ? { must } : undefined;
}

function toHit(point: { id: string | number; score?: number; payload?: Record<string, unknown> | null }): SearchHit {
  const payload = (point.payload ?? {}) as Partial<QdrantPayload>;
  return {
    id: String(point.id),
    text: payload.text ?? '',
    metadata: payload.metadata as ChunkMetadata,
    score: point.score ?? 0,
    isParent: Boolean(payload.isParent),
  };
}
