/** The storage contract shared by the Qdrant and embedded drivers. */
import type { ChunkMetadata, ChunkStrategy } from '@goarag/shared';
import type { SparseVector } from './bm25.js';

/** A point as written to the store. */
export interface VectorRecord {
  id: string;
  chunkKey: string;
  text: string;
  /** Text that was embedded (chunk text plus its metadata header). */
  embedText: string;
  vector: Float32Array;
  sparse: SparseVector;
  metadata: ChunkMetadata;
  isParent: boolean;
}

/** A point as read back from a search. */
export interface SearchHit {
  id: string;
  text: string;
  metadata: ChunkMetadata;
  score: number;
  isParent: boolean;
}

export interface SearchFilter {
  /** Restrict to these dataset language tags. */
  languages?: string[];
  /** Restrict to these chunking strategies. */
  strategies?: ChunkStrategy[];
  /** Include coarse parent spans in results. Defaults to false. */
  includeParents?: boolean;
  /** Restrict to specific document ids. */
  documentIds?: string[];
}

export interface StoreStats {
  vectors: number;
  documents: number;
  chunks: number;
  parents: number;
  averageChunkChars: number;
  averageChunkTokens: number;
  languages: Array<{ language: string; count: number }>;
  strategies: Array<{ strategy: ChunkStrategy; count: number }>;
  lastIndexedAt: string | null;
}

export interface VectorStore {
  readonly name: string;
  readonly collection: string;

  /** Create the collection/indexes if absent. Safe to call repeatedly. */
  init(dimensions: number): Promise<void>;

  /** Delete all indexed data. */
  reset(): Promise<void>;

  /** Insert or update points. Ids are deterministic, so this is idempotent. */
  upsert(records: readonly VectorRecord[]): Promise<void>;

  /** Dense (cosine) nearest-neighbour search. */
  searchDense(vector: Float32Array, limit: number, filter?: SearchFilter): Promise<SearchHit[]>;

  /** Sparse (BM25) search. */
  searchSparse(vector: SparseVector, limit: number, filter?: SearchFilter): Promise<SearchHit[]>;

  /** Fetch specific points by id — used for parent-chunk expansion. */
  fetchByIds(ids: readonly string[]): Promise<SearchHit[]>;

  /** Aggregate statistics for the dashboard. */
  stats(): Promise<StoreStats>;

  /** Stream every stored record, used to rebuild the BM25 model. */
  scrollAll(batchSize: number): AsyncGenerator<SearchHit[], void, unknown>;

  healthCheck(): Promise<{ ok: boolean; detail: string }>;

  close(): Promise<void>;
}
