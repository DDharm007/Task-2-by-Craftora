/**
 * Embedded vector store — a disk-persisted, in-process implementation of the
 * same `VectorStore` contract Qdrant satisfies.
 *
 * This exists so the full pipeline runs with no external services: `npm run
 * dev` works on a laptop without Docker, CI can index and query, and the
 * benchmark is reproducible anywhere. Qdrant remains the production driver
 * (`VECTOR_STORE=qdrant`), and because both sit behind one interface, swapping
 * between them changes nothing else in the system.
 *
 * Implementation notes:
 *   • Dense search is an exact brute-force cosine scan. For the corpus sizes
 *     this store targets (tens of thousands of vectors) an exact scan is both
 *     faster and more accurate than building an ANN index — and it makes the
 *     benchmark a true measure of retrieval quality, with no recall loss from
 *     approximate search muddying the numbers.
 *   • Sparse search walks an inverted index, visiting only the postings for
 *     query terms rather than every document.
 *   • Vectors are stored as one flat Float32Array so the hot loop stays on a
 *     contiguous buffer instead of chasing thousands of separate arrays.
 */
import { createReadStream } from 'node:fs';
import { mkdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import path from 'node:path';
import type { ChunkMetadata, ChunkStrategy } from '@voxrag/shared';
import { logger } from '../../utils/logger.js';
import { errors } from '../../utils/errors.js';
import type { SparseVector } from './bm25.js';
import { sparseDot } from './bm25.js';
import type { SearchFilter, SearchHit, StoreStats, VectorRecord, VectorStore } from './types.js';

interface StoredRecord {
  id: string;
  chunkKey: string;
  text: string;
  metadata: ChunkMetadata;
  isParent: boolean;
  sparse: SparseVector;
}

/** One line of the on-disk JSONL, vectors included. */
interface SerializedRecord extends StoredRecord {
  vector: number[];
}

export class EmbeddedVectorStore implements VectorStore {
  readonly name = 'embedded';
  readonly collection: string;

  private readonly dir: string;
  private readonly dataFile: string;

  private records: StoredRecord[] = [];
  private index = new Map<string, number>();
  /** All dense vectors laid out end to end: record `i` occupies [i*dim, (i+1)*dim). */
  private matrix = new Float32Array(0);
  private dimensions = 0;
  /** term id → record indices containing it. */
  private inverted = new Map<number, number[]>();
  private loaded = false;
  private dirty = false;

  constructor(collection: string, dir: string) {
    this.collection = collection;
    this.dir = dir;
    this.dataFile = path.join(dir, `${collection}.jsonl`);
  }

  async init(dimensions: number): Promise<void> {
    this.dimensions = dimensions;
    await mkdir(this.dir, { recursive: true });
    if (!this.loaded) await this.load();
  }

  /** Stream the JSONL back in. Streaming matters: the file can be >100MB. */
  private async load(): Promise<void> {
    this.loaded = true;
    let exists = true;
    try {
      await stat(this.dataFile);
    } catch {
      exists = false;
    }
    if (!exists) return;

    const started = Date.now();
    const parsed: SerializedRecord[] = [];
    const stream = createReadStream(this.dataFile, { encoding: 'utf8' });
    const lines = createInterface({ input: stream, crlfDelay: Infinity });

    for await (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        parsed.push(JSON.parse(trimmed) as SerializedRecord);
      } catch {
        // A torn final line from an interrupted write — skip it.
        logger.warn({ file: this.dataFile }, 'Skipping malformed record in embedded store');
      }
    }

    this.rebuild(parsed);
    logger.info(
      { records: this.records.length, ms: Date.now() - started, file: this.dataFile },
      'Loaded embedded vector store',
    );
  }

  /** Rebuild all in-memory structures from a full record list. */
  private rebuild(parsed: SerializedRecord[]): void {
    const dim = this.dimensions || parsed[0]?.vector.length || 0;
    this.dimensions = dim;
    this.records = [];
    this.index.clear();
    this.inverted.clear();
    this.matrix = new Float32Array(parsed.length * dim);

    parsed.forEach((record, i) => {
      this.records.push({
        id: record.id,
        chunkKey: record.chunkKey,
        text: record.text,
        metadata: record.metadata,
        isParent: record.isParent,
        sparse: record.sparse,
      });
      this.index.set(record.id, i);
      this.matrix.set(record.vector.slice(0, dim), i * dim);
      for (const term of record.sparse.indices) {
        let postings = this.inverted.get(term);
        if (!postings) {
          postings = [];
          this.inverted.set(term, postings);
        }
        postings.push(i);
      }
    });
  }

  async reset(): Promise<void> {
    this.records = [];
    this.index.clear();
    this.inverted.clear();
    this.matrix = new Float32Array(0);
    this.dirty = false;
    await rm(this.dataFile, { force: true });
  }

  async upsert(records: readonly VectorRecord[]): Promise<void> {
    if (records.length === 0) return;
    if (!this.loaded) await this.load();
    if (this.dimensions === 0) this.dimensions = records[0]?.vector.length ?? 0;

    // Deterministic ids make re-indexing idempotent: rewrite in place when the
    // id is already known, otherwise append.
    const appended: VectorRecord[] = [];
    for (const record of records) {
      const existing = this.index.get(record.id);
      if (existing === undefined) {
        appended.push(record);
      } else {
        this.records[existing] = toStored(record);
        this.matrix.set(record.vector, existing * this.dimensions);
      }
    }

    if (appended.length > 0) {
      const previous = this.matrix;
      const offset = this.records.length;
      this.matrix = new Float32Array((offset + appended.length) * this.dimensions);
      this.matrix.set(previous, 0);

      appended.forEach((record, i) => {
        const position = offset + i;
        this.records.push(toStored(record));
        this.index.set(record.id, position);
        this.matrix.set(record.vector, position * this.dimensions);
        for (const term of record.sparse.indices) {
          let postings = this.inverted.get(term);
          if (!postings) {
            postings = [];
            this.inverted.set(term, postings);
          }
          postings.push(position);
        }
      });
    }

    this.dirty = true;
  }

  /** Persist to disk atomically, so a crash mid-write cannot corrupt the store. */
  async flush(): Promise<void> {
    if (!this.dirty) return;
    await mkdir(this.dir, { recursive: true });
    const temporary = `${this.dataFile}.tmp`;

    const lines: string[] = [];
    for (let i = 0; i < this.records.length; i += 1) {
      const record = this.records[i] as StoredRecord;
      const vector = Array.from(this.matrix.subarray(i * this.dimensions, (i + 1) * this.dimensions));
      // Four decimals is well inside the precision cosine ranking needs, and
      // roughly halves the file versus full float printing.
      lines.push(
        JSON.stringify({ ...record, vector: vector.map((v) => Math.round(v * 10_000) / 10_000) }),
      );
    }

    await writeFile(temporary, `${lines.join('\n')}\n`, 'utf8');
    await rename(temporary, this.dataFile);
    this.dirty = false;
    logger.info({ records: this.records.length, file: this.dataFile }, 'Persisted embedded vector store');
  }

  async searchDense(vector: Float32Array, limit: number, filter?: SearchFilter): Promise<SearchHit[]> {
    if (!this.loaded) await this.load();
    if (this.records.length === 0) return [];
    if (vector.length !== this.dimensions) {
      throw errors.vectorStore(
        `Query vector has ${vector.length} dimensions but the index stores ${this.dimensions}. ` +
          `Re-index after changing the embedding model.`,
      );
    }

    const scored = new TopK(limit);
    for (let i = 0; i < this.records.length; i += 1) {
      if (!matches(this.records[i] as StoredRecord, filter)) continue;
      // Both sides are L2-normalised, so the dot product is cosine similarity.
      const offset = i * this.dimensions;
      let score = 0;
      for (let d = 0; d < this.dimensions; d += 1) {
        score += (vector[d] as number) * (this.matrix[offset + d] as number);
      }
      scored.push(i, score);
    }

    return scored.drain().map(({ index, score }) => this.toHit(index, score));
  }

  async searchSparse(vector: SparseVector, limit: number, filter?: SearchFilter): Promise<SearchHit[]> {
    if (!this.loaded) await this.load();
    if (this.records.length === 0 || vector.indices.length === 0) return [];

    // Visit only documents that share a term with the query.
    const candidates = new Set<number>();
    for (const term of vector.indices) {
      const postings = this.inverted.get(term);
      if (!postings) continue;
      for (const position of postings) candidates.add(position);
    }

    const scored = new TopK(limit);
    for (const position of candidates) {
      const record = this.records[position] as StoredRecord;
      if (!matches(record, filter)) continue;
      const score = sparseDot(vector, record.sparse);
      if (score > 0) scored.push(position, score);
    }

    return scored.drain().map(({ index, score }) => this.toHit(index, score));
  }

  async fetchByIds(ids: readonly string[]): Promise<SearchHit[]> {
    if (!this.loaded) await this.load();
    const out: SearchHit[] = [];
    for (const id of ids) {
      const position = this.index.get(id);
      if (position !== undefined) out.push(this.toHit(position, 1));
    }
    return out;
  }

  async *scrollAll(batchSize: number): AsyncGenerator<SearchHit[], void, unknown> {
    if (!this.loaded) await this.load();
    for (let i = 0; i < this.records.length; i += batchSize) {
      yield this.records
        .slice(i, i + batchSize)
        .map((_, offset) => this.toHit(i + offset, 0));
    }
  }

  async stats(): Promise<StoreStats> {
    if (!this.loaded) await this.load();

    const documents = new Set<string>();
    const languages = new Map<string, number>();
    const strategies = new Map<ChunkStrategy, number>();
    let parents = 0;
    let chunks = 0;
    let totalChars = 0;
    let totalTokens = 0;
    let lastIndexedAt: string | null = null;

    for (const record of this.records) {
      documents.add(record.metadata.documentId);
      languages.set(record.metadata.language, (languages.get(record.metadata.language) ?? 0) + 1);
      strategies.set(record.metadata.strategy, (strategies.get(record.metadata.strategy) ?? 0) + 1);
      if (record.isParent) {
        parents += 1;
      } else {
        chunks += 1;
        totalChars += record.text.length;
        totalTokens += record.metadata.tokenCount;
      }
      if (!lastIndexedAt || record.metadata.indexedAt > lastIndexedAt) {
        lastIndexedAt = record.metadata.indexedAt;
      }
    }

    return {
      vectors: this.records.length,
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
    if (!this.loaded) await this.load();
    return {
      ok: true,
      detail: `embedded · ${this.records.length} vectors · ${this.dataFile}`,
    };
  }

  async close(): Promise<void> {
    await this.flush();
  }

  private toHit(position: number, score: number): SearchHit {
    const record = this.records[position] as StoredRecord;
    return {
      id: record.id,
      text: record.text,
      metadata: record.metadata,
      score,
      isParent: record.isParent,
    };
  }
}

function toStored(record: VectorRecord): StoredRecord {
  return {
    id: record.id,
    chunkKey: record.chunkKey,
    text: record.text,
    metadata: record.metadata,
    isParent: record.isParent,
    sparse: record.sparse,
  };
}

function matches(record: StoredRecord, filter?: SearchFilter): boolean {
  if (!filter?.includeParents && record.isParent) return false;
  if (filter?.languages?.length && !filter.languages.includes(record.metadata.language)) return false;
  if (filter?.strategies?.length && !filter.strategies.includes(record.metadata.strategy)) return false;
  if (filter?.documentIds?.length && !filter.documentIds.includes(record.metadata.documentId)) return false;
  return true;
}

/**
 * Bounded min-heap for top-k selection.
 *
 * Sorting every candidate would be O(n log n) on each search; this keeps the
 * scan O(n log k), which matters because dense search touches every vector.
 */
class TopK {
  private readonly heap: Array<{ index: number; score: number }> = [];

  constructor(private readonly capacity: number) {}

  push(index: number, score: number): void {
    if (this.capacity <= 0) return;
    if (this.heap.length < this.capacity) {
      this.heap.push({ index, score });
      this.siftUp(this.heap.length - 1);
      return;
    }
    const smallest = this.heap[0];
    if (smallest && score > smallest.score) {
      this.heap[0] = { index, score };
      this.siftDown(0);
    }
  }

  /** Results sorted by descending score. */
  drain(): Array<{ index: number; score: number }> {
    return [...this.heap].sort((a, b) => b.score - a.score);
  }

  private siftUp(start: number): void {
    let i = start;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if ((this.heap[i] as { score: number }).score >= (this.heap[parent] as { score: number }).score) break;
      this.swap(i, parent);
      i = parent;
    }
  }

  private siftDown(start: number): void {
    let i = start;
    for (;;) {
      const left = 2 * i + 1;
      const right = left + 1;
      let smallest = i;
      if (left < this.heap.length && (this.heap[left] as { score: number }).score < (this.heap[smallest] as { score: number }).score) {
        smallest = left;
      }
      if (right < this.heap.length && (this.heap[right] as { score: number }).score < (this.heap[smallest] as { score: number }).score) {
        smallest = right;
      }
      if (smallest === i) break;
      this.swap(i, smallest);
      i = smallest;
    }
  }

  private swap(a: number, b: number): void {
    const temp = this.heap[a] as { index: number; score: number };
    this.heap[a] = this.heap[b] as { index: number; score: number };
    this.heap[b] = temp;
  }
}
