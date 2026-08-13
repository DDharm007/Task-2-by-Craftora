/**
 * BM25 lexical scoring, expressed as sparse vectors.
 *
 * The keyword half of hybrid retrieval. Dense embeddings are strong at
 * paraphrase but weak at rare literal tokens — model numbers, proper nouns,
 * transliterated names — which is exactly what BM25 is good at.
 *
 * The BM25 score factorises cleanly into a sparse dot product:
 *
 *     score(q,d) = Σ_t  idf(t) · tfWeight(t,d)
 *
 * so we store `tfWeight` as the document-side sparse vector and `idf` as the
 * query-side one. That lets Qdrant compute BM25 natively with its sparse
 * vector index, instead of us pulling candidates back to score in Node.
 *
 * Tokenisation is Unicode-aware (see utils/text.ts) so Devanagari, Bengali,
 * Tamil and Arabic script tokenise on the same footing as Latin.
 */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { tokenize } from '../../utils/text.js';

/** Standard BM25 parameters. k1 controls term-frequency saturation, b length normalisation. */
const K1 = 1.2;
const B = 0.75;

export interface SparseVector {
  indices: number[];
  values: number[];
}

export interface Bm25Snapshot {
  version: 1;
  documentCount: number;
  averageLength: number;
  /** term → [vocabularyId, documentFrequency] */
  vocabulary: Record<string, [number, number]>;
  k1: number;
  b: number;
}

export class Bm25Model {
  private documentFrequency = new Map<string, number>();
  private vocabulary = new Map<string, number>();
  private documentCount = 0;
  private averageLength = 1;
  private nextId = 1;

  get size(): number {
    return this.vocabulary.size;
  }

  get docCount(): number {
    return this.documentCount;
  }

  get isEmpty(): boolean {
    return this.documentCount === 0;
  }

  /**
   * Build corpus statistics from every document that will be indexed.
   *
   * BM25 needs global document frequency and average length up front, so
   * indexing is necessarily two-pass: collect stats, then encode vectors.
   */
  fit(documents: readonly string[]): void {
    this.documentFrequency.clear();
    this.vocabulary.clear();
    this.nextId = 1;
    this.documentCount = documents.length;

    let totalLength = 0;
    for (const doc of documents) {
      const tokens = tokenize(doc);
      totalLength += tokens.length;
      // Document frequency counts documents, not occurrences.
      for (const term of new Set(tokens)) {
        this.documentFrequency.set(term, (this.documentFrequency.get(term) ?? 0) + 1);
      }
    }
    this.averageLength = documents.length > 0 ? Math.max(1, totalLength / documents.length) : 1;

    // Assign stable vocabulary ids in a deterministic order so a rebuild on
    // another machine produces byte-identical sparse indices.
    for (const term of [...this.documentFrequency.keys()].sort()) {
      this.vocabulary.set(term, this.nextId);
      this.nextId += 1;
    }
  }

  /** Inverse document frequency, in the BM25+ smoothed form (always ≥ 0). */
  idf(term: string): number {
    const df = this.documentFrequency.get(term) ?? 0;
    if (df === 0) return 0;
    return Math.log(1 + (this.documentCount - df + 0.5) / (df + 0.5));
  }

  /**
   * Document-side sparse vector: term-frequency weight with length
   * normalisation. Paired with `queryVector`, the dot product is BM25.
   */
  documentVector(text: string): SparseVector {
    const tokens = tokenize(text);
    if (tokens.length === 0) return { indices: [], values: [] };

    const counts = new Map<string, number>();
    for (const token of tokens) counts.set(token, (counts.get(token) ?? 0) + 1);

    const lengthNorm = K1 * (1 - B + (B * tokens.length) / this.averageLength);
    const indices: number[] = [];
    const values: number[] = [];

    for (const [term, tf] of counts) {
      const id = this.vocabulary.get(term);
      if (id === undefined) continue;
      const weight = (tf * (K1 + 1)) / (tf + lengthNorm);
      if (weight > 0) {
        indices.push(id);
        values.push(weight);
      }
    }
    return sortSparse({ indices, values });
  }

  /** Query-side sparse vector: pure IDF over the query's terms. */
  queryVector(text: string): SparseVector {
    const tokens = tokenize(text);
    if (tokens.length === 0) return { indices: [], values: [] };

    const seen = new Map<number, number>();
    for (const term of tokens) {
      const id = this.vocabulary.get(term);
      if (id === undefined) continue; // unseen term matches nothing
      const weight = this.idf(term);
      if (weight <= 0) continue;
      // Repeating a term in the query mildly increases its weight.
      seen.set(id, (seen.get(id) ?? 0) + weight);
    }
    return sortSparse({ indices: [...seen.keys()], values: [...seen.values()] });
  }

  /**
   * Exact BM25 between a query and a document, used by the embedded store
   * where we can score directly instead of going through sparse vectors.
   */
  score(queryTokens: readonly string[], documentText: string): number {
    const tokens = tokenize(documentText);
    if (tokens.length === 0 || queryTokens.length === 0) return 0;

    const counts = new Map<string, number>();
    for (const token of tokens) counts.set(token, (counts.get(token) ?? 0) + 1);

    const lengthNorm = K1 * (1 - B + (B * tokens.length) / this.averageLength);
    let total = 0;
    for (const term of new Set(queryTokens)) {
      const tf = counts.get(term);
      if (!tf) continue;
      total += this.idf(term) * ((tf * (K1 + 1)) / (tf + lengthNorm));
    }
    return total;
  }

  toSnapshot(): Bm25Snapshot {
    const vocabulary: Record<string, [number, number]> = {};
    for (const [term, id] of this.vocabulary) {
      vocabulary[term] = [id, this.documentFrequency.get(term) ?? 0];
    }
    return {
      version: 1,
      documentCount: this.documentCount,
      averageLength: this.averageLength,
      vocabulary,
      k1: K1,
      b: B,
    };
  }

  static fromSnapshot(snapshot: Bm25Snapshot): Bm25Model {
    const model = new Bm25Model();
    model.documentCount = snapshot.documentCount;
    model.averageLength = snapshot.averageLength;
    let maxId = 0;
    for (const [term, [id, df]] of Object.entries(snapshot.vocabulary)) {
      model.vocabulary.set(term, id);
      model.documentFrequency.set(term, df);
      if (id > maxId) maxId = id;
    }
    model.nextId = maxId + 1;
    return model;
  }

  async save(filePath: string): Promise<void> {
    await mkdir(path.dirname(filePath), { recursive: true });
    await writeFile(filePath, JSON.stringify(this.toSnapshot()), 'utf8');
  }

  static async load(filePath: string): Promise<Bm25Model | null> {
    try {
      const raw = await readFile(filePath, 'utf8');
      const snapshot = JSON.parse(raw) as Bm25Snapshot;
      if (snapshot.version !== 1) return null;
      return Bm25Model.fromSnapshot(snapshot);
    } catch {
      return null;
    }
  }
}

/** Qdrant requires sparse indices in ascending order. */
function sortSparse(vector: SparseVector): SparseVector {
  const pairs = vector.indices
    .map((index, i) => [index, vector.values[i] as number] as const)
    .sort((a, b) => a[0] - b[0]);
  return {
    indices: pairs.map(([index]) => index),
    values: pairs.map(([, value]) => value),
  };
}

/** Dot product of two sparse vectors, both sorted by index. */
export function sparseDot(a: SparseVector, b: SparseVector): number {
  let i = 0;
  let j = 0;
  let total = 0;
  while (i < a.indices.length && j < b.indices.length) {
    const ai = a.indices[i] as number;
    const bj = b.indices[j] as number;
    if (ai === bj) {
      total += (a.values[i] as number) * (b.values[j] as number);
      i += 1;
      j += 1;
    } else if (ai < bj) {
      i += 1;
    } else {
      j += 1;
    }
  }
  return total;
}
