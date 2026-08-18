/**
 * Two-tier retrieval cache.
 *
 * The tiers exist because a *semantic* cache cannot skip the embedding. To
 * decide whether an incoming query is "close enough" to a stored one you have
 * to compare their vectors, and getting a vector means running the encoder.
 * So the two tiers save very different amounts:
 *
 *   L1 — exact match on the normalised query string.
 *        Skips the entire path, embedding included. ~0.05ms.
 *
 *   L2 — cosine match against the vectors of previously-seen queries.
 *        Costs one embedding (~7ms), then skips search, fusion, MMR,
 *        reranking and parent expansion (~12ms).
 *
 * L1 is what makes a repeated question instant; L2 is what catches "what is a
 * corporation" against "what's a corporation?" once normalisation alone no
 * longer matches them.
 *
 * Entries are keyed by query *and* by the retrieval options that produced
 * them — a result computed at topK=10 is not a valid answer for topK=50, and
 * silently returning it would make the settings panel look broken.
 *
 * Deliberately in-process and bounded, like the analytics store: this is a
 * latency optimisation for one server instance, not a shared cache tier. A
 * multi-replica deployment would put Redis here instead, and the interface
 * would not change.
 */
import type { RetrievedChunk } from '@goarag/shared';
import type { RetrievalOptions } from '@goarag/shared';
import { config } from '../../config/env.js';
import { logger } from '../../utils/logger.js';

/** What a completed retrieval contributes back to the cache. */
export interface CachedRetrieval {
  chunks: RetrievedChunk[];
  candidates: RetrievedChunk[];
  agreement: number;
  queryVector: Float32Array;
  rerankerProvider: string;
  degraded: boolean;
}

interface CacheEntry extends CachedRetrieval {
  /** Normalised query text, kept for logging and L1 identity. */
  normalized: string;
  /** Fingerprint of the retrieval options this result was computed under. */
  optionsKey: string;
  storedAt: number;
}

export type CacheTier = 'l1-exact' | 'l2-semantic';

export interface CacheHit extends CachedRetrieval {
  tier: CacheTier;
  /** Cosine similarity to the stored query. Always 1 for an L1 hit. */
  similarity: number;
}

/**
 * Normalise a query for cache identity.
 *
 * NFKC first so that visually identical text composed differently — which is
 * common in Devanagari, where a glyph may arrive precomposed or as a base plus
 * combining mark — collapses to one form. Then case-fold, flatten whitespace,
 * and drop trailing terminal punctuation, since "what is a corporation" and
 * "What is a corporation?" are the same question.
 *
 * Deliberately conservative otherwise: stripping *all* punctuation would merge
 * queries that genuinely differ, and this runs on the hot path where the whole
 * budget for it is under a millisecond.
 */
export function normalizeQuery(query: string): string {
  return query
    .normalize('NFKC')
    .toLowerCase()
    .replace(/\s+/gu, ' ')
    .trim()
    .replace(/[?!.।]+$/u, '')
    .trim();
}

/** Options that change the result, flattened to a stable string. */
function optionsFingerprint(options: RetrievalOptions): string {
  return [
    options.topK ?? config.retrieval.topK,
    options.rerankTopN ?? config.retrieval.rerankTopN,
    options.enableRerank ?? true,
    options.enableMmr ?? config.retrieval.enableMmr,
    options.enableParentExpansion ?? config.retrieval.enableParentExpansion,
    (options.languages ?? []).join('+'),
  ].join('|');
}

/** Cosine similarity of two already-normalised vectors. */
function cosine(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return 0;
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) sum += (a[i] as number) * (b[i] as number);
  return sum;
}

class RetrievalCache {
  /** Insertion-ordered, so the oldest key is the first one Map yields. */
  private readonly entries = new Map<string, CacheEntry>();
  private l1Hits = 0;
  private l2Hits = 0;
  private misses = 0;

  /**
   * Runtime override of the configured setting, or null to defer to config.
   * The benchmark uses this to hold the cache off for the duration of a timed
   * run without mutating (readonly) config or requiring a restart.
   */
  private override: boolean | null = null;

  private get enabled(): boolean {
    return this.override ?? config.retrieval.cache.enabled;
  }

  setEnabled(value: boolean | null): void {
    this.override = value;
  }

  private isFresh(entry: CacheEntry): boolean {
    return Date.now() - entry.storedAt < config.retrieval.cache.ttlMs;
  }

  /** L1: exact match on the normalised string. Skips embedding entirely. */
  lookupExact(normalized: string, options: RetrievalOptions): CacheHit | null {
    if (!this.enabled) return null;
    const key = `${optionsFingerprint(options)}::${normalized}`;
    const entry = this.entries.get(key);
    if (!entry) return null;
    if (!this.isFresh(entry)) {
      this.entries.delete(key);
      return null;
    }
    // Refresh recency so a repeatedly-asked question is the last to be evicted.
    this.entries.delete(key);
    this.entries.set(key, entry);
    this.l1Hits += 1;
    return { ...entry, tier: 'l1-exact', similarity: 1 };
  }

  /**
   * L2: nearest stored query by cosine, above the configured threshold.
   *
   * A linear scan over at most `maxEntries` vectors — 256 × 384 floats is
   * ~0.1ms, far below the cost of the search it avoids, and it keeps the
   * structure simple enough to reason about. An index here would be
   * optimising the wrong thing.
   */
  lookupSemantic(vector: Float32Array, options: RetrievalOptions): CacheHit | null {
    if (!this.enabled) return null;
    const fingerprint = optionsFingerprint(options);
    const threshold = config.retrieval.cache.similarity;

    let best: CacheEntry | null = null;
    let bestScore = threshold;

    for (const [key, entry] of this.entries) {
      if (entry.optionsKey !== fingerprint) continue;
      if (!this.isFresh(entry)) {
        this.entries.delete(key);
        continue;
      }
      const score = cosine(vector, entry.queryVector);
      if (score > bestScore) {
        bestScore = score;
        best = entry;
      }
    }

    if (!best) return null;
    this.l2Hits += 1;
    logger.debug({ similarity: Number(bestScore.toFixed(4)) }, 'Semantic cache hit');
    return { ...best, tier: 'l2-semantic', similarity: bestScore };
  }

  store(normalized: string, options: RetrievalOptions, result: CachedRetrieval): void {
    if (!this.enabled) return;
    const optionsKey = optionsFingerprint(options);
    const key = `${optionsKey}::${normalized}`;

    this.entries.delete(key);
    this.entries.set(key, { ...result, normalized, optionsKey, storedAt: Date.now() });

    while (this.entries.size > config.retrieval.cache.maxEntries) {
      const oldest = this.entries.keys().next().value;
      if (oldest === undefined) break;
      this.entries.delete(oldest);
    }
  }

  recordMiss(): void {
    this.misses += 1;
  }

  stats(): { l1Hits: number; l2Hits: number; misses: number; size: number } {
    return { l1Hits: this.l1Hits, l2Hits: this.l2Hits, misses: this.misses, size: this.entries.size };
  }

  /** Drop everything. Used by the benchmark so its numbers stand alone. */
  clear(): void {
    this.entries.clear();
    this.l1Hits = 0;
    this.l2Hits = 0;
    this.misses = 0;
  }
}

export const retrievalCache = new RetrievalCache();
