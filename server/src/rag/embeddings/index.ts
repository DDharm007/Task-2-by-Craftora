/**
 * Embedding provider factory, with an LRU cache in front.
 *
 * The cache matters more than it looks: benchmark runs, repeated UI queries
 * and conversation follow-ups embed the same strings constantly, and a cache
 * hit turns a ~200ms CPU forward pass into a map lookup.
 */
import { createHash } from 'node:crypto';
import { config } from '../../config/env.js';
import { logger } from '../../utils/logger.js';
import { LocalBgeM3Provider } from './local-bge-m3.js';
import type { EmbeddingInputType, EmbeddingProvider } from './types.js';

/** Bounded LRU over embedded texts. */
class EmbeddingCache {
  private readonly map = new Map<string, Float32Array>();

  constructor(private readonly capacity: number) {}

  get(key: string): Float32Array | undefined {
    const hit = this.map.get(key);
    if (hit) {
      // Refresh recency.
      this.map.delete(key);
      this.map.set(key, hit);
    }
    return hit;
  }

  set(key: string, value: Float32Array): void {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, value);
    if (this.map.size > this.capacity) {
      const oldest = this.map.keys().next().value;
      if (oldest !== undefined) this.map.delete(oldest);
    }
  }

  get size(): number {
    return this.map.size;
  }
}

/** Wraps a provider so repeated texts skip the model entirely. */
class CachedEmbeddingProvider implements EmbeddingProvider {
  readonly name: string;
  readonly model: string;
  readonly dimensions: number;

  private readonly cache = new EmbeddingCache(4_096);
  private hits = 0;
  private misses = 0;

  constructor(private readonly inner: EmbeddingProvider) {
    this.name = inner.name;
    this.model = inner.model;
    this.dimensions = inner.dimensions;
  }

  async embed(texts: readonly string[], inputType: EmbeddingInputType): Promise<Float32Array[]> {
    if (texts.length === 0) return [];

    const results = new Array<Float32Array | undefined>(texts.length);
    const missingIndices: number[] = [];
    const missingTexts: string[] = [];

    texts.forEach((text, i) => {
      const key = cacheKey(text, inputType);
      const cached = this.cache.get(key);
      if (cached) {
        results[i] = cached;
        this.hits += 1;
      } else {
        missingIndices.push(i);
        missingTexts.push(text);
        this.misses += 1;
      }
    });

    if (missingTexts.length > 0) {
      const computed = await this.inner.embed(missingTexts, inputType);
      computed.forEach((vector, i) => {
        const target = missingIndices[i] as number;
        results[target] = vector;
        this.cache.set(cacheKey(texts[target] as string, inputType), vector);
      });
    }

    return results as Float32Array[];
  }

  warmup(): Promise<void> {
    return this.inner.warmup();
  }

  healthCheck(): Promise<{ ok: boolean; detail: string }> {
    return this.inner.healthCheck();
  }

  stats(): { hits: number; misses: number; size: number } {
    return { hits: this.hits, misses: this.misses, size: this.cache.size };
  }
}

function cacheKey(text: string, inputType: EmbeddingInputType): string {
  return `${inputType}:${createHash('sha1').update(text).digest('base64')}`;
}

let provider: EmbeddingProvider | null = null;

/** The process-wide embedding provider. */
export function getEmbeddingProvider(): EmbeddingProvider {
  if (!provider) {
    const inner = new LocalBgeM3Provider();
    logger.info(
      { provider: inner.name, model: inner.model, dimensions: inner.dimensions },
      'Embedding provider selected',
    );
    provider = new CachedEmbeddingProvider(inner);
  }
  return provider;
}

/** Embed a single text — the common case for queries. */
export async function embedOne(text: string, inputType: EmbeddingInputType): Promise<Float32Array> {
  const [vector] = await getEmbeddingProvider().embed([text], inputType);
  if (!vector) throw new Error('Embedding provider returned no vector');
  return vector;
}

export { CachedEmbeddingProvider };
export type { EmbeddingProvider, EmbeddingInputType } from './types.js';
export { dot, normalizeVector } from './types.js';
