/**
 * Sentence embeddings running in-process via ONNX Runtime (Transformers.js).
 *
 * Local rather than through an API: it removes a network hop from the hot
 * path — which matters a great deal against a 50ms end-to-end budget, where
 * a round trip to a hosted embedding endpoint would spend the whole thing —
 * and makes indexing cost-free. See docs/ARCHITECTURE.md.
 *
 * The model is not hard-coded. Which encoder, how it pools, and whether it
 * wants asymmetric query/passage prefixes are all config (see env.ts), because
 * those three things have to move together: pooling and prefixes are
 * properties of how a model was *trained*, not preferences, and getting either
 * wrong silently degrades retrieval instead of failing loudly.
 *
 * Two configurations are known-good here:
 *
 *   intfloat/multilingual-e5-small  384d  mean  "query: "/"passage: "  (default)
 *   BAAI/bge-m3                    1024d  cls   no prefixes
 *
 * Both are multilingual XLM-RoBERTa encoders, which is what a
 * Hindi-query-over-English-passages corpus needs. The default is the smaller
 * one for latency; see docs/LATENCY.md for the measured trade.
 *
 * Weights are downloaded once into ./models on first use.
 */
import path from 'node:path';
import { config } from '../../config/env.js';
import { logger } from '../../utils/logger.js';
import { chunkArray, once } from '../../utils/async.js';
import { errors } from '../../utils/errors.js';
import { normalizeVector, type EmbeddingInputType, type EmbeddingProvider } from './types.js';

type TransformersModule = typeof import('@huggingface/transformers');

/** Cached module handle — the import itself is heavy. */
const loadTransformers = once(async (): Promise<TransformersModule> => {
  const mod = await import('@huggingface/transformers');
  // Keep weights inside the repo so they survive across runs and are easy to clear.
  mod.env.cacheDir = config.embedding.cacheDir;
  mod.env.allowLocalModels = true;
  // ONNX Runtime spawns its own thread pool; cap it so indexing does not
  // starve the event loop that is still serving HTTP.
  const cpus = Math.max(1, (await import('node:os')).cpus().length);
  if (mod.env.backends?.onnx?.wasm) {
    mod.env.backends.onnx.wasm.numThreads = Math.max(1, Math.min(4, cpus - 1));
  }
  return mod;
});

interface FeatureExtractionPipeline {
  (
    texts: string[],
    options: { pooling?: 'none' | 'mean' | 'cls'; normalize?: boolean; truncation?: boolean; max_length?: number },
  ): Promise<{ dims: number[]; data: Float32Array | number[] }>;
}

export class LocalOnnxEmbeddingProvider implements EmbeddingProvider {
  readonly name = 'local:onnx';
  readonly model: string;
  readonly dimensions: number;

  private readonly quantization: string;
  private readonly batchSize: number;
  private readonly maxTokens: number;
  private readonly queryMaxTokens: number;
  private readonly pooling: 'cls' | 'mean';
  private readonly queryPrefix: string;
  private readonly passagePrefix: string;
  private readonly getPipeline: () => Promise<FeatureExtractionPipeline>;

  constructor() {
    this.model = config.embedding.model;
    this.dimensions = config.embedding.dimensions;
    this.quantization = config.embedding.quantization;
    this.batchSize = config.embedding.batchSize;
    this.maxTokens = config.embedding.maxTokens;
    this.queryMaxTokens = config.embedding.queryMaxTokens;
    this.pooling = config.embedding.pooling;
    this.queryPrefix = normalizePrefix(config.embedding.queryPrefix);
    this.passagePrefix = normalizePrefix(config.embedding.passagePrefix);

    this.getPipeline = once(async () => {
      const { pipeline } = await loadTransformers();
      const repo = resolveOnnxRepo(this.model);
      logger.info(
        { model: repo, quantization: this.quantization, cacheDir: config.embedding.cacheDir },
        'Loading local embedding model (first run downloads weights)',
      );
      const started = Date.now();
      const extractor = (await pipeline('feature-extraction', repo, {
        dtype: this.quantization as 'fp32' | 'fp16' | 'q8' | 'int8' | 'uint8' | 'q4' | 'q4f16',
        device: 'cpu',
      })) as unknown as FeatureExtractionPipeline;
      logger.info({ model: repo, ms: Date.now() - started }, 'Embedding model ready');
      return extractor;
    });
  }

  /**
   * Warm the model.
   *
   * ONNX Runtime allocates its arenas and picks kernels on the first forward
   * pass, which costs multiples of a steady-state call — so this runs several
   * passes, not one. Without it the first real query carries that cost and
   * drags p100 well past the latency budget. Called at boot (index.ts) and
   * before the benchmark's timed window.
   */
  async warmup(): Promise<void> {
    const extractor = await this.getPipeline();
    for (let i = 0; i < 3; i += 1) {
      await extractor([`${this.queryPrefix}warmup`], {
        pooling: this.pooling,
        normalize: true,
        truncation: true,
        max_length: this.queryMaxTokens,
      });
    }
  }

  async embed(texts: readonly string[], inputType: EmbeddingInputType): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    const extractor = await this.getPipeline();
    const isQuery = inputType === 'query';
    const prefix = isQuery ? this.queryPrefix : this.passagePrefix;
    // Transformer cost grows with sequence length, so this window is what
    // bounds worst-case latency — and a question is not a passage. See
    // EMBEDDING_QUERY_MAX_TOKENS in env.ts for why the query side is capped
    // far lower.
    const maxTokens = isQuery ? this.queryMaxTokens : this.maxTokens;
    const prepared = texts.map((t) => prepareText(t, prefix, maxTokens));
    const out: Float32Array[] = [];

    try {
      for (const batch of chunkArray(prepared, this.batchSize)) {
        const result = await extractor(batch, {
          pooling: this.pooling,
          normalize: true,
          truncation: true,
          max_length: maxTokens,
        });
        out.push(...unpackTensor(result, batch.length));
      }
    } catch (error) {
      throw errors.embedding(
        `Local embedding failed for ${texts.length} text(s): ${(error as Error).message}`,
        error,
      );
    }

    return out;
  }

  async healthCheck(): Promise<{ ok: boolean; detail: string }> {
    try {
      const [vector] = await this.embed(['health check'], 'query');
      if (!vector || vector.length !== this.dimensions) {
        return { ok: false, detail: `expected ${this.dimensions} dims, got ${vector?.length ?? 0}` };
      }
      return {
        ok: true,
        detail: `${this.model} (${this.quantization}, ${this.dimensions}d, ${this.pooling})`,
      };
    } catch (error) {
      return { ok: false, detail: (error as Error).message };
    }
  }
}

/**
 * E5's markers are literally "query: " and "passage: " — the trailing space is
 * part of the trained string. `.env` parsers strip trailing whitespace off an
 * unquoted value, so relying on the operator to quote it would mean the prefix
 * silently degrades to "query:foo" and retrieval quietly gets worse. Restore
 * it here instead.
 */
function normalizePrefix(prefix: string): string {
  const trimmed = prefix.trim();
  return trimmed ? `${trimmed} ` : '';
}

/**
 * Apply the model's side-specific prefix and clamp runaway inputs.
 *
 * The prefix is whatever the configured model was trained with: E5 wants a
 * literal "query: " / "passage: " marker and loses accuracy without it, BGE
 * was trained with neither and loses accuracy with one. Both are just config
 * (see env.ts), so this function stays agnostic.
 */
function prepareText(text: string, prefix: string, maxTokens: number): string {
  const trimmed = text.trim();
  if (!trimmed) return prefix || ' ';
  // The tokenizer truncates properly, but it still has to *tokenize* whatever
  // it is handed first — so clamp the string before it gets there. The bound
  // is generous against the token window (worst case for these multilingual
  // SentencePiece vocabularies is roughly one token per character on
  // Devanagari), so it never cuts text the window would have kept.
  const maxChars = maxTokens * 8;
  const clamped = trimmed.length > maxChars ? trimmed.slice(0, maxChars) : trimmed;
  return prefix ? `${prefix}${clamped}` : clamped;
}

/** Map a model id to a repo that actually ships Transformers.js ONNX weights. */
function resolveOnnxRepo(model: string): string {
  const normalized = model.trim();
  // Neither upstream repo ships the Transformers.js layout; these are the
  // canonical converted builds.
  if (/^BAAI\/bge-m3$/i.test(normalized)) return 'onnx-community/bge-m3-ONNX';
  if (/^intfloat\/multilingual-e5-small$/i.test(normalized)) {
    return 'Xenova/multilingual-e5-small';
  }
  return normalized;
}

/** Split a [batch, dim] (or [batch, seq, dim]) tensor into per-row vectors. */
function unpackTensor(
  result: { dims: number[]; data: Float32Array | number[] },
  expectedRows: number,
): Float32Array[] {
  const data = result.data instanceof Float32Array ? result.data : Float32Array.from(result.data);
  const dim = result.dims[result.dims.length - 1] ?? data.length / Math.max(1, expectedRows);
  const rows = Math.max(1, Math.round(data.length / dim));
  const out: Float32Array[] = [];
  for (let i = 0; i < rows; i += 1) {
    // Copy — the backing buffer is reused between calls.
    out.push(normalizeVector(Float32Array.from(data.subarray(i * dim, (i + 1) * dim))));
  }
  return out;
}

export { resolveOnnxRepo };
export const localModelCacheDir = () => path.resolve(config.embedding.cacheDir);
