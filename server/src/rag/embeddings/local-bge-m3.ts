/**
 * BAAI/bge-m3 running in-process via ONNX Runtime (Transformers.js).
 *
 * This is the spec's embedding model. It runs locally rather than through an
 * API, which runs the model in-process with no external dependency —
 * see docs/ARCHITECTURE.md. Running locally also removes a network hop from
 * the hot path and makes indexing cost-free.
 *
 * BGE-M3 is a multilingual (XLM-RoBERTa) model producing 1024-dim dense
 * vectors, which is exactly what a Hindi-query-over-English-passages corpus
 * needs. We use CLS pooling — the pooling method BGE-M3 was trained with;
 * mean pooling measurably degrades its retrieval quality.
 *
 * Weights are downloaded once into ./models on first use (~542MB at int8).
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

export class LocalBgeM3Provider implements EmbeddingProvider {
  readonly name = 'local:onnx';
  readonly model: string;
  readonly dimensions: number;

  private readonly quantization: string;
  private readonly batchSize: number;
  private readonly maxTokens: number;
  private readonly getPipeline: () => Promise<FeatureExtractionPipeline>;

  constructor() {
    this.model = config.embedding.model;
    this.dimensions = config.embedding.dimensions;
    this.quantization = config.embedding.quantization;
    this.batchSize = config.embedding.batchSize;
    this.maxTokens = config.embedding.maxTokens;

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

  async warmup(): Promise<void> {
    const extractor = await this.getPipeline();
    await extractor(['warmup'], { pooling: 'cls', normalize: true, truncation: true, max_length: 32 });
  }

  async embed(texts: readonly string[], inputType: EmbeddingInputType): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    const extractor = await this.getPipeline();
    const prepared = texts.map((t) => prepareText(t, inputType));
    const out: Float32Array[] = [];

    try {
      for (const batch of chunkArray(prepared, this.batchSize)) {
        const result = await extractor(batch, {
          pooling: 'cls',
          normalize: true,
          truncation: true,
          max_length: this.maxTokens,
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
      return { ok: true, detail: `${this.model} (${this.quantization}, ${this.dimensions}d)` };
    } catch (error) {
      return { ok: false, detail: (error as Error).message };
    }
  }
}

/**
 * BGE-M3 asymmetric prefixes.
 *
 * BGE-M3, unlike bge-large-en, was trained *without* an instruction prefix on
 * either side, so adding one hurts. We deliberately pass text through
 * unchanged and only trim runaway inputs.
 */
function prepareText(text: string, _inputType: EmbeddingInputType): string {
  const trimmed = text.trim();
  if (!trimmed) return ' ';
  // Guard against pathological inputs; the tokenizer truncates properly but
  // handing it megabytes of text wastes time before it gets there.
  const maxChars = 8_000;
  return trimmed.length > maxChars ? trimmed.slice(0, maxChars) : trimmed;
}

/** Map a model id to a repo that actually ships ONNX weights. */
function resolveOnnxRepo(model: string): string {
  const normalized = model.trim();
  // BAAI's own repo has ONNX but not the Transformers.js layout; the
  // onnx-community mirror is the canonical converted build.
  if (/^BAAI\/bge-m3$/i.test(normalized)) return 'onnx-community/bge-m3-ONNX';
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
