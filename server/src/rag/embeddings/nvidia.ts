/**
 * NVIDIA NIM hosted embeddings.
 *
 * The fast path: no model download, and indexing runs an order of magnitude
 * quicker than CPU ONNX. Enabled with `EMBEDDING_PROVIDER=nvidia`.
 *
 * Note on model choice: the spec names BAAI/bge-m3, and NVIDIA does list
 * `baai/bge-m3` in /v1/models — but that NIM returns HTTP 500 for every
 * request shape we tried. This provider therefore defaults to
 * `nvidia/nv-embedqa-e5-v5`, which is also 1024-dim so it is a drop-in swap.
 * Set NVIDIA_EMBEDDING_MODEL to override once the bge-m3 NIM is fixed.
 *
 * NIM embeddings are asymmetric: `input_type` must be "query" at search time
 * and "passage" at index time, or recall degrades badly.
 */
import { config } from '../../config/env.js';
import { logger } from '../../utils/logger.js';
import { chunkArray, retry, withTimeout } from '../../utils/async.js';
import { errors } from '../../utils/errors.js';
import { normalizeVector, type EmbeddingInputType, type EmbeddingProvider } from './types.js';

interface NimEmbeddingResponse {
  data?: Array<{ index: number; embedding: number[] }>;
  usage?: { prompt_tokens: number; total_tokens: number };
  detail?: unknown;
  error?: unknown;
}

export class NvidiaEmbeddingProvider implements EmbeddingProvider {
  readonly name = 'nvidia:nim';
  readonly model: string;
  readonly dimensions: number;

  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly batchSize: number;

  constructor() {
    this.model = config.embedding.nvidiaModel;
    this.dimensions = config.embedding.dimensions;
    // Deliberately its own credentials, not config.llm's — chat generation
    // (GROQ_*) and this NIM embeddings path are unrelated providers that
    // happen to both be NVIDIA-API-shaped; borrowing the LLM's key/base URL
    // here meant this silently broke the moment GROQ_* stopped pointing at
    // NVIDIA NIM.
    this.baseUrl = config.embedding.nvidiaBaseUrl.replace(/\/+$/, '');
    this.apiKey = config.embedding.nvidiaApiKey;
    if (!this.apiKey) {
      throw errors.embedding(
        'EMBEDDING_PROVIDER=nvidia requires NVIDIA_EMBEDDING_API_KEY to be set.',
      );
    }
    // NIM caps batch size; 32 stays comfortably under it while amortising RTT.
    this.batchSize = Math.min(32, Math.max(1, config.embedding.batchSize * 4));
  }

  async warmup(): Promise<void> {
    await this.embed(['warmup'], 'query');
  }

  async embed(texts: readonly string[], inputType: EmbeddingInputType): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    const out: Float32Array[] = [];

    for (const batch of chunkArray([...texts], this.batchSize)) {
      const vectors = await retry(
        () => this.embedBatch(batch, inputType),
        {
          retries: 2,
          baseDelayMs: 500,
          shouldRetry: (error) => {
            const status = (error as { status?: number }).status;
            // Retry transport errors and 429/5xx; a 400 will never succeed.
            return status === undefined || status === 429 || status >= 500;
          },
          onRetry: (error, attempt, delayMs) =>
            logger.warn(
              { attempt, delayMs, error: (error as Error).message },
              'Retrying NVIDIA embedding request',
            ),
        },
      );
      out.push(...vectors);
    }
    return out;
  }

  private async embedBatch(batch: string[], inputType: EmbeddingInputType): Promise<Float32Array[]> {
    const response = await withTimeout(
      fetch(`${this.baseUrl}/embeddings`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify({
          input: batch.map((t) => (t.trim() ? t.slice(0, 8_000) : ' ')),
          model: this.model,
          input_type: inputType,
          encoding_format: 'float',
          truncate: 'END',
        }),
      }),
      60_000,
      'NVIDIA embeddings request timed out',
    );

    const body = (await response.json().catch(() => ({}))) as NimEmbeddingResponse;

    if (!response.ok) {
      const detail = JSON.stringify(body.detail ?? body.error ?? body).slice(0, 400);
      const error = errors.embedding(
        `NVIDIA embeddings returned HTTP ${response.status} for model ${this.model}: ${detail}`,
      );
      (error as unknown as { status: number }).status = response.status;
      throw error;
    }

    if (!Array.isArray(body.data) || body.data.length === 0) {
      throw errors.embedding('NVIDIA embeddings returned no data');
    }

    // The API may reorder results, so sort by the echoed index.
    const sorted = [...body.data].sort((a, b) => a.index - b.index);
    if (sorted.length !== batch.length) {
      throw errors.embedding(
        `NVIDIA embeddings returned ${sorted.length} vectors for ${batch.length} inputs`,
      );
    }

    return sorted.map((item) => normalizeVector(Float32Array.from(item.embedding)));
  }

  async healthCheck(): Promise<{ ok: boolean; detail: string }> {
    try {
      const [vector] = await this.embed(['health check'], 'query');
      if (!vector) return { ok: false, detail: 'no vector returned' };
      if (vector.length !== this.dimensions) {
        return {
          ok: false,
          detail: `dimension mismatch: EMBEDDING_DIMENSIONS=${this.dimensions} but ${this.model} returned ${vector.length}`,
        };
      }
      return { ok: true, detail: `${this.model} (${vector.length}d)` };
    } catch (error) {
      return { ok: false, detail: (error as Error).message };
    }
  }
}
