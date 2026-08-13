/** Contract every embedding provider implements. */

export type EmbeddingInputType = 'query' | 'passage';

export interface EmbeddingProvider {
  readonly name: string;
  readonly model: string;
  readonly dimensions: number;

  /**
   * Embed a batch of texts.
   *
   * Implementations must return L2-normalised vectors so downstream cosine
   * similarity reduces to a dot product, and must preserve input order.
   */
  embed(texts: readonly string[], inputType: EmbeddingInputType): Promise<Float32Array[]>;

  /** Warm the provider (download/compile the model) so first query isn't slow. */
  warmup(): Promise<void>;

  /** Cheap liveness probe used by /api/health. */
  healthCheck(): Promise<{ ok: boolean; detail: string }>;
}

/** L2-normalise in place and return the same array. */
export function normalizeVector(vector: Float32Array): Float32Array {
  let sumSquares = 0;
  for (let i = 0; i < vector.length; i += 1) sumSquares += (vector[i] as number) ** 2;
  const magnitude = Math.sqrt(sumSquares);
  if (magnitude > 0) {
    for (let i = 0; i < vector.length; i += 1) vector[i] = (vector[i] as number) / magnitude;
  }
  return vector;
}

/** Cosine similarity. Assumes both inputs are already normalised. */
export function dot(a: Float32Array | number[], b: Float32Array | number[]): number {
  const length = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < length; i += 1) sum += (a[i] as number) * (b[i] as number);
  return sum;
}
