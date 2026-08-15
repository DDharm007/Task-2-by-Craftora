/** Async primitives used across the pipeline: timing, retries, timeouts, batching. */

/** Monotonic high-resolution clock in milliseconds. */
export function now(): number {
  return Number(process.hrtime.bigint() / 1_000n) / 1000;
}

/** Run `fn`, returning its value alongside the elapsed milliseconds. */
export async function timed<T>(fn: () => Promise<T> | T): Promise<{ value: T; durationMs: number }> {
  const start = now();
  const value = await fn();
  return { value, durationMs: now() - start };
}

/** Accumulates per-stage durations for one request. */
export class StageTimer {
  private readonly marks = new Map<string, number>();
  private readonly durations = new Map<string, number>();
  readonly startedAt = now();

  start(stage: string): void {
    this.marks.set(stage, now());
  }

  /** Close an open stage and return its duration. */
  end(stage: string): number {
    const started = this.marks.get(stage);
    const duration = started === undefined ? 0 : now() - started;
    this.durations.set(stage, (this.durations.get(stage) ?? 0) + duration);
    this.marks.delete(stage);
    return duration;
  }

  /** Record a duration measured elsewhere (e.g. inside a parallel branch). */
  set(stage: string, durationMs: number): void {
    this.durations.set(stage, durationMs);
  }

  get(stage: string): number {
    return Math.round((this.durations.get(stage) ?? 0) * 1000) / 1000;
  }

  /** Wall-clock time since the timer was created. */
  total(): number {
    return now() - this.startedAt;
  }

  /** Time a function and record it under `stage`. */
  async measure<T>(stage: string, fn: () => Promise<T> | T): Promise<T> {
    this.start(stage);
    try {
      return await fn();
    } finally {
      this.end(stage);
    }
  }

  snapshot(): Record<string, number> {
    const out: Record<string, number> = {};
    for (const key of this.durations.keys()) out[key] = this.get(key);
    return out;
  }
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Reject if `promise` has not settled within `ms`. */
export async function withTimeout<T>(
  promise: Promise<T>,
  ms: number,
  message = `Operation timed out after ${ms}ms`,
): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => {
          const err = new Error(message);
          err.name = 'TimeoutError';
          reject(err);
        }, ms);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

export interface RetryOptions {
  retries: number;
  /** Base delay; grows exponentially with full jitter. */
  baseDelayMs?: number;
  maxDelayMs?: number;
  /** Return false to fail immediately instead of retrying. */
  shouldRetry?: (error: unknown, attempt: number) => boolean;
  onRetry?: (error: unknown, attempt: number, delayMs: number) => void;
}

/**
 * Retry with exponential backoff and full jitter.
 * Jitter matters here — without it, a transient upstream outage causes every
 * in-flight request to retry in lockstep and re-trigger the same rate limit.
 */
export async function retry<T>(fn: (attempt: number) => Promise<T>, options: RetryOptions): Promise<T> {
  const { retries, baseDelayMs = 400, maxDelayMs = 8_000, shouldRetry, onRetry } = options;
  let lastError: unknown;

  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      return await fn(attempt);
    } catch (error) {
      lastError = error;
      if (attempt === retries) break;
      if (shouldRetry && !shouldRetry(error, attempt)) break;
      const ceiling = Math.min(maxDelayMs, baseDelayMs * 2 ** attempt);
      const delay = Math.round(Math.random() * ceiling);
      onRetry?.(error, attempt + 1, delay);
      await sleep(delay);
    }
  }
  throw lastError;
}

/** Split an array into fixed-size batches. */
export function chunkArray<T>(items: readonly T[], size: number): T[][] {
  if (size <= 0) throw new Error('Batch size must be positive');
  const out: T[][] = [];
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));
  return out;
}

/**
 * Map over items with bounded concurrency, preserving input order in the result.
 */
export async function mapWithConcurrency<T, R>(
  items: readonly T[],
  concurrency: number,
  mapper: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  if (items.length === 0) return [];
  const limit = Math.max(1, Math.min(concurrency, items.length));
  const results = new Array<R>(items.length);
  let cursor = 0;

  const workers = Array.from({ length: limit }, async () => {
    for (;;) {
      const index = cursor;
      cursor += 1;
      if (index >= items.length) return;
      results[index] = await mapper(items[index] as T, index);
    }
  });

  await Promise.all(workers);
  return results;
}

/**
 * Ensure an expensive async initialiser runs exactly once, even under
 * concurrent callers. Used for lazily loading ONNX models.
 */
export function once<T>(factory: () => Promise<T>): () => Promise<T> {
  let pending: Promise<T> | null = null;
  return () => {
    if (!pending) {
      pending = factory().catch((error) => {
        pending = null; // allow a later retry after a failed load
        throw error;
      });
    }
    return pending;
  };
}
