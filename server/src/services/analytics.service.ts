/**
 * In-process metrics store.
 *
 * Keeps a bounded ring buffer of recent requests and computes latency
 * percentiles, token totals and guardrail counts on demand. Deliberately
 * in-memory: this is observability for a single service instance, not a
 * metrics backend. In a multi-replica deployment these numbers would be
 * emitted to Prometheus instead — the structured logs carry the same fields.
 */
import type {
  AnalyticsSnapshot,
  AnswerStatus,
  GuardrailReport,
  LatencyBreakdown,
  LatencyPercentiles,
  RequestLogEntry,
  StageLatencyStats,
  TokenUsage,
} from '@voxrag/shared';

/** How many requests to retain. Enough for meaningful percentiles, bounded memory. */
const MAX_ENTRIES = 1_000;

interface RecordedRequest {
  entry: RequestLogEntry;
  latency: LatencyBreakdown;
  usage: TokenUsage;
}

const requests: RecordedRequest[] = [];
const guardrailEvents = new Map<string, number>();
const startedAt = Date.now();

let failedRequests = 0;

export interface RecordInput {
  requestId: string;
  query: string;
  status: AnswerStatus;
  confidence: number;
  latency: LatencyBreakdown;
  usage: TokenUsage;
  chunkCount: number;
  voice: boolean;
  guardrails: GuardrailReport;
}

export function recordRequest(input: RecordInput): void {
  const entry: RequestLogEntry = {
    requestId: input.requestId,
    query: input.query.slice(0, 300),
    status: input.status,
    confidence: input.confidence,
    totalLatencyMs: Math.round(input.latency.total),
    chunkCount: input.chunkCount,
    tokensUsed: input.usage.totalTokens,
    voice: input.voice,
    createdAt: new Date().toISOString(),
  };

  requests.push({ entry, latency: input.latency, usage: input.usage });
  if (requests.length > MAX_ENTRIES) requests.shift();

  // Count anything that tripped, so the dashboard shows guardrail activity
  // rather than only outright blocks.
  for (const result of input.guardrails.results) {
    if (result.verdict === 'pass') continue;
    guardrailEvents.set(result.id, (guardrailEvents.get(result.id) ?? 0) + 1);
  }
}

export function recordFailure(): void {
  failedRequests += 1;
}

/**
 * Percentiles using the nearest-rank method.
 *
 * Nearest-rank rather than linear interpolation because every reported value
 * is then an latency we actually observed, which is what you want when
 * debugging a tail — an interpolated p99 corresponds to no real request.
 */
export function percentiles(values: readonly number[]): LatencyPercentiles {
  if (values.length === 0) {
    return { p50: 0, p70: 0, p95: 0, p99: 0, p100: 0, mean: 0, min: 0, count: 0 };
  }

  const sorted = [...values].sort((a, b) => a - b);
  const at = (p: number): number => {
    const rank = Math.ceil((p / 100) * sorted.length);
    const index = Math.min(sorted.length - 1, Math.max(0, rank - 1));
    return round(sorted[index] as number);
  };

  return {
    p50: at(50),
    p70: at(70),
    p95: at(95),
    p99: at(99),
    p100: round(sorted[sorted.length - 1] as number),
    mean: round(sorted.reduce((sum, value) => sum + value, 0) / sorted.length),
    min: round(sorted[0] as number),
    count: sorted.length,
  };
}

function round(value: number): number {
  return Math.round(value * 100) / 100;
}

function stageStats(): StageLatencyStats {
  const pick = (selector: (latency: LatencyBreakdown) => number | null): number[] =>
    requests
      .map((request) => selector(request.latency))
      .filter((value): value is number => typeof value === 'number' && Number.isFinite(value));

  return {
    embedding: percentiles(pick((l) => l.embedding)),
    // Retrieval covers both arms plus fusion — they run concurrently, so the
    // stage cost is the slower arm, not the sum.
    retrieval: percentiles(
      requests.map((request) =>
        Math.max(request.latency.denseRetrieval, request.latency.sparseRetrieval) + request.latency.fusion,
      ),
    ),
    reranking: percentiles(pick((l) => l.reranking)),
    generation: percentiles(pick((l) => l.generation)),
    transcription: percentiles(pick((l) => l.transcription)),
    total: percentiles(pick((l) => l.total)),
  };
}

/** Requests bucketed by hour, oldest first. */
function throughput(): AnalyticsSnapshot['throughput'] {
  const buckets = new Map<string, { count: number; totalLatency: number }>();

  for (const request of requests) {
    const bucket = `${request.entry.createdAt.slice(0, 13)}:00`;
    const current = buckets.get(bucket) ?? { count: 0, totalLatency: 0 };
    current.count += 1;
    current.totalLatency += request.entry.totalLatencyMs;
    buckets.set(bucket, current);
  }

  return [...buckets.entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([bucket, value]) => ({
      bucket,
      count: value.count,
      averageLatencyMs: round(value.totalLatency / value.count),
    }));
}

export function getAnalytics(recentLimit = 25): AnalyticsSnapshot {
  const successful = requests.filter((request) => request.entry.status === 'answered').length;
  const blocked = requests.filter((request) => request.entry.status === 'blocked').length;
  const lowConfidence = requests.filter(
    (request) => request.entry.status === 'low_confidence' || request.entry.status === 'insufficient_context',
  ).length;

  const tokensUsed = requests.reduce<TokenUsage>(
    (sum, request) => ({
      promptTokens: sum.promptTokens + request.usage.promptTokens,
      completionTokens: sum.completionTokens + request.usage.completionTokens,
      reasoningTokens: sum.reasoningTokens + request.usage.reasoningTokens,
      totalTokens: sum.totalTokens + request.usage.totalTokens,
    }),
    { promptTokens: 0, completionTokens: 0, reasoningTokens: 0, totalTokens: 0 },
  );

  const totalLatency = requests.reduce((sum, request) => sum + request.entry.totalLatencyMs, 0);
  const totalConfidence = requests.reduce((sum, request) => sum + request.entry.confidence, 0);

  return {
    totalRequests: requests.length,
    successfulRequests: successful,
    failedRequests,
    blockedRequests: blocked,
    lowConfidenceRequests: lowConfidence,
    averageLatencyMs: requests.length ? round(totalLatency / requests.length) : 0,
    averageConfidence: requests.length ? round(totalConfidence / requests.length) : 0,
    tokensUsed,
    latency: stageStats(),
    throughput: throughput(),
    recent: requests
      .slice(-recentLimit)
      .map((request) => request.entry)
      .reverse(),
    guardrailEvents: Object.fromEntries(guardrailEvents),
    uptimeSeconds: Math.round((Date.now() - startedAt) / 1000),
  };
}

/** Clear all metrics — used by the benchmark so its numbers stand alone. */
export function resetAnalytics(): void {
  requests.length = 0;
  guardrailEvents.clear();
  failedRequests = 0;
}

export function uptimeSeconds(): number {
  return Math.round((Date.now() - startedAt) / 1000);
}
