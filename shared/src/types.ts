/**
 * Core domain types shared between the GoaRAG server and client.
 *
 * These describe the wire format of the HTTP/SSE API. Keep them free of any
 * runtime dependency so the browser bundle stays small.
 */

// ─── Chunking ────────────────────────────────────────────────────────────────

/** Every chunking strategy implemented by the indexing pipeline. */
export type ChunkStrategy =
  | 'recursive'
  | 'semantic'
  | 'sliding_window'
  | 'parent'
  | 'overlap'
  | 'metadata';

/**
 * Metadata attached to every indexed chunk. This is what gets stored in the
 * vector store payload and returned to the UI for the chunk inspector.
 */
export interface ChunkMetadata {
  /** Stable id of the source document this chunk came from. */
  documentId: string;
  /** Human-readable origin, e.g. `MSMARCO-XI/validation#1102432`. */
  source: string;
  /** BCP-47-ish language tag from the dataset, e.g. `hin_Deva` or `eng_Latn`. */
  language: string;
  /** Id of the passage within the source document. */
  passageId: string;
  /** Zero-based index of this chunk within its passage. */
  chunkIndex: number;
  /** Id of the coarse parent chunk, if this is a child chunk. */
  parentChunk: string | null;
  /** Short topic label derived from the document's query/title. */
  topic: string;
  /** Which strategy produced this chunk. */
  strategy: ChunkStrategy;
  /** Approximate token count of the chunk text. */
  tokenCount: number;
  /** Character offsets into the original passage text. */
  charStart: number;
  charEnd: number;
  /** True when the dataset marked this passage as relevant to its query. */
  isSelected: boolean;
  /** The dataset query this passage was retrieved for (ground truth link). */
  queryId: string | null;
  /** Original untranslated query text, when available. */
  queryText: string | null;
  /** ISO timestamp of indexing. */
  indexedAt: string;
}

/** A chunk produced by the chunking pipeline, ready to embed. */
export interface Chunk {
  id: string;
  text: string;
  metadata: ChunkMetadata;
}

/** A normalized document produced by the dataset loader. */
export interface SourceDocument {
  documentId: string;
  passageId: string;
  text: string;
  language: string;
  source: string;
  topic: string;
  isSelected: boolean;
  queryId: string | null;
  queryText: string | null;
}

// ─── Retrieval ───────────────────────────────────────────────────────────────

/** A chunk returned by the retriever, with the scores that got it there. */
export interface RetrievedChunk {
  id: string;
  text: string;
  metadata: ChunkMetadata;
  /** Cosine similarity from dense vector search (0-1), null if sparse-only hit. */
  denseScore: number | null;
  /** BM25 relevance from keyword search, null if dense-only hit. */
  sparseScore: number | null;
  /** Reciprocal Rank Fusion score combining dense + sparse rankings. */
  fusedScore: number;
  /** Cross-encoder relevance after reranking (0-1), null before rerank. */
  rerankScore: number | null;
  /** Final score used for ordering and confidence. */
  score: number;
  /** 1-based rank in the fused candidate list, before reranking. */
  rankBeforeRerank: number;
  /** 1-based rank after reranking, null if this chunk was dropped. */
  rankAfterRerank: number | null;
  /** Parent chunk text, when parent expansion supplied extra context. */
  parentText: string | null;
  /** Which retrieval arms found this chunk. */
  matchedBy: Array<'dense' | 'sparse'>;
}

/** A citation surfaced next to the generated answer. */
export interface Citation {
  /** 1-based marker used in the answer text, e.g. `[2]`. */
  index: number;
  chunkId: string;
  documentId: string;
  source: string;
  topic: string;
  language: string;
  score: number;
  /** Short preview of the cited text. */
  snippet: string;
}

// ─── Guardrails ──────────────────────────────────────────────────────────────

export type GuardrailId =
  | 'prompt_injection'
  | 'jailbreak'
  | 'toxicity'
  | 'off_topic'
  | 'similarity_threshold'
  | 'context_verification'
  | 'hallucination'
  | 'confidence';

export type GuardrailStage = 'pre_retrieval' | 'post_retrieval' | 'post_generation';

export type GuardrailVerdict = 'pass' | 'warn' | 'block';

/** Result of a single guardrail check. */
export interface GuardrailResult {
  id: GuardrailId;
  stage: GuardrailStage;
  verdict: GuardrailVerdict;
  /** 0-1, higher means more evidence the guardrail was tripped. */
  score: number;
  /** Threshold this check compared against. */
  threshold: number;
  /** Human-readable explanation shown in the UI. */
  reason: string;
  /** Matched patterns / offending spans, for the inspector. */
  evidence: string[];
  durationMs: number;
}

/** Aggregate guardrail outcome for one request. */
export interface GuardrailReport {
  passed: boolean;
  blocked: boolean;
  blockedBy: GuardrailId | null;
  results: GuardrailResult[];
  totalDurationMs: number;
}

// ─── Confidence ──────────────────────────────────────────────────────────────

/** Breakdown of how the final confidence score was computed. */
export interface ConfidenceBreakdown {
  /** Weighted overall confidence, 0-1. */
  overall: number;
  /** Top reranked chunk score. */
  topScore: number;
  /** Mean score across the chunks actually sent to the LLM. */
  meanScore: number;
  /** Agreement between dense and sparse rankings, 0-1. */
  retrievalAgreement: number;
  /** Fraction of answer sentences grounded in the retrieved context, 0-1. */
  groundedness: number;
  /** How much of the query's content words appear in the context, 0-1. */
  contextCoverage: number;
  /** Whether the score cleared CONFIDENCE_THRESHOLD. */
  sufficient: boolean;
  threshold: number;
}

// ─── Latency ─────────────────────────────────────────────────────────────────

/** Per-stage timings, in milliseconds. */
export interface LatencyBreakdown {
  transcription: number | null;
  guardrailsPre: number;
  embedding: number;
  denseRetrieval: number;
  sparseRetrieval: number;
  fusion: number;
  reranking: number;
  promptBuilding: number;
  generation: number;
  guardrailsPost: number;
  /** Wall-clock time for the whole request. */
  total: number;
  /** Time from request start to the first streamed answer token. */
  timeToFirstToken: number | null;
}

export interface TokenUsage {
  promptTokens: number;
  completionTokens: number;
  reasoningTokens: number;
  totalTokens: number;
}

// ─── Transcription ───────────────────────────────────────────────────────────

export interface TranscriptWord {
  text: string;
  start: number;
  end: number;
  type: string;
  /** Log-probability from the STT model; used for transcript confidence. */
  logprob: number | null;
}

export interface TranscriptionResult {
  text: string;
  languageCode: string | null;
  languageProbability: number | null;
  durationSeconds: number | null;
  words: TranscriptWord[];
  /** 0-1 confidence derived from per-word log-probabilities. */
  confidence: number;
  provider: 'elevenlabs';
  model: string;
  latencyMs: number;
}

// ─── Query pipeline ──────────────────────────────────────────────────────────

export interface ConversationTurn {
  role: 'user' | 'assistant';
  content: string;
}

export type AnswerStatus =
  | 'answered'
  | 'insufficient_context'
  | 'blocked'
  | 'low_confidence'
  /**
   * Answered, but *not* from the indexed corpus — retrieval found nothing
   * usable and the answer came from web search and/or the model's own
   * knowledge. Distinct from `answered` because in a RAG system the
   * provenance of an answer is part of its meaning: presenting an ungrounded
   * answer as a retrieved one is the failure mode RAG exists to prevent.
   */
  | 'unverified';

/**
 * Where an answer's content actually came from.
 *
 * `corpus` is the only value that means "retrieval-augmented"; the others
 * record that the pipeline degraded to something else and say so out loud.
 */
export type AnswerSource = 'corpus' | 'web' | 'model_knowledge' | 'none';

/** The full result of one RAG query. */
export interface QueryResult {
  requestId: string;
  query: string;
  /** Present only for voice queries. */
  transcription: TranscriptionResult | null;
  answer: string;
  status: AnswerStatus;
  /**
   * Provenance of `answer`. Always set — a consumer should be able to tell a
   * grounded answer from an ungrounded one without inferring it from an empty
   * `citations` array.
   */
  answerSource: AnswerSource;
  /** Model's chain-of-thought, when thinking mode is enabled. */
  reasoning: string | null;
  citations: Citation[];
  chunks: RetrievedChunk[];
  confidence: ConfidenceBreakdown;
  guardrails: GuardrailReport;
  latency: LatencyBreakdown;
  usage: TokenUsage;
  model: string;
  /** Which components actually served this request. */
  providers: {
    embedding: string;
    vectorStore: string;
    reranker: string;
    llm: string;
    stt: string | null;
  };
  createdAt: string;
}

// ─── Streaming (SSE) ─────────────────────────────────────────────────────────

/** Server-sent event payloads emitted by `/api/query` and `/api/voice-query`. */
export type StreamEvent =
  | { type: 'start'; requestId: string; createdAt: string }
  | { type: 'stage'; stage: PipelineStage; status: 'started' | 'completed'; durationMs?: number; detail?: string }
  | { type: 'transcript'; transcription: TranscriptionResult }
  | { type: 'guardrails'; report: GuardrailReport }
  | { type: 'chunks'; chunks: RetrievedChunk[] }
  | { type: 'reasoning'; delta: string }
  | { type: 'token'; delta: string }
  | { type: 'done'; result: QueryResult }
  | { type: 'error'; error: ApiErrorBody };

export type PipelineStage =
  | 'transcription'
  | 'guardrails'
  | 'embedding'
  | 'retrieval'
  | 'reranking'
  | 'prompt'
  | 'generation'
  | 'verification';

// ─── Errors ──────────────────────────────────────────────────────────────────

export type ErrorCode =
  | 'VALIDATION_ERROR'
  | 'RATE_LIMITED'
  | 'PAYLOAD_TOO_LARGE'
  | 'UNSUPPORTED_MEDIA_TYPE'
  | 'STT_FAILED'
  | 'TTS_FAILED'
  | 'EMBEDDING_FAILED'
  | 'VECTOR_STORE_UNAVAILABLE'
  | 'INDEX_EMPTY'
  | 'LLM_FAILED'
  | 'LLM_TIMEOUT'
  | 'GUARDRAIL_BLOCKED'
  | 'NOT_FOUND'
  | 'INTERNAL_ERROR';

export interface ApiErrorBody {
  code: ErrorCode;
  message: string;
  /** Safe, structured detail. Never contains secrets or stack traces in prod. */
  details?: unknown;
  requestId: string;
}

// ─── Observability ───────────────────────────────────────────────────────────

export interface LatencyPercentiles {
  p50: number;
  p70: number;
  p95: number;
  p99: number;
  p100: number;
  mean: number;
  min: number;
  count: number;
}

export interface StageLatencyStats {
  embedding: LatencyPercentiles;
  retrieval: LatencyPercentiles;
  reranking: LatencyPercentiles;
  generation: LatencyPercentiles;
  transcription: LatencyPercentiles;
  total: LatencyPercentiles;
}

export interface AnalyticsSnapshot {
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  blockedRequests: number;
  lowConfidenceRequests: number;
  averageLatencyMs: number;
  averageConfidence: number;
  tokensUsed: TokenUsage;
  latency: StageLatencyStats;
  /** Rolling window of recent requests, newest first. */
  recent: RequestLogEntry[];
  /** Count of guardrail trips by id. */
  guardrailEvents: Record<string, number>;
  /** Requests per hour bucket, oldest first. */
  throughput: Array<{ bucket: string; count: number; averageLatencyMs: number }>;
  uptimeSeconds: number;
}

export interface RequestLogEntry {
  requestId: string;
  query: string;
  status: AnswerStatus;
  confidence: number;
  totalLatencyMs: number;
  chunkCount: number;
  tokensUsed: number;
  voice: boolean;
  createdAt: string;
}

export interface IndexStats {
  documents: number;
  vectors: number;
  chunks: number;
  averageChunkSizeChars: number;
  averageChunkTokens: number;
  languages: Array<{ language: string; count: number }>;
  strategies: Array<{ strategy: ChunkStrategy; count: number }>;
  collection: string;
  vectorStore: string;
  embeddingModel: string;
  embeddingDimensions: number;
  lastIndexedAt: string | null;
  indexed: boolean;
}

export interface StatsResponse {
  index: IndexStats;
  analytics: AnalyticsSnapshot;
}

export type HealthStatus = 'ok' | 'degraded' | 'down';

export interface ComponentHealth {
  name: string;
  status: HealthStatus;
  detail: string;
  latencyMs: number | null;
}

export interface HealthResponse {
  status: HealthStatus;
  version: string;
  uptimeSeconds: number;
  timestamp: string;
  components: ComponentHealth[];
  /**
   * The models actually configured, as plain fields rather than parsed out of
   * a component's human-readable `detail` string — the sidebar footer used to
   * hardcode these and silently went stale the moment the backend switched
   * providers.
   */
  models: {
    llm: string;
    embedding: string;
  };
}

// ─── Benchmark ───────────────────────────────────────────────────────────────

/** Retrieval quality metrics computed against the dataset's `is_selected` labels. */
export interface RetrievalQualityMetrics {
  recallAt5: number;
  recallAt10: number;
  precisionAt5: number;
  mrr: number;
  ndcgAt5: number;
  hitRate: number;
  evaluatedQueries: number;
}

export interface BenchmarkCase {
  query: string;
  language: string;
  expectedDocumentIds: string[];
  retrievedDocumentIds: string[];
  hit: boolean;
  reciprocalRank: number;
  latencyMs: number;
  confidence: number;
  status: AnswerStatus;
}

export interface BenchmarkResult {
  startedAt: string;
  finishedAt: string;
  durationMs: number;
  sampleSize: number;
  /** Whether the LLM was invoked (slow) or only retrieval was measured. */
  generationEnabled: boolean;
  latency: StageLatencyStats;
  quality: RetrievalQualityMetrics;
  averageConfidence: number;
  tokensUsed: TokenUsage;
  cases: BenchmarkCase[];
  config: {
    embeddingProvider: string;
    embeddingModel: string;
    vectorStore: string;
    rerankerProvider: string;
    llmModel: string;
    topK: number;
    rerankTopN: number;
  };
}
