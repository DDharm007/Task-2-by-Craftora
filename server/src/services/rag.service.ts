/**
 * The RAG orchestrator.
 *
 * Owns the end-to-end query pipeline and is the single place that decides
 * whether a question gets answered, refused, or blocked:
 *
 *   guardrails(in) → retrieve → guardrails(evidence) → prompt → generate
 *   → guardrails(answer) → confidence gate → result
 *
 * Both entry points (`runQuery` and `streamQuery`) share this logic; the
 * streaming variant just emits events as it goes. Every stage is timed, and
 * the timings are what the analytics endpoint reports.
 */
import { randomUUID } from 'node:crypto';
import type {
  AnswerSource,
  AnswerStatus,
  ConversationTurn,
  GuardrailReport,
  LatencyBreakdown,
  QueryResult,
  RetrievalOptions,
  RetrievedChunk,
  StreamEvent,
  TokenUsage,
  TranscriptionResult,
} from '@goarag/shared';
import { INSUFFICIENT_EVIDENCE_MESSAGE, LOW_CONFIDENCE_MESSAGE } from '@goarag/shared';
import { config } from '../config/env.js';
import { requestLogger } from '../utils/logger.js';
import { StageTimer, now } from '../utils/async.js';
import { errors } from '../utils/errors.js';
import { getEmbeddingProvider } from '../rag/embeddings/index.js';
import { getVectorStore } from '../rag/vector/index.js';
import { getReranker } from '../rag/reranker/index.js';
import { retrieve } from '../rag/retriever/index.js';
import { buildPrompt } from '../rag/prompt/index.js';
import {
  groundednessAgainstTexts,
  mergeReports,
  runGenerationGuardrails,
  runInputGuardrails,
  runRetrievalGuardrails,
} from '../rag/guardrails/index.js';
import { generateCompletion, streamCompletion } from './llm.service.js';
import { webSearch, formatWebContext } from './websearch.service.js';
import { getIndexStats } from './indexing.service.js';
import { recordRequest } from './analytics.service.js';

export interface QueryInput {
  query: string;
  history?: ConversationTurn[];
  options?: RetrievalOptions;
  /** Present when the query originated from voice. */
  transcription?: TranscriptionResult | null;
  signal?: AbortSignal;
  requestId?: string;
}

const EMPTY_USAGE: TokenUsage = {
  promptTokens: 0,
  completionTokens: 0,
  reasoningTokens: 0,
  totalTokens: 0,
};

/** Assemble the latency breakdown from the stage timer. */
function buildLatency(
  timer: StageTimer,
  transcriptionMs: number | null,
  timeToFirstToken: number | null,
): LatencyBreakdown {
  return {
    transcription: transcriptionMs,
    guardrailsPre: timer.get('guardrailsPre'),
    embedding: timer.get('embedding'),
    denseRetrieval: timer.get('denseRetrieval'),
    sparseRetrieval: timer.get('sparseRetrieval'),
    fusion: timer.get('fusion'),
    diversity: timer.get('diversity'),
    reranking: timer.get('reranking'),
    expansion: timer.get('expansion'),
    promptBuilding: timer.get('promptBuilding'),
    generation: timer.get('generation'),
    guardrailsPost: timer.get('guardrailsPost'),
    total: Math.round(timer.total() * 100) / 100,
    timeToFirstToken,
  };
}

async function describeProviders(voice: boolean) {
  const store = await getVectorStore();
  return {
    embedding: `${getEmbeddingProvider().name}:${getEmbeddingProvider().model}`,
    vectorStore: store.name,
    reranker: getReranker().name,
    llm: config.llm.model,
    stt: voice ? 'elevenlabs' : null,
  };
}

/** Fail fast with a clear message when nothing has been indexed yet. */
async function assertIndexed(): Promise<void> {
  const stats = await getIndexStats();
  if (!stats.indexed) throw errors.indexEmpty();
}

/**
 * Build the terminal result for a request that never reached generation
 * (blocked by an input guardrail, or with no usable evidence).
 */
function shortCircuitResult(input: {
  requestId: string;
  query: string;
  transcription: TranscriptionResult | null;
  answer: string;
  status: AnswerStatus;
  /** Defaults to `none`: a short-circuit produced no answer content. */
  answerSource?: AnswerSource;
  guardrails: GuardrailReport;
  chunks: RetrievedChunk[];
  latency: LatencyBreakdown;
  providers: Awaited<ReturnType<typeof describeProviders>>;
}): QueryResult {
  return {
    requestId: input.requestId,
    query: input.query,
    transcription: input.transcription,
    answer: input.answer,
    status: input.status,
    answerSource: input.answerSource ?? 'none',
    reasoning: null,
    citations: [],
    chunks: input.chunks,
    confidence: {
      overall: 0,
      topScore: 0,
      meanScore: 0,
      retrievalAgreement: 0,
      groundedness: 0,
      contextCoverage: 0,
      sufficient: false,
      threshold: config.guardrails.confidenceThreshold,
    },
    guardrails: input.guardrails,
    latency: input.latency,
    usage: { ...EMPTY_USAGE },
    model: config.llm.model,
    providers: input.providers,
    createdAt: new Date().toISOString(),
  };
}

/**
 * Run a query to completion (non-streaming).
 *
 * `streamQuery` below is the primary path; this exists for the JSON API and
 * the benchmark, which want the whole result at once.
 */
export async function runQuery(input: QueryInput): Promise<QueryResult> {
  const requestId = input.requestId ?? randomUUID();
  const log = requestLogger(requestId);
  const timer = new StageTimer();
  const options = input.options ?? {};
  const voice = Boolean(input.transcription);
  const transcriptionMs = input.transcription?.latencyMs ?? null;

  // ── 1. input guardrails ───────────────────────────────────────────────────
  timer.start('guardrailsPre');
  const inputReport = runInputGuardrails(input.query);
  timer.end('guardrailsPre');

  const providers = await describeProviders(voice);

  if (inputReport.blocked) {
    log.warn({ blockedBy: inputReport.blockedBy }, 'Request blocked by input guardrail');
    const result = shortCircuitResult({
      requestId,
      query: input.query,
      transcription: input.transcription ?? null,
      answer: refusalFor(inputReport),
      status: 'blocked',
      guardrails: inputReport,
      chunks: [],
      latency: buildLatency(timer, transcriptionMs, null),
      providers,
    });
    recordRequest({
      requestId,
      query: input.query,
      status: result.status,
      confidence: 0,
      latency: result.latency,
      usage: result.usage,
      chunkCount: 0,
      voice,
      guardrails: inputReport,
    });
    return result;
  }

  await assertIndexed();

  // ── 2. retrieval ──────────────────────────────────────────────────────────
  const retrieval = await retrieve({ query: input.query, options, timer });

  // ── 3. evidence guardrails ────────────────────────────────────────────────
  timer.start('guardrailsPost');
  const evidence = runRetrievalGuardrails({
    query: input.query,
    chunks: retrieval.chunks,
    agreement: retrieval.agreement,
  });
  timer.end('guardrailsPost');

  if (evidence.insufficientEvidence || options.retrievalOnly) {
    if (options.retrievalOnly) {
      // Retrieval-only mode — return chunks with no answer.
      const report = mergeReports(inputReport.results, evidence.results);
      const result = shortCircuitResult({
        requestId,
        query: input.query,
        transcription: input.transcription ?? null,
        answer: '',
        status: 'answered',
        guardrails: report,
        chunks: retrieval.chunks,
        latency: buildLatency(timer, transcriptionMs, null),
        providers,
      });
      recordRequest({
        requestId, query: input.query, status: 'answered',
        confidence: 0, latency: result.latency, usage: result.usage,
        chunkCount: retrieval.chunks.length, voice, guardrails: report,
      });
      return result;
    }

    // ── Web search + model knowledge fallback ────────────────────────────────
    log.info({ topScore: evidence.topScore }, 'Insufficient RAG evidence — trying web search fallback');

    const webResult = await webSearch(input.query).catch(() => ({ snippets: [], query: input.query, latencyMs: 0 }));
    const webContext = formatWebContext(webResult);

    const fallbackMessages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
      {
        role: 'system',
        content: [
          'You are a helpful, knowledgeable assistant. The user asked a question that was not covered by the knowledge base.',
          'Answer using the web search results below if they are relevant, or use your own knowledge if they are not.',
          'Be concise, accurate, and conversational. Answer in the same language as the question.',
          'If the question is a greeting or casual conversation, respond naturally and warmly.',
          webContext ? `\n${webContext}` : '',
        ].filter(Boolean).join('\n'),
      },
      ...(input.history ?? []).map((turn) => ({
        role: turn.role as 'user' | 'assistant',
        content: turn.content,
      })),
      { role: 'user', content: input.query },
    ];

    timer.start('generation');
    const fallback = await generateCompletion(fallbackMessages, {
      enableThinking: options.enableThinking ?? config.llm.enableThinking,
      signal: input.signal,
    });
    timer.end('generation');

    const report = mergeReports(inputReport.results, evidence.results);
    // The similarity check genuinely failed — the corpus had no usable
    // evidence. Downgrade it from `block` to `warn` because the request did
    // recover into an answer, but leave it visible: the result is reported as
    // `unverified` rather than `answered`, so nothing here is claiming the
    // answer was grounded.
    const similarityResult = report.results.find((r) => r.id === 'similarity_threshold');
    if (similarityResult && similarityResult.verdict === 'block') {
      similarityResult.verdict = 'warn';
      report.blocked = false;
      report.blockedBy = null;
      report.passed = true;
    }

    const answerText = fallback.content.trim() || "I'm sorry, I couldn't find a good answer for that.";
    const answerSource: AnswerSource = webResult.snippets.length > 0 ? 'web' : 'model_knowledge';

    // Measured, not assumed. These used to be the literals 0.6/0.4 with
    // `groundedness: 0, sufficient: true` hardcoded — which reported a
    // fabricated confidence breakdown for exactly the answers least deserving
    // of confidence. Grounding the answer against the web snippets that
    // actually informed it is the honest measurement available here.
    const fallbackGroundedness =
      groundednessAgainstTexts(answerText, webResult.snippets.map((s) => s.snippet));

    const result: QueryResult = {
      requestId,
      query: input.query,
      transcription: input.transcription ?? null,
      answer: answerText,
      status: 'unverified',
      answerSource,
      reasoning: fallback.reasoning || null,
      citations: [],
      chunks: retrieval.chunks,
      confidence: {
        overall: Number((0.55 * fallbackGroundedness).toFixed(4)),
        topScore: evidence.topScore,
        meanScore: evidence.meanScore,
        retrievalAgreement: retrieval.agreement,
        groundedness: Number(fallbackGroundedness.toFixed(4)),
        contextCoverage: 0,
        // Never `true` on this path: the corpus did not support the answer,
        // and that is precisely what this flag is supposed to communicate.
        sufficient: false,
        threshold: config.guardrails.confidenceThreshold,
      },
      guardrails: report,
      latency: buildLatency(timer, transcriptionMs, null),
      usage: fallback.usage,
      model: fallback.model,
      providers,
      createdAt: new Date().toISOString(),
    };
    recordRequest({
      requestId, query: input.query, status: 'unverified',
      confidence: result.confidence.overall, latency: result.latency, usage: result.usage,
      chunkCount: retrieval.chunks.length, voice, guardrails: report,
    });
    return result;
  }

  // ── 4. prompt ─────────────────────────────────────────────────────────────
  timer.start('promptBuilding');
  const prompt = buildPrompt({
    query: input.query,
    chunks: retrieval.chunks,
    history: input.history ?? [],
  });
  timer.end('promptBuilding');

  // ── 5. generation ─────────────────────────────────────────────────────────
  timer.start('generation');
  const generation = await generateCompletion(prompt.messages, {
    enableThinking: options.enableThinking ?? config.llm.enableThinking,
    signal: input.signal,
  });
  timer.end('generation');

  // ── 6. answer guardrails + confidence ─────────────────────────────────────
  timer.start('guardrailsPost');
  const verification = runGenerationGuardrails({
    answer: generation.content,
    chunks: prompt.includedChunks,
    topScore: evidence.topScore,
    meanScore: evidence.meanScore,
    agreement: retrieval.agreement,
    contextCoverage: evidence.contextCoverage,
  });
  timer.end('guardrailsPost');

  const report = mergeReports(inputReport.results, evidence.results, verification.results);
  const { answer, status } = finalizeAnswer(generation.content, verification.confidence.sufficient);

  const result: QueryResult = {
    requestId,
    query: input.query,
    transcription: input.transcription ?? null,
    answer,
    status,
    // The real RAG path: generated strictly from retrieved corpus chunks.
    // `none` when the confidence gate replaced the answer with a refusal.
    answerSource: status === 'answered' ? 'corpus' : 'none',
    reasoning: generation.reasoning || null,
    citations: prompt.citations,
    chunks: retrieval.chunks,
    confidence: verification.confidence,
    guardrails: report,
    latency: buildLatency(timer, transcriptionMs, null),
    usage: generation.usage,
    model: generation.model,
    providers,
    createdAt: new Date().toISOString(),
  };

  log.info(
    {
      status,
      confidence: verification.confidence.overall,
      chunks: retrieval.chunks.length,
      totalMs: Math.round(result.latency.total),
      tokens: generation.usage.totalTokens,
    },
    'Query complete',
  );

  recordRequest({
    requestId,
    query: input.query,
    status,
    confidence: verification.confidence.overall,
    latency: result.latency,
    usage: result.usage,
    chunkCount: retrieval.chunks.length,
    voice,
    guardrails: report,
  });

  return result;
}

/**
 * Streaming variant.
 *
 * Emits stage transitions, the retrieved chunks, then answer tokens as they
 * arrive, and finally the complete result. The UI renders retrieval before
 * the first token exists, which is most of the perceived speed.
 */
export async function* streamQuery(input: QueryInput): AsyncGenerator<StreamEvent, void, unknown> {
  const requestId = input.requestId ?? randomUUID();
  const log = requestLogger(requestId);
  const timer = new StageTimer();
  const options = input.options ?? {};
  const voice = Boolean(input.transcription);
  const transcriptionMs = input.transcription?.latencyMs ?? null;

  yield { type: 'start', requestId, createdAt: new Date().toISOString() };

  if (input.transcription) {
    yield { type: 'transcript', transcription: input.transcription };
  }

  const providers = await describeProviders(voice);

  // ── 1. input guardrails ───────────────────────────────────────────────────
  yield { type: 'stage', stage: 'guardrails', status: 'started' };
  timer.start('guardrailsPre');
  const inputReport = runInputGuardrails(input.query);
  const guardrailMs = timer.end('guardrailsPre');
  yield { type: 'stage', stage: 'guardrails', status: 'completed', durationMs: guardrailMs };
  yield { type: 'guardrails', report: inputReport };

  if (inputReport.blocked) {
    log.warn({ blockedBy: inputReport.blockedBy }, 'Request blocked by input guardrail');
    const result = shortCircuitResult({
      requestId,
      query: input.query,
      transcription: input.transcription ?? null,
      answer: refusalFor(inputReport),
      status: 'blocked',
      guardrails: inputReport,
      chunks: [],
      latency: buildLatency(timer, transcriptionMs, null),
      providers,
    });
    recordRequest({
      requestId,
      query: input.query,
      status: 'blocked',
      confidence: 0,
      latency: result.latency,
      usage: result.usage,
      chunkCount: 0,
      voice,
      guardrails: inputReport,
    });
    yield { type: 'token', delta: result.answer };
    yield { type: 'done', result };
    return;
  }

  await assertIndexed();

  // ── 2. retrieval ──────────────────────────────────────────────────────────
  yield { type: 'stage', stage: 'embedding', status: 'started' };
  yield { type: 'stage', stage: 'retrieval', status: 'started' };
  const retrieval = await retrieve({ query: input.query, options, timer });
  yield {
    type: 'stage',
    stage: 'embedding',
    status: 'completed',
    durationMs: retrieval.timings.embedding,
  };
  yield {
    type: 'stage',
    stage: 'retrieval',
    status: 'completed',
    durationMs: Math.max(retrieval.timings.dense, retrieval.timings.sparse) + retrieval.timings.fusion,
    detail: `${retrieval.candidates.length} candidates`,
  };
  yield {
    type: 'stage',
    stage: 'reranking',
    status: 'completed',
    durationMs: retrieval.timings.rerank,
    detail: retrieval.rerankerProvider,
  };
  yield { type: 'chunks', chunks: retrieval.chunks };

  // ── 3. evidence guardrails ────────────────────────────────────────────────
  timer.start('guardrailsPost');
  const evidence = runRetrievalGuardrails({
    query: input.query,
    chunks: retrieval.chunks,
    agreement: retrieval.agreement,
  });
  timer.end('guardrailsPost');

  if (evidence.insufficientEvidence) {
    // ── Web search + model knowledge fallback ────────────────────────────────
    log.info({ topScore: evidence.topScore }, 'Insufficient RAG evidence — web search fallback');

    yield { type: 'stage', stage: 'generation', status: 'started' };
    timer.start('generation');

    const webResult = await webSearch(input.query).catch(() => ({ snippets: [], query: input.query, latencyMs: 0 }));
    const webContext = formatWebContext(webResult);

    const fallbackMessages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
      {
        role: 'system',
        content: [
          'You are a helpful, knowledgeable assistant. The user asked a question that was not covered by the knowledge base.',
          'Answer using the web search results below if they are relevant, or use your own knowledge if they are not.',
          'Be concise, accurate, and conversational. Answer in the same language as the question.',
          'If the question is a greeting or casual conversation, respond naturally and warmly.',
          webContext ? `\n${webContext}` : '',
        ].filter(Boolean).join('\n'),
      },
      ...(input.history ?? []).map((turn) => ({
        role: turn.role as 'user' | 'assistant',
        content: turn.content,
      })),
      { role: 'user', content: input.query },
    ];

    const report = mergeReports(inputReport.results, evidence.results);
    yield { type: 'guardrails', report };

    const fallbackIterator = streamCompletion(fallbackMessages, {
      enableThinking: options.enableThinking ?? config.llm.enableThinking,
      signal: input.signal,
    });

    let timeToFirstToken: number | null = null;
    let fallback: Awaited<ReturnType<typeof generateCompletion>>;

    for (;;) {
      const step = await fallbackIterator.next();
      if (step.done) { fallback = step.value; break; }
      if (step.value.content) {
        if (timeToFirstToken === null) timeToFirstToken = Math.round(timer.total() * 100) / 100;
        yield { type: 'token', delta: step.value.content };
      }
    }

    const generationMs = timer.end('generation');
    yield { type: 'stage', stage: 'generation', status: 'completed', durationMs: generationMs };

    const fallbackAnswer =
      fallback.content.trim() || "I'm sorry, I couldn't find a good answer for that.";
    const fallbackGroundedness = groundednessAgainstTexts(
      fallbackAnswer,
      webResult.snippets.map((s) => s.snippet),
    );

    const fallbackResult: QueryResult = {
      requestId,
      query: input.query,
      transcription: input.transcription ?? null,
      answer: fallbackAnswer,
      // See the non-streaming path above: not `answered`, because the corpus
      // did not support this — and the confidence below is measured, not the
      // hardcoded 0.6/0.4 this used to report.
      status: 'unverified',
      answerSource: webResult.snippets.length > 0 ? 'web' : 'model_knowledge',
      reasoning: fallback.reasoning || null,
      citations: [],
      chunks: retrieval.chunks,
      confidence: {
        overall: Number((0.55 * fallbackGroundedness).toFixed(4)),
        topScore: evidence.topScore,
        meanScore: evidence.meanScore,
        retrievalAgreement: retrieval.agreement,
        groundedness: Number(fallbackGroundedness.toFixed(4)),
        contextCoverage: 0,
        sufficient: false,
        threshold: config.guardrails.confidenceThreshold,
      },
      guardrails: report,
      latency: buildLatency(timer, transcriptionMs, timeToFirstToken),
      usage: fallback.usage,
      model: fallback.model,
      providers,
      createdAt: new Date().toISOString(),
    };
    recordRequest({
      requestId, query: input.query, status: 'unverified',
      confidence: fallbackResult.confidence.overall, latency: fallbackResult.latency,
      usage: fallbackResult.usage, chunkCount: retrieval.chunks.length, voice, guardrails: report,
    });
    yield { type: 'done', result: fallbackResult };
    return;
  }

  // ── 4. prompt ─────────────────────────────────────────────────────────────
  yield { type: 'stage', stage: 'prompt', status: 'started' };
  timer.start('promptBuilding');
  const prompt = buildPrompt({
    query: input.query,
    chunks: retrieval.chunks,
    history: input.history ?? [],
  });
  const promptMs = timer.end('promptBuilding');
  yield {
    type: 'stage',
    stage: 'prompt',
    status: 'completed',
    durationMs: promptMs,
    detail: `${prompt.citations.length} blocks · ~${prompt.estimatedTokens} tokens`,
  };

  // ── 5. generation ─────────────────────────────────────────────────────────
  yield { type: 'stage', stage: 'generation', status: 'started' };
  timer.start('generation');

  const iterator = streamCompletion(prompt.messages, {
    enableThinking: options.enableThinking ?? config.llm.enableThinking,
    signal: input.signal,
  });

  let timeToFirstToken: number | null = null;
  let generation: Awaited<ReturnType<typeof generateCompletion>>;

  for (;;) {
    const step = await iterator.next();
    if (step.done) {
      generation = step.value;
      break;
    }
    if (step.value.reasoning) {
      yield { type: 'reasoning', delta: step.value.reasoning };
    }
    if (step.value.content) {
      if (timeToFirstToken === null) timeToFirstToken = Math.round(timer.total() * 100) / 100;
      yield { type: 'token', delta: step.value.content };
    }
  }

  const generationMs = timer.end('generation');
  yield { type: 'stage', stage: 'generation', status: 'completed', durationMs: generationMs };

  // ── 6. verification ───────────────────────────────────────────────────────
  yield { type: 'stage', stage: 'verification', status: 'started' };
  timer.start('guardrailsPost');
  const verification = runGenerationGuardrails({
    answer: generation.content,
    chunks: prompt.includedChunks,
    topScore: evidence.topScore,
    meanScore: evidence.meanScore,
    agreement: retrieval.agreement,
    contextCoverage: evidence.contextCoverage,
  });
  const verifyMs = timer.end('guardrailsPost');
  yield { type: 'stage', stage: 'verification', status: 'completed', durationMs: verifyMs };

  const report = mergeReports(inputReport.results, evidence.results, verification.results);
  yield { type: 'guardrails', report };

  const { answer, status } = finalizeAnswer(generation.content, verification.confidence.sufficient);

  const result: QueryResult = {
    requestId,
    query: input.query,
    transcription: input.transcription ?? null,
    answer,
    status,
    // Grounded in retrieved corpus chunks — the real RAG path.
    answerSource: status === 'answered' ? 'corpus' : 'none',
    reasoning: generation.reasoning || null,
    citations: prompt.citations,
    chunks: retrieval.chunks,
    confidence: verification.confidence,
    guardrails: report,
    latency: buildLatency(timer, transcriptionMs, timeToFirstToken),
    usage: generation.usage,
    model: generation.model,
    providers,
    createdAt: new Date().toISOString(),
  };

  log.info(
    {
      status,
      confidence: verification.confidence.overall,
      chunks: retrieval.chunks.length,
      totalMs: Math.round(result.latency.total),
      ttftMs: timeToFirstToken,
      tokens: generation.usage.totalTokens,
    },
    'Streaming query complete',
  );

  recordRequest({
    requestId,
    query: input.query,
    status,
    confidence: verification.confidence.overall,
    latency: result.latency,
    usage: result.usage,
    chunkCount: retrieval.chunks.length,
    voice,
    guardrails: report,
  });

  yield { type: 'done', result };
}

/**
 * Apply the confidence gate.
 *
 * The model may already have refused on its own, in which case its wording is
 * kept. Otherwise a below-threshold answer is replaced outright rather than
 * hedged — a confident-sounding answer we do not trust is the worst outcome.
 */
function finalizeAnswer(
  content: string,
  sufficient: boolean,
): { answer: string; status: AnswerStatus } {
  const trimmed = content.trim();

  if (!trimmed) {
    return { answer: INSUFFICIENT_EVIDENCE_MESSAGE, status: 'insufficient_context' };
  }
  if (trimmed.includes(INSUFFICIENT_EVIDENCE_MESSAGE)) {
    return { answer: INSUFFICIENT_EVIDENCE_MESSAGE, status: 'insufficient_context' };
  }
  if (!sufficient) {
    return { answer: LOW_CONFIDENCE_MESSAGE, status: 'low_confidence' };
  }
  return { answer: trimmed, status: 'answered' };
}

/** User-facing message for a blocked request. */
function refusalFor(report: GuardrailReport): string {
  switch (report.blockedBy) {
    case 'prompt_injection':
      return 'This request was blocked because it tried to override the assistant’s instructions.';
    case 'jailbreak':
      return 'This request was blocked because it tried to bypass safety constraints.';
    case 'toxicity':
      return 'This request was blocked because it asks for harmful content.';
    default:
      return 'This request was blocked by a safety guardrail.';
  }
}

export { now };
