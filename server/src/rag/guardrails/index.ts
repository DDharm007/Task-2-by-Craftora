/**
 * The guardrail engine.
 *
 * Eight checks across three stages:
 *
 *   pre-retrieval   prompt_injection, jailbreak, toxicity
 *   post-retrieval  similarity_threshold, off_topic, context_verification
 *   post-generation hallucination, confidence
 *
 * Pre-retrieval checks are cheap regex/lexical passes that run before any
 * model is touched, so an attack costs nothing. Post-retrieval checks decide
 * whether the evidence is good enough to answer at all. Post-generation checks
 * verify the answer against the evidence that was actually supplied.
 *
 * A `block` verdict short-circuits the pipeline; `warn` is recorded and
 * lowers confidence without stopping the request.
 */
import type {
  ConfidenceBreakdown,
  GuardrailReport,
  GuardrailResult,
  RetrievedChunk,
} from '@goarag/shared';
import { config } from '../../config/env.js';
import { now } from '../../utils/async.js';
import { contentTokens, coverage, splitSentences, tokenize } from '../../utils/text.js';
import {
  evaluateRules,
  JAILBREAK_RULES,
  OUT_OF_SCOPE_HINTS,
  PROMPT_INJECTION_RULES,
  TOXICITY_RULES,
} from './patterns.js';

/** Score above which a pre-retrieval check blocks the request. */
const BLOCK_THRESHOLD = 0.8;
/** Score above which a check is recorded as a warning. */
const WARN_THRESHOLD = 0.45;

function makeResult(
  id: GuardrailResult['id'],
  stage: GuardrailResult['stage'],
  score: number,
  threshold: number,
  reason: string,
  evidence: string[],
  startedAt: number,
  verdictOverride?: GuardrailResult['verdict'],
): GuardrailResult {
  const verdict =
    verdictOverride ?? (score >= BLOCK_THRESHOLD ? 'block' : score >= WARN_THRESHOLD ? 'warn' : 'pass');
  return {
    id,
    stage,
    verdict,
    score: Number(score.toFixed(4)),
    threshold,
    reason,
    evidence,
    durationMs: Number((now() - startedAt).toFixed(3)),
  };
}

// ─── Stage 1: pre-retrieval ──────────────────────────────────────────────────

/**
 * Input guardrails. Run before embedding so a hostile query never reaches a
 * model or the vector store.
 */
export function runInputGuardrails(query: string): GuardrailReport {
  const startedAt = now();
  const results: GuardrailResult[] = [];

  if (!config.guardrails.enabled) {
    return { passed: true, blocked: false, blockedBy: null, results, totalDurationMs: 0 };
  }

  // 1. Prompt injection
  {
    const t = now();
    const { score, matches } = evaluateRules(query, PROMPT_INJECTION_RULES);
    results.push(
      makeResult(
        'prompt_injection',
        'pre_retrieval',
        score,
        BLOCK_THRESHOLD,
        score >= BLOCK_THRESHOLD
          ? 'Query attempts to override system instructions'
          : score >= WARN_THRESHOLD
            ? 'Query contains instruction-like phrasing'
            : 'No injection patterns detected',
        matches,
        t,
      ),
    );
  }

  // 2. Jailbreak
  {
    const t = now();
    const { score, matches } = evaluateRules(query, JAILBREAK_RULES);
    results.push(
      makeResult(
        'jailbreak',
        'pre_retrieval',
        score,
        BLOCK_THRESHOLD,
        score >= BLOCK_THRESHOLD
          ? 'Query attempts to bypass safety constraints'
          : score >= WARN_THRESHOLD
            ? 'Query contains constraint-evasion phrasing'
            : 'No jailbreak patterns detected',
        matches,
        t,
      ),
    );
  }

  // 3. Toxicity
  {
    const t = now();
    const { score, matches } = evaluateRules(query, TOXICITY_RULES);
    results.push(
      makeResult(
        'toxicity',
        'pre_retrieval',
        score,
        BLOCK_THRESHOLD,
        score >= BLOCK_THRESHOLD
          ? 'Query solicits harmful content'
          : score >= WARN_THRESHOLD
            ? 'Query contains potentially harmful phrasing'
            : 'No toxic content detected',
        matches,
        t,
      ),
    );
  }

  const blocking = results.find((result) => result.verdict === 'block');
  return {
    passed: !blocking,
    blocked: Boolean(blocking),
    blockedBy: blocking?.id ?? null,
    results,
    totalDurationMs: Number((now() - startedAt).toFixed(3)),
  };
}

// ─── Stage 2: post-retrieval ─────────────────────────────────────────────────

export interface RetrievalGuardrailInput {
  query: string;
  chunks: readonly RetrievedChunk[];
  agreement: number;
}

export interface RetrievalGuardrailOutput {
  results: GuardrailResult[];
  /** True when there is not enough evidence to attempt an answer. */
  insufficientEvidence: boolean;
  topScore: number;
  meanScore: number;
  contextCoverage: number;
}

/**
 * Evidence guardrails. Decide whether retrieval produced anything worth
 * sending to the LLM — the single most effective defence against
 * hallucination is simply not asking the question when the evidence is thin.
 */
export function runRetrievalGuardrails(input: RetrievalGuardrailInput): RetrievalGuardrailOutput {
  const { query, chunks, agreement } = input;
  const results: GuardrailResult[] = [];

  const scores = chunks.map((chunk) => chunk.score);
  const topScore = scores.length > 0 ? Math.max(...scores) : 0;
  const meanScore = scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;

  // How much of the query's vocabulary appears in the retrieved context.
  const contextTokens = new Set<string>();
  for (const chunk of chunks) {
    for (const token of tokenize(`${chunk.text} ${chunk.parentText ?? ''}`)) contextTokens.add(token);
  }
  const queryTerms = contentTokens(query);
  const contextCoverage = coverage(queryTerms, contextTokens);

  if (!config.guardrails.enabled) {
    return { results, insufficientEvidence: chunks.length === 0, topScore, meanScore, contextCoverage };
  }

  // 4. Similarity threshold
  {
    const t = now();
    const threshold = config.guardrails.similarityThreshold;
    const failed = chunks.length === 0 || topScore < threshold;
    results.push(
      makeResult(
        'similarity_threshold',
        'post_retrieval',
        topScore,
        threshold,
        chunks.length === 0
          ? 'Retrieval returned no chunks'
          : failed
            ? `Best chunk scored ${topScore.toFixed(3)}, below the ${threshold} threshold`
            : `Best chunk scored ${topScore.toFixed(3)}`,
        chunks.slice(0, 3).map((chunk) => `${chunk.id.slice(0, 8)}: ${chunk.score.toFixed(3)}`),
        t,
        failed ? 'block' : 'pass',
      ),
    );
  }

  // 5. Off-topic
  {
    const t = now();
    const { score: hintScore, matches } = evaluateRules(query, OUT_OF_SCOPE_HINTS);
    // Combine an explicit out-of-scope hint with weak lexical grounding. Either
    // alone is unreliable: a corpus question can use generative phrasing, and
    // a cross-lingual question legitimately shares few tokens with its source.
    const lexicalMiss = 1 - contextCoverage;
    const offTopicScore = Math.min(1, hintScore * 0.6 + lexicalMiss * 0.4);
    const isOffTopic = hintScore >= 0.6 && contextCoverage < 0.25;
    results.push(
      makeResult(
        'off_topic',
        'post_retrieval',
        offTopicScore,
        0.6,
        isOffTopic
          ? 'Query appears to fall outside the indexed corpus'
          : `Query overlaps the retrieved context (coverage ${(contextCoverage * 100).toFixed(0)}%)`,
        matches,
        t,
        isOffTopic ? 'warn' : 'pass',
      ),
    );
  }

  // 6. Context verification
  {
    const t = now();
    // Do the retrieved chunks actually corroborate each other, or is the top
    // hit an isolated outlier? Agreement between the retrieval arms plus
    // score consistency across the top chunks is a good proxy.
    const spread = scores.length > 1 ? topScore - (scores[scores.length - 1] as number) : 0;
    const consistency = topScore > 0 ? 1 - Math.min(1, spread / Math.max(topScore, 1e-6)) : 0;
    const verification = 0.5 * agreement + 0.3 * consistency + 0.2 * Math.min(1, chunks.length / 3);
    const weak = verification < 0.25;
    results.push(
      makeResult(
        'context_verification',
        'post_retrieval',
        verification,
        0.25,
        weak
          ? 'Retrieved chunks corroborate each other weakly'
          : `Context verified (arm agreement ${(agreement * 100).toFixed(0)}%, ${chunks.length} chunks)`,
        [
          `agreement=${agreement.toFixed(3)}`,
          `consistency=${consistency.toFixed(3)}`,
          `chunks=${chunks.length}`,
        ],
        t,
        weak ? 'warn' : 'pass',
      ),
    );
  }

  const insufficientEvidence = results.some(
    (result) => result.id === 'similarity_threshold' && result.verdict === 'block',
  );

  return { results, insufficientEvidence, topScore, meanScore, contextCoverage };
}

// ─── Stage 3: post-generation ────────────────────────────────────────────────

export interface GenerationGuardrailInput {
  answer: string;
  chunks: readonly RetrievedChunk[];
  topScore: number;
  meanScore: number;
  agreement: number;
  contextCoverage: number;
}

export interface GenerationGuardrailOutput {
  results: GuardrailResult[];
  confidence: ConfidenceBreakdown;
  groundedness: number;
  hallucinated: boolean;
}

/**
 * Verify the generated answer against the context it was given, then compute
 * the final confidence score.
 */
export function runGenerationGuardrails(input: GenerationGuardrailInput): GenerationGuardrailOutput {
  const { answer, chunks, topScore, meanScore, agreement, contextCoverage } = input;
  const results: GuardrailResult[] = [];

  const groundedness = computeGroundedness(answer, chunks);

  // 7. Hallucination
  if (config.guardrails.enabled && config.guardrails.hallucinationCheck) {
    const t = now();
    const threshold = config.guardrails.groundednessThreshold;
    const hallucinationScore = 1 - groundedness;
    const failed = groundedness < threshold;
    results.push(
      makeResult(
        'hallucination',
        'post_generation',
        hallucinationScore,
        1 - threshold,
        failed
          ? `Only ${(groundedness * 100).toFixed(0)}% of the answer is supported by the retrieved context`
          : `${(groundedness * 100).toFixed(0)}% of the answer is supported by the retrieved context`,
        [`groundedness=${groundedness.toFixed(3)}`, `threshold=${threshold}`],
        t,
        failed ? 'warn' : 'pass',
      ),
    );
  }

  // 8. Confidence
  const confidence = computeConfidence({
    topScore,
    meanScore,
    agreement,
    groundedness,
    contextCoverage,
  });

  if (config.guardrails.enabled) {
    const t = now();
    results.push(
      makeResult(
        'confidence',
        'post_generation',
        confidence.overall,
        confidence.threshold,
        confidence.sufficient
          ? `Confidence ${(confidence.overall * 100).toFixed(0)}% clears the ${(confidence.threshold * 100).toFixed(0)}% threshold`
          : `Confidence ${(confidence.overall * 100).toFixed(0)}% is below the ${(confidence.threshold * 100).toFixed(0)}% threshold`,
        [
          `top=${topScore.toFixed(3)}`,
          `mean=${meanScore.toFixed(3)}`,
          `agreement=${agreement.toFixed(3)}`,
          `grounded=${groundedness.toFixed(3)}`,
          `coverage=${contextCoverage.toFixed(3)}`,
        ],
        t,
        confidence.sufficient ? 'pass' : 'block',
      ),
    );
  }

  return {
    results,
    confidence,
    groundedness,
    hallucinated: groundedness < config.guardrails.groundednessThreshold,
  };
}

/**
 * Fraction of the answer's sentences that are supported by the context.
 *
 * For each sentence we take its content words and measure how many appear in
 * the retrieved text. A sentence is "grounded" when most of its substance is
 * traceable to the evidence. This catches the common failure mode — a fluent
 * answer that quietly introduces facts absent from the sources — without
 * needing a second model call.
 *
 * Citation markers, refusals and boilerplate are excluded so they neither
 * inflate nor deflate the score.
 */
export function computeGroundedness(answer: string, chunks: readonly RetrievedChunk[]): number {
  return groundednessAgainstTexts(
    answer,
    chunks.map((chunk) => `${chunk.text} ${chunk.parentText ?? ''}`),
  );
}

/**
 * The same measurement against arbitrary source text.
 *
 * Used for the web-search fallback, whose sources are snippets rather than
 * retrieved chunks — that path previously reported a hardcoded groundedness
 * of 0 alongside a hardcoded confidence, which is exactly where a real number
 * matters most.
 */
export function groundednessAgainstTexts(answer: string, sources: readonly string[]): number {
  if (!answer.trim() || sources.length === 0) return 0;

  const contextTokens = new Set<string>();
  for (const source of sources) {
    for (const token of tokenize(source)) contextTokens.add(token);
  }
  if (contextTokens.size === 0) return 0;

  const stripped = answer.replace(/\[\d+\]/gu, ' ');
  const sentences = splitSentences(stripped).filter((sentence) => {
    const terms = contentTokens(sentence.text);
    // Ignore sentences with too little substance to judge.
    return terms.length >= 3;
  });

  if (sentences.length === 0) {
    // Single short answer ("42", "Yes") — judge the whole string instead.
    const terms = contentTokens(stripped);
    return terms.length === 0 ? 0 : coverage(terms, contextTokens);
  }

  let supported = 0;
  for (const sentence of sentences) {
    const terms = contentTokens(sentence.text);
    // 60% of a sentence's content words appearing in the context is a strong
    // signal it was drawn from there rather than from model priors.
    if (coverage(terms, contextTokens) >= 0.6) supported += 1;
  }
  return supported / sentences.length;
}

/**
 * Blend retrieval and generation signals into one confidence score.
 *
 * Weights favour reranker score (the most direct measure of whether the
 * evidence answers the question) and groundedness (whether the answer used
 * it), with arm agreement and coverage as corroborating signals.
 */
export function computeConfidence(input: {
  topScore: number;
  meanScore: number;
  agreement: number;
  groundedness: number;
  contextCoverage: number;
}): ConfidenceBreakdown {
  const { topScore, meanScore, agreement, groundedness, contextCoverage } = input;

  const overall =
    0.35 * clamp01(topScore) +
    0.15 * clamp01(meanScore) +
    0.3 * clamp01(groundedness) +
    0.1 * clamp01(agreement) +
    0.1 * clamp01(contextCoverage);

  const threshold = config.guardrails.confidenceThreshold;

  return {
    overall: Number(overall.toFixed(4)),
    topScore: Number(clamp01(topScore).toFixed(4)),
    meanScore: Number(clamp01(meanScore).toFixed(4)),
    retrievalAgreement: Number(clamp01(agreement).toFixed(4)),
    groundedness: Number(clamp01(groundedness).toFixed(4)),
    contextCoverage: Number(clamp01(contextCoverage).toFixed(4)),
    sufficient: overall >= threshold,
    threshold,
  };
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

/** Merge staged guardrail results into the single report the API returns. */
export function mergeReports(...reports: Array<GuardrailResult[]>): GuardrailReport {
  const results = reports.flat();
  const blocking = results.find((result) => result.verdict === 'block');
  return {
    passed: !blocking,
    blocked: Boolean(blocking),
    blockedBy: blocking?.id ?? null,
    results,
    totalDurationMs: Number(results.reduce((sum, result) => sum + result.durationMs, 0).toFixed(3)),
  };
}
