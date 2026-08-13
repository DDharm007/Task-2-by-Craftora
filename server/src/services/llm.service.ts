/**
 * Chat client for the generation half of the RAG pipeline.
 *
 * Uses the OpenAI SDK against an OpenAI-compatible endpoint — configured via
 * `GROQ_*` (default: Groq, `llama-3.3-70b-versatile`), but any compatible
 * gateway works by repointing GROQ_BASE_URL/GROQ_MODEL.
 *
 * Two behaviours shape the implementation:
 *
 *  • Reasoning models (NVIDIA NIM/Nemotron) emit `reasoning_content` deltas
 *    before any answer tokens when `LLM_ENABLE_THINKING` is on, and a grounded
 *    RAG answer can take 30-120s. Thinking is off by default and opt-in per
 *    request; the reasoning stream is surfaced separately when enabled. The
 *    fields it adds are NIM-specific — Groq 400s on them, so leave it off
 *    unless pointed at a NIM deployment.
 *  • Non-streaming requests on very large models routinely exceed a
 *    two-minute gateway timeout. We always stream on the wire, and synthesise
 *    a non-streaming response by accumulating deltas when the caller wants one.
 */
import OpenAI from 'openai';
import type { TokenUsage } from '@voxrag/shared';
import { config } from '../config/env.js';
import { logger } from '../utils/logger.js';
import { retry, withTimeout } from '../utils/async.js';
import { errors } from '../utils/errors.js';
import type { ChatMessage } from '../rag/prompt/index.js';

export interface GenerationOptions {
  enableThinking?: boolean;
  temperature?: number;
  maxTokens?: number;
  /** Aborts the upstream request when the client disconnects. */
  signal?: AbortSignal;
}

export interface GenerationChunk {
  /** Answer text delta. */
  content?: string;
  /** Chain-of-thought delta, only present in thinking mode. */
  reasoning?: string;
}

export interface GenerationResult {
  content: string;
  reasoning: string;
  usage: TokenUsage;
  model: string;
  finishReason: string | null;
}

const EMPTY_USAGE: TokenUsage = {
  promptTokens: 0,
  completionTokens: 0,
  reasoningTokens: 0,
  totalTokens: 0,
};

let client: OpenAI | null = null;

function getClient(): OpenAI {
  if (!client) {
    client = new OpenAI({
      apiKey: config.llm.apiKey,
      baseURL: config.llm.baseUrl,
      timeout: config.llm.timeoutMs,
      // Retries are handled by our own policy so the two do not compound.
      maxRetries: 0,
    });
  }
  return client;
}

/** Body shared by both entry points. NIM-specific fields are not in the SDK's types. */
function buildRequestBody(messages: readonly ChatMessage[], options: GenerationOptions) {
  const enableThinking = options.enableThinking ?? config.llm.enableThinking;
  return {
    model: config.llm.model,
    messages: messages as Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
    temperature: options.temperature ?? config.llm.temperature,
    top_p: config.llm.topP,
    max_tokens: options.maxTokens ?? config.llm.maxTokens,
    stream: true as const,
    stream_options: { include_usage: true },
    // NVIDIA extensions — passed through the SDK's escape hatch below.
    ...(enableThinking
      ? {
          reasoning_budget: config.llm.reasoningBudget,
          ...(config.llm.model.includes('nemotron') ? { chat_template_kwargs: { enable_thinking: true } } : {}),
        }
      : (config.llm.model.includes('nemotron') ? { chat_template_kwargs: { enable_thinking: false } } : {})),
  };
}

/** True for transport hiccups and upstream 429/5xx — never for a malformed request. */
function isRetryable(error: unknown): boolean {
  const status = (error as { status?: number }).status;
  if (status === undefined) return true;
  return status === 408 || status === 409 || status === 429 || status >= 500;
}

/**
 * Stream a completion.
 *
 * Yields answer and reasoning deltas as they arrive; returns the accumulated
 * result. The connection is retried only before the first token — once we have
 * emitted text to the client, restarting would duplicate output.
 */
export async function* streamCompletion(
  messages: readonly ChatMessage[],
  options: GenerationOptions = {},
): AsyncGenerator<GenerationChunk, GenerationResult, unknown> {
  const body = buildRequestBody(messages, options);
  let content = '';
  let reasoning = '';
  let usage: TokenUsage = { ...EMPTY_USAGE };
  let finishReason: string | null = null;
  let emitted = false;

  const openStream = async () => {
    const stream = await getClient().chat.completions.create(
      body as unknown as Parameters<OpenAI['chat']['completions']['create']>[0],
      { signal: options.signal },
    );
    return stream as unknown as AsyncIterable<{
      choices?: Array<{
        delta?: { content?: string | null; reasoning_content?: string | null };
        finish_reason?: string | null;
      }>;
      usage?: {
        prompt_tokens?: number;
        completion_tokens?: number;
        total_tokens?: number;
        completion_tokens_details?: { reasoning_tokens?: number };
      } | null;
    }>;
  };

  let stream: Awaited<ReturnType<typeof openStream>>;
  try {
    stream = await retry(openStream, {
      retries: config.llm.maxRetries,
      baseDelayMs: 800,
      shouldRetry: (error) => !emitted && isRetryable(error),
      onRetry: (error, attempt, delayMs) =>
        logger.warn(
          { attempt, delayMs, error: (error as Error).message },
          'Retrying Nemotron stream connection',
        ),
    });
  } catch (error) {
    throw toLlmError(error);
  }

  try {
    for await (const event of stream) {
      const choice = event.choices?.[0];
      if (choice?.finish_reason) finishReason = choice.finish_reason;

      const reasoningDelta = choice?.delta?.reasoning_content;
      if (reasoningDelta) {
        reasoning += reasoningDelta;
        yield { reasoning: reasoningDelta };
      }

      const contentDelta = choice?.delta?.content;
      if (contentDelta) {
        content += contentDelta;
        emitted = true;
        yield { content: contentDelta };
      }

      // NIM reports usage on the final chunk when include_usage is set.
      if (event.usage) {
        usage = {
          promptTokens: event.usage.prompt_tokens ?? 0,
          completionTokens: event.usage.completion_tokens ?? 0,
          reasoningTokens: event.usage.completion_tokens_details?.reasoning_tokens ?? 0,
          totalTokens: event.usage.total_tokens ?? 0,
        };
      }
    }
  } catch (error) {
    // A client disconnect surfaces as an abort; that is not a failure.
    if ((error as Error).name === 'AbortError') {
      logger.debug('Generation aborted by client');
    } else {
      throw toLlmError(error);
    }
  }

  if (usage.totalTokens === 0) {
    // Fall back to an estimate when the gateway omitted usage.
    const promptChars = messages.reduce((sum, message) => sum + message.content.length, 0);
    usage = {
      promptTokens: Math.ceil(promptChars / 4),
      completionTokens: Math.ceil(content.length / 4),
      reasoningTokens: Math.ceil(reasoning.length / 4),
      totalTokens: Math.ceil((promptChars + content.length + reasoning.length) / 4),
    };
  }

  return { content, reasoning, usage, model: config.llm.model, finishReason };
}

/**
 * Non-streaming completion.
 *
 * Still streams on the wire — see the note at the top of this file — and
 * collapses the deltas into one result.
 */
export async function generateCompletion(
  messages: readonly ChatMessage[],
  options: GenerationOptions = {},
): Promise<GenerationResult> {
  const run = async (): Promise<GenerationResult> => {
    const iterator = streamCompletion(messages, options);
    for (;;) {
      const step = await iterator.next();
      if (step.done) return step.value;
    }
  };

  try {
    return await withTimeout(run(), config.llm.timeoutMs, `Nemotron did not respond within ${config.llm.timeoutMs}ms`);
  } catch (error) {
    throw toLlmError(error);
  }
}

function toLlmError(error: unknown): Error {
  const err = error as { status?: number; message?: string; name?: string };
  if (err.name === 'TimeoutError' || err.name === 'AbortError') {
    return errors.llmTimeout(config.llm.timeoutMs);
  }
  const status = err.status;
  if (status === 401 || status === 403) {
    return errors.llm('The LLM provider rejected the API key. Check GROQ_API_KEY.', error);
  }
  if (status === 404) {
    return errors.llm(
      `Model ${config.llm.model} was not found on ${config.llm.baseUrl}. Check GROQ_MODEL.`,
      error,
    );
  }
  if (status === 429) {
    // Surface the provider's own wording. "Retry shortly" is actively
    // misleading when the limit is a *daily* quota that resets in an hour —
    // and the message distinguishes a per-minute blip from an exhausted
    // budget, which need completely different responses.
    const detail = err.message ?? '';
    const window = /tokens per day|TPD/i.test(detail) ? 'daily token quota exhausted' : 'rate limit reached';
    const retryIn = /try again in ([^.]+)/i.exec(detail)?.[1];
    return errors.llm(
      `LLM ${window}.${retryIn ? ` Retry in ${retryIn.trim()}.` : ' Retry shortly.'}`,
      error,
    );
  }
  if (status === 400 && /reasoning_budget|enable_thinking|chat_template_kwargs/i.test(err.message ?? '')) {
    // Highly specific because it is the exact misconfiguration that made every
    // query 502 until it was tracked down: NIM-only thinking fields sent to a
    // provider that does not accept them.
    return errors.llm(
      `${config.llm.baseUrl} rejected the reasoning parameters. Set LLM_ENABLE_THINKING=false — ` +
        'thinking is NVIDIA NIM/Nemotron-only.',
      error,
    );
  }
  return errors.llm(`LLM request failed: ${err.message ?? String(error)}`, error);
}

/** Liveness probe used by /api/health. Deliberately tiny. */
export async function llmHealthCheck(): Promise<{ ok: boolean; detail: string }> {
  try {
    const result = await withTimeout(
      generateCompletion([{ role: 'user', content: 'Reply with the single word: ok' }], {
        enableThinking: false,
        maxTokens: 16,
      }),
      20_000,
      'health check timed out',
    );
    return { ok: true, detail: `${config.llm.model} · ${result.usage.totalTokens} tokens` };
  } catch (error) {
    return { ok: false, detail: (error as Error).message };
  }
}
