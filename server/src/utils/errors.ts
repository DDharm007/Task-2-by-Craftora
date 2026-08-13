/**
 * Structured application errors.
 *
 * Every error crossing the HTTP boundary becomes an `AppError` so responses
 * carry a stable machine-readable `code` alongside a human message.
 */
import type { ApiErrorBody, ErrorCode } from '@voxrag/shared';

export class AppError extends Error {
  readonly code: ErrorCode;
  readonly statusCode: number;
  readonly details: unknown;
  /** True when retrying the same request could plausibly succeed. */
  readonly retryable: boolean;

  constructor(
    code: ErrorCode,
    message: string,
    options: { statusCode?: number; details?: unknown; retryable?: boolean; cause?: unknown } = {},
  ) {
    super(message, { cause: options.cause });
    this.name = 'AppError';
    this.code = code;
    this.statusCode = options.statusCode ?? defaultStatusFor(code);
    this.details = options.details;
    this.retryable = options.retryable ?? false;
    Error.captureStackTrace?.(this, AppError);
  }

  toBody(requestId: string, includeDetails: boolean): ApiErrorBody {
    return {
      code: this.code,
      message: this.message,
      requestId,
      ...(includeDetails && this.details !== undefined ? { details: this.details } : {}),
    };
  }
}

function defaultStatusFor(code: ErrorCode): number {
  switch (code) {
    case 'VALIDATION_ERROR':
      return 400;
    case 'GUARDRAIL_BLOCKED':
      return 422;
    case 'RATE_LIMITED':
      return 429;
    case 'PAYLOAD_TOO_LARGE':
      return 413;
    case 'UNSUPPORTED_MEDIA_TYPE':
      return 415;
    case 'NOT_FOUND':
      return 404;
    case 'INDEX_EMPTY':
      return 503;
    case 'VECTOR_STORE_UNAVAILABLE':
      return 503;
    case 'LLM_TIMEOUT':
      return 504;
    case 'STT_FAILED':
    case 'TTS_FAILED':
    case 'EMBEDDING_FAILED':
    case 'LLM_FAILED':
      return 502;
    default:
      return 500;
  }
}

export const errors = {
  validation: (message: string, details?: unknown) =>
    new AppError('VALIDATION_ERROR', message, { details }),
  notFound: (message = 'Resource not found') => new AppError('NOT_FOUND', message),
  indexEmpty: () =>
    new AppError(
      'INDEX_EMPTY',
      'The vector index is empty. Run `npm run index` to download and index the dataset.',
    ),
  vectorStore: (message: string, cause?: unknown) =>
    new AppError('VECTOR_STORE_UNAVAILABLE', message, { cause, retryable: true }),
  embedding: (message: string, cause?: unknown) =>
    new AppError('EMBEDDING_FAILED', message, { cause, retryable: true }),
  stt: (message: string, details?: unknown, cause?: unknown) =>
    new AppError('STT_FAILED', message, { details, cause, retryable: true }),
  tts: (message: string, statusCode = 502, retryable = false) =>
    new AppError('TTS_FAILED', message, { statusCode, retryable }),
  llm: (message: string, cause?: unknown) =>
    new AppError('LLM_FAILED', message, { cause, retryable: true }),
  llmTimeout: (ms: number) =>
    new AppError('LLM_TIMEOUT', `The language model did not respond within ${ms}ms.`, {
      retryable: true,
    }),
  internal: (message = 'An unexpected error occurred', cause?: unknown) =>
    new AppError('INTERNAL_ERROR', message, { cause }),
};

/** Normalise anything thrown into an `AppError`. */
export function toAppError(err: unknown): AppError {
  if (err instanceof AppError) return err;
  if (err instanceof Error) {
    // Undici/fetch abort surfaces as a TimeoutError or AbortError.
    if (err.name === 'AbortError' || err.name === 'TimeoutError') {
      return new AppError('LLM_TIMEOUT', 'The upstream request timed out.', {
        cause: err,
        retryable: true,
      });
    }
    return new AppError('INTERNAL_ERROR', err.message, { cause: err });
  }
  return new AppError('INTERNAL_ERROR', String(err));
}
