/**
 * Express middleware: request identity, validation, uploads, rate limiting and
 * the terminal error handler.
 */
import { randomUUID } from 'node:crypto';
import type { ErrorRequestHandler, NextFunction, Request, RequestHandler, Response } from 'express';
import rateLimit, { type RateLimitRequestHandler } from 'express-rate-limit';
import multer from 'multer';
import { ZodError, type ZodSchema } from 'zod';
import { SUPPORTED_AUDIO_MIME_TYPES } from '@goarag/shared';
import { config } from '../config/env.js';
import { logger } from '../utils/logger.js';
import { AppError, toAppError } from '../utils/errors.js';

declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace Express {
    interface Request {
      requestId: string;
      startedAt: number;
    }
  }
}

/** Attach a request id, echo it back, and log the outcome. */
export const requestContext: RequestHandler = (req, res, next) => {
  const incoming = req.header('x-request-id');
  req.requestId = incoming && /^[\w-]{1,64}$/.test(incoming) ? incoming : randomUUID();
  req.startedAt = Date.now();
  res.setHeader('x-request-id', req.requestId);

  res.on('finish', () => {
    const durationMs = Date.now() - req.startedAt;
    const level = res.statusCode >= 500 ? 'error' : res.statusCode >= 400 ? 'warn' : 'info';
    logger[level](
      {
        requestId: req.requestId,
        method: req.method,
        path: req.originalUrl.split('?')[0],
        status: res.statusCode,
        durationMs,
      },
      'request',
    );
  });

  next();
};

/** Validate `req.body` against a schema, replacing it with the parsed value. */
export function validateBody<T>(schema: ZodSchema<T>): RequestHandler {
  return (req, _res, next) => {
    try {
      req.body = schema.parse(req.body ?? {});
      next();
    } catch (error) {
      next(asValidationError(error));
    }
  };
}

/** Validate `req.query`. Express 4 allows reassignment; Express 5 does not. */
export function validateQuery<T>(schema: ZodSchema<T>): RequestHandler {
  return (req, _res, next) => {
    try {
      const parsed = schema.parse(req.query ?? {});
      // Stash the parsed value rather than mutating req.query, which is a
      // getter on some Express versions.
      (req as Request & { validatedQuery?: unknown }).validatedQuery = parsed;
      next();
    } catch (error) {
      next(asValidationError(error));
    }
  };
}

/** Read the value stored by `validateQuery`. */
export function validatedQuery<T>(req: Request): T {
  return (req as Request & { validatedQuery: T }).validatedQuery;
}

function asValidationError(error: unknown): AppError {
  if (error instanceof ZodError) {
    return new AppError('VALIDATION_ERROR', 'Request validation failed', {
      details: error.issues.map((issue) => ({
        path: issue.path.join('.'),
        message: issue.message,
      })),
    });
  }
  return toAppError(error);
}

/**
 * Audio upload handling.
 *
 * Memory storage: files go straight to ElevenLabs and are never persisted, so
 * writing them to disk would only add I/O and a cleanup obligation. The size
 * limit is enforced by multer before the body is buffered.
 */
export const audioUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: config.stt.maxUploadBytes, files: 1, fields: 12 },
  fileFilter: (_req, file, callback) => {
    // Browsers append codec parameters, e.g. `audio/webm;codecs=opus`.
    const mime = file.mimetype.split(';')[0]?.trim().toLowerCase() ?? '';
    if ((SUPPORTED_AUDIO_MIME_TYPES as readonly string[]).includes(mime)) {
      callback(null, true);
      return;
    }
    callback(
      new AppError(
        'UNSUPPORTED_MEDIA_TYPE',
        `Unsupported audio type "${file.mimetype}". Supported: ${SUPPORTED_AUDIO_MIME_TYPES.join(', ')}`,
      ),
    );
  },
});

/** Translate multer's own errors into structured AppErrors. */
export const handleUploadErrors: ErrorRequestHandler = (err, _req, _res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      next(
        new AppError(
          'PAYLOAD_TOO_LARGE',
          `Audio file exceeds the ${config.stt.maxUploadBytes / 1024 / 1024}MB limit.`,
        ),
      );
      return;
    }
    next(new AppError('VALIDATION_ERROR', `Upload rejected: ${err.message}`));
    return;
  }
  next(err);
};

function makeLimiter(max: number, name: string): RateLimitRequestHandler {
  return rateLimit({
    windowMs: config.rateLimit.windowMs,
    max,
    standardHeaders: 'draft-7',
    legacyHeaders: false,
    // Skip limiting in tests so the suite is not rate-limited against itself.
    skip: () => config.env === 'test',
    handler: (req, _res, next) => {
      logger.warn({ requestId: req.requestId, limiter: name, ip: req.ip }, 'Rate limit exceeded');
      next(
        new AppError('RATE_LIMITED', 'Too many requests. Please slow down and try again shortly.', {
          details: { windowMs: config.rateLimit.windowMs, max },
        }),
      );
    },
  });
}

/** Global limiter, plus tighter ones on the endpoints that cost real money. */
export const globalRateLimit = makeLimiter(config.rateLimit.max, 'global');
export const queryRateLimit = makeLimiter(config.rateLimit.queryMax, 'query');
export const transcribeRateLimit = makeLimiter(config.rateLimit.transcribeMax, 'transcribe');

/** 404 handler for unmatched API routes. */
export const notFoundHandler: RequestHandler = (req, _res, next) => {
  next(new AppError('NOT_FOUND', `No route matches ${req.method} ${req.originalUrl}`));
};

/**
 * Terminal error handler.
 *
 * Everything reaching here becomes a structured `ApiErrorBody`. Internal
 * details and stack traces are withheld in production so an error response
 * never leaks configuration or file paths.
 */
export function errorHandler(
  err: unknown,
  req: Request,
  res: Response,
  _next: NextFunction,
): void {
  const appError = toAppError(err);
  const requestId = req.requestId ?? 'unknown';

  const logLevel = appError.statusCode >= 500 ? 'error' : 'warn';
  logger[logLevel](
    {
      requestId,
      code: appError.code,
      status: appError.statusCode,
      message: appError.message,
      ...(appError.statusCode >= 500 ? { stack: appError.stack } : {}),
    },
    'Request failed',
  );

  // The stream may already be open with headers sent — emit a final SSE error
  // event instead of trying to write a status code.
  if (res.headersSent) {
    if (res.getHeader('content-type')?.toString().includes('text/event-stream')) {
      res.write(
        `event: error\ndata: ${JSON.stringify({
          type: 'error',
          error: appError.toBody(requestId, !config.isProduction),
        })}\n\n`,
      );
    }
    res.end();
    return;
  }

  res.status(appError.statusCode).json(appError.toBody(requestId, !config.isProduction));
}

/** Wrap an async handler so rejections reach the error middleware. */
export function asyncHandler(
  handler: (req: Request, res: Response, next: NextFunction) => Promise<unknown>,
): RequestHandler {
  return (req, res, next) => {
    handler(req, res, next).catch(next);
  };
}
