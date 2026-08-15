/**
 * Structured logging.
 *
 * Pretty, colourised output in development; newline-delimited JSON in
 * production so it can be shipped straight to a log aggregator.
 */
import pino, { type Logger } from 'pino';
import { config } from '../config/env.js';

const redactPaths = [
  'req.headers.authorization',
  'req.headers["xi-api-key"]',
  'req.headers.cookie',
  'apiKey',
  '*.apiKey',
  'GROQ_API_KEY',
  'ELEVENLABS_API_KEY',
  'SARVAM_API_KEY',
  'QDRANT_API_KEY',
];

export const logger: Logger = pino({
  level: config.logLevel,
  redact: { paths: redactPaths, censor: '[redacted]' },
  base: { service: 'goarag-server' },
  timestamp: pino.stdTimeFunctions.isoTime,
  ...(config.isProduction
    ? {}
    : {
        transport: {
          target: 'pino-pretty',
          options: {
            colorize: true,
            translateTime: 'HH:MM:ss.l',
            ignore: 'pid,hostname,service',
            singleLine: false,
          },
        },
      }),
});

/** Child logger bound to a request id, so one request's lines can be grepped. */
export function requestLogger(requestId: string): Logger {
  return logger.child({ requestId });
}

/**
 * Log a completed pipeline stage. Kept as one helper so every stage emits the
 * same shape and the analytics store can be rebuilt from logs alone.
 */
export function logStage(
  log: Logger,
  stage: string,
  durationMs: number,
  detail: Record<string, unknown> = {},
): void {
  log.debug({ stage, durationMs: Math.round(durationMs), ...detail }, `stage:${stage}`);
}
