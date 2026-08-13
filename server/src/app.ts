/** Express application assembly. */
import express, { type Express } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { config } from './config/env.js';
import { logger } from './utils/logger.js';
import {
  errorHandler,
  globalRateLimit,
  notFoundHandler,
  requestContext,
} from './middleware/index.js';
import { apiRouter } from './routes/index.js';

export function createApp(): Express {
  const app = express();

  // Behind Railway/Render/Fly the client IP arrives in X-Forwarded-For, and
  // rate limiting is per-IP — without this every request looks like the proxy.
  app.set('trust proxy', 1);
  app.disable('x-powered-by');

  app.use(
    helmet({
      // The API serves JSON and SSE only; a restrictive default CSP is right,
      // and the frontend is served separately by Vite/Vercel.
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'none'"],
          frameAncestors: ["'none'"],
          baseUri: ["'none'"],
        },
      },
      crossOriginResourcePolicy: { policy: 'cross-origin' },
      // Long-lived SSE connections must not be buffered or sniffed.
      noSniff: true,
    }),
  );

  app.use(
    cors({
      origin(origin, callback) {
        // Non-browser clients (curl, server-to-server) send no Origin.
        if (!origin) return callback(null, true);
        if (config.corsOrigins.includes(origin) || config.corsOrigins.includes('*')) {
          return callback(null, true);
        }
        // In development, accept any localhost port. Vite silently moves to
        // 5174+ when 5173 is taken, and a CORS failure there looks like the
        // API is down rather than a port mismatch.
        if (!config.isProduction && /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/.test(origin)) {
          return callback(null, true);
        }
        logger.warn({ origin }, 'Blocked by CORS');
        return callback(new Error(`Origin ${origin} is not allowed by CORS`));
      },
      credentials: true,
      exposedHeaders: ['x-request-id', 'x-tts-provider', 'x-tts-model', 'x-tts-latency-ms'],
      maxAge: 86_400,
    }),
  );

  app.use(requestContext);

  // Multipart bodies are handled by multer on their own routes; these limits
  // apply to JSON and urlencoded payloads only.
  app.use(express.json({ limit: '1mb' }));
  app.use(express.urlencoded({ extended: true, limit: '1mb' }));

  app.use('/api', globalRateLimit, apiRouter);

  app.get('/', (_req, res) => {
    res.json({
      name: 'VoxRAG API',
      version: config.version,
      docs: '/api/health',
      endpoints: [
        'POST /api/query',
        'POST /api/transcribe',
        'POST /api/voice-query',
        'GET  /api/benchmark',
        'GET  /api/health',
        'GET  /api/stats',
      ],
    });
  });

  app.use(notFoundHandler);
  app.use(errorHandler);

  return app;
}
