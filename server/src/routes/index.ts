/**
 * API routes.
 *
 * Every route is: rate limit → validate → controller. Endpoints that call a
 * paid upstream (transcription, generation) get a tighter limiter than reads.
 */
import { Router } from 'express';
import {
  benchmarkQuerySchema,
  queryRequestSchema,
  speakRequestSchema,
  statsQuerySchema,
} from '@voxrag/shared';
import {
  asyncHandler,
  audioUpload,
  handleUploadErrors,
  queryRateLimit,
  transcribeRateLimit,
  validateBody,
  validateQuery,
} from '../middleware/index.js';
import {
  getBenchmark,
  getHealth,
  getStats,
  postQuery,
  postSpeak,
  postTranscribe,
  postVoiceQuery,
  postTTS,
} from '../controllers/index.js';

export const apiRouter: Router = Router();

// ── reads ────────────────────────────────────────────────────────────────────
apiRouter.get('/health', asyncHandler(getHealth));
apiRouter.get('/stats', validateQuery(statsQuerySchema), asyncHandler(getStats));

// A benchmark run is expensive; the shared query limiter keeps it from being
// used as an amplification vector.
apiRouter.get('/benchmark', queryRateLimit, validateQuery(benchmarkQuerySchema), asyncHandler(getBenchmark));

// ── text query ───────────────────────────────────────────────────────────────
apiRouter.post('/query', queryRateLimit, validateBody(queryRequestSchema), asyncHandler(postQuery));

// ── voice ────────────────────────────────────────────────────────────────────
// `handleUploadErrors` sits directly after the multer middleware so size and
// MIME rejections become structured errors rather than raw multer failures.
apiRouter.post(
  '/transcribe',
  transcribeRateLimit,
  audioUpload.single('file'),
  handleUploadErrors,
  asyncHandler(postTranscribe),
);

apiRouter.post(
  '/voice-query',
  transcribeRateLimit,
  audioUpload.single('file'),
  handleUploadErrors,
  asyncHandler(postVoiceQuery),
);

// Speech synthesis is billed per character, so it shares the tighter limiter.
// `validateBody` turns a bad payload into a 400 VALIDATION_ERROR; parsing inside
// the controller instead surfaces it as a 500 with a raw Zod dump.
apiRouter.post('/speak', transcribeRateLimit, validateBody(speakRequestSchema), asyncHandler(postSpeak));
apiRouter.post('/tts', transcribeRateLimit, validateBody(speakRequestSchema), asyncHandler(postTTS));
