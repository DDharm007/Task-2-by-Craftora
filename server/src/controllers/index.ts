/**
 * HTTP controllers.
 *
 * Thin by design: parse the validated request, call a service, shape the
 * response. All pipeline logic lives in `services/` and `rag/`.
 */
import type { Request, Response } from 'express';
import type {
  BenchmarkQuery,
  HealthResponse,
  QueryRequest,
  SpeakRequest,
  StatsQuery,
  StreamEvent,
  ComponentHealth,
} from '@voxrag/shared';
import {
  transcribeOptionsSchema,
  voiceQueryOptionsSchema,
} from '@voxrag/shared';
import { config } from '../config/env.js';
import { AppError, toAppError } from '../utils/errors.js';
import { logger } from '../utils/logger.js';
import { validatedQuery } from '../middleware/index.js';
import { runQuery, streamQuery } from '../services/rag.service.js';
import { synthesize } from '../services/tts.service.js';
import { transcribe, sttHealthCheck } from '../services/stt.service.js';
import { llmHealthCheck } from '../services/llm.service.js';
import { getIndexStats } from '../services/indexing.service.js';
import { getAnalytics, uptimeSeconds, recordFailure } from '../services/analytics.service.js';
import { runBenchmark } from '../services/benchmark.service.js';
import { getVectorStore } from '../rag/vector/index.js';
import { getEmbeddingProvider } from '../rag/embeddings/index.js';

// ─── Server-Sent Events ──────────────────────────────────────────────────────

/** Open an SSE stream with the headers proxies need to not buffer it. */
function openEventStream(res: Response): void {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    // Nginx buffers event streams by default, which defeats the point.
    'X-Accel-Buffering': 'no',
  });
  res.flushHeaders?.();
}

function sendEvent(res: Response, event: StreamEvent): void {
  res.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
}

/**
 * Drive a stream generator to completion over SSE.
 *
 * The abort controller is wired to the response so that when a user navigates
 * away mid-answer we stop paying for the generation instead of streaming into
 * a closed socket.
 */
async function pipeStream(
  req: Request,
  res: Response,
  makeStream: (signal: AbortSignal) => AsyncGenerator<StreamEvent, void, unknown>,
): Promise<void> {
  const controller = new AbortController();
  let closed = false;

  // Listen on the *response*, not the request. Node emits `close` on the
  // request once its body has been fully consumed — which for a multipart
  // upload is the moment multer finishes reading the file, long before we
  // have streamed anything. Watching that would abort every voice query
  // immediately. The response only closes when the socket does, so
  // `writableEnded === false` there is a genuine client disconnect.
  res.on('close', () => {
    if (!res.writableEnded) {
      closed = true;
      controller.abort();
    }
  });

  openEventStream(res);

  try {
    for await (const event of makeStream(controller.signal)) {
      if (closed) break;
      sendEvent(res, event);
    }
  } catch (error) {
    if (!closed) {
      const appError = toAppError(error);
      logger.error(
        { requestId: req.requestId, code: appError.code, message: appError.message },
        'Stream failed',
      );
      recordFailure();
      sendEvent(res, {
        type: 'error',
        error: appError.toBody(req.requestId, !config.isProduction),
      });
    }
  } finally {
    if (!res.writableEnded) res.end();
  }
}

// ─── POST /api/query ─────────────────────────────────────────────────────────

export async function postQuery(req: Request, res: Response): Promise<void> {
  const body = req.body as QueryRequest;

  if (body.stream) {
    await pipeStream(req, res, (signal) =>
      streamQuery({
        query: body.query,
        history: body.history,
        options: body.options,
        signal,
        requestId: req.requestId,
      }),
    );
    return;
  }

  const result = await runQuery({
    query: body.query,
    history: body.history,
    options: body.options,
    requestId: req.requestId,
  });
  res.json(result);
}

// ─── POST /api/transcribe ────────────────────────────────────────────────────

export async function postTranscribe(req: Request, res: Response): Promise<void> {
  const file = req.file;
  if (!file) {
    throw new AppError('VALIDATION_ERROR', 'No audio file was uploaded. Send it as the `file` field.');
  }

  const options = transcribeOptionsSchema.parse(req.body ?? {});

  const result = await transcribe({
    audio: file.buffer,
    filename: file.originalname || 'recording.webm',
    mimeType: file.mimetype,
    ...(options.languageCode ? { languageCode: options.languageCode } : {}),
    diarize: options.diarize,
  });

  res.json(result);
}

// ─── POST /api/voice-query ───────────────────────────────────────────────────

/**
 * Voice → transcript → RAG in one round trip.
 *
 * Streaming is the interesting path: the transcript is emitted the moment STT
 * returns, so the UI shows what was heard while retrieval is still running.
 */
export async function postVoiceQuery(req: Request, res: Response): Promise<void> {
  const file = req.file;
  if (!file) {
    throw new AppError('VALIDATION_ERROR', 'No audio file was uploaded. Send it as the `file` field.');
  }

  const options = voiceQueryOptionsSchema.parse(req.body ?? {});

  if (options.stream) {
    await pipeStream(req, res, (signal) =>
      (async function* () {
        const transcription = await transcribe({
          audio: file.buffer,
          filename: file.originalname || 'recording.webm',
          mimeType: file.mimetype,
          ...(options.languageCode ? { languageCode: options.languageCode } : {}),
          diarize: options.diarize,
        });

        if (!transcription.text.trim()) {
          throw new AppError(
            'STT_FAILED',
            'No speech was detected in the recording. Try again and speak clearly.',
          );
        }

        yield* streamQuery({
          query: transcription.text,
          history: options.history,
          options: options.options,
          transcription,
          signal,
          requestId: req.requestId,
        });
      })(),
    );
    return;
  }

  const transcription = await transcribe({
    audio: file.buffer,
    filename: file.originalname || 'recording.webm',
    mimeType: file.mimetype,
    ...(options.languageCode ? { languageCode: options.languageCode } : {}),
    diarize: options.diarize,
  });

  if (!transcription.text.trim()) {
    throw new AppError(
      'STT_FAILED',
      'No speech was detected in the recording. Try again and speak clearly.',
    );
  }

  const result = await runQuery({
    query: transcription.text,
    history: options.history,
    options: options.options,
    transcription,
    requestId: req.requestId,
  });
  res.json(result);
}

// ─── POST /api/speak ─────────────────────────────────────────────────────────

/**
 * Read an answer aloud with ElevenLabs, completing the voice loop.
 *
 * Returns raw audio rather than base64 JSON so the browser can stream it
 * straight into an <audio> element without buffering the whole clip first.
 */
export async function postSpeak(req: Request, res: Response): Promise<void> {
  // Already validated by `validateBody(speakRequestSchema)` on the route.
  const body = req.body as SpeakRequest;

  const result = await synthesize({
    text: body.text,
    ...(body.voiceId ? { voiceId: body.voiceId } : {}),
    ...(body.languageCode ? { languageCode: body.languageCode } : {}),
  });

  res.setHeader('Content-Type', result.contentType);
  res.setHeader('Content-Length', String(result.audio.length));
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-TTS-Provider', result.provider);
  res.setHeader('X-TTS-Model', result.model);
  res.setHeader('X-TTS-Latency-Ms', String(result.latencyMs));
  res.send(result.audio);
}

// ─── GET /api/health ─────────────────────────────────────────────────────────

/**
 * Health check.
 *
 * `?deep=true` probes the upstream providers, which costs real API calls.
 * The default is a cheap local check suitable for a load-balancer probe.
 */
export async function getHealth(req: Request, res: Response): Promise<void> {
  const deep = req.query.deep === 'true' || req.query.deep === '1';
  const components: ComponentHealth[] = [];

  const measure = async (
    name: string,
    probe: () => Promise<{ ok: boolean; detail: string }>,
  ): Promise<void> => {
    const started = Date.now();
    try {
      const outcome = await probe();
      components.push({
        name,
        status: outcome.ok ? 'ok' : 'down',
        detail: outcome.detail,
        latencyMs: Date.now() - started,
      });
    } catch (error) {
      components.push({
        name,
        status: 'down',
        detail: (error as Error).message,
        latencyMs: Date.now() - started,
      });
    }
  };

  await measure('vector_store', async () => {
    const store = await getVectorStore();
    return store.healthCheck();
  });

  const stats = await getIndexStats().catch(() => null);
  components.push({
    name: 'index',
    status: stats?.indexed ? 'ok' : 'degraded',
    detail: stats?.indexed
      ? `${stats.vectors.toLocaleString()} vectors · ${stats.documents.toLocaleString()} documents`
      : 'empty — run `npm run index`',
    latencyMs: null,
  });

  components.push({
    name: 'embeddings',
    status: 'ok',
    detail: `${getEmbeddingProvider().name} · ${getEmbeddingProvider().model}`,
    latencyMs: null,
  });

  components.push({
    name: 'speech_to_text',
    status: config.stt.enabled ? 'ok' : 'degraded',
    detail: config.stt.enabled
      ? `elevenlabs · ${config.stt.model}`
      : 'ELEVENLABS_API_KEY not set — voice input disabled',
    latencyMs: null,
  });

  if (deep) {
    await measure('llm', llmHealthCheck);
    if (config.stt.enabled) await measure('speech_to_text_live', sttHealthCheck);
  } else {
    components.push({
      name: 'llm',
      status: 'ok',
      detail: `${config.llm.model} (not probed — add ?deep=true)`,
      latencyMs: null,
    });
  }

  const down = components.filter((component) => component.status === 'down');
  const degraded = components.filter((component) => component.status === 'degraded');

  const body: HealthResponse = {
    status: down.length > 0 ? 'down' : degraded.length > 0 ? 'degraded' : 'ok',
    version: config.version,
    uptimeSeconds: uptimeSeconds(),
    timestamp: new Date().toISOString(),
    components,
    models: {
      llm: config.llm.model,
      embedding: getEmbeddingProvider().model,
    },
  };

  // 503 on hard failure so orchestrators can act on it; degraded is still 200
  // because the service can serve traffic.
  res.status(body.status === 'down' ? 503 : 200).json(body);
}

// ─── GET /api/stats ──────────────────────────────────────────────────────────

export async function getStats(req: Request, res: Response): Promise<void> {
  const query = validatedQuery<StatsQuery>(req);
  const index = await getIndexStats();
  res.json({ index, analytics: getAnalytics(query.recentLimit) });
}

// ─── GET /api/benchmark ──────────────────────────────────────────────────────

export async function getBenchmark(req: Request, res: Response): Promise<void> {
  const query = validatedQuery<BenchmarkQuery>(req);
  const result = await runBenchmark({
    sampleSize: query.sampleSize,
    generation: query.generation,
    concurrency: query.concurrency,
    ...(query.language ? { language: query.language } : {}),
  });
  res.json(result);
}

// ─── POST /api/tts ───────────────────────────────────────────────────────────

/**
 * Alias of `POST /api/speak`, kept so either path works.
 *
 * It delegates rather than duplicating: `synthesize()` resolves to a
 * `SynthesisResult` (audio plus metadata), not a bare Buffer, and it is also
 * where Zod validation and the 402 free-plan handling live.
 */
export const postTTS = postSpeak;
