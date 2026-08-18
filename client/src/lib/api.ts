/**
 * API client.
 *
 * Two shapes: plain JSON requests, and Server-Sent Event streams for the query
 * endpoints. The SSE reader is hand-rolled rather than using `EventSource`
 * because EventSource cannot issue POST requests or send a body.
 */
import type {
  BenchmarkResult,
  ConversationTurn,
  DatasetSuggestionsResponse,
  HealthResponse,
  QueryResult,
  RetrievalOptions,
  StatsResponse,
  StreamEvent,
  TranscriptionResult,
  ApiErrorBody,
} from '@goarag/shared';

/**
 * API origin.
 *
 * Always empty in development: Vite proxies `/api` to the server, which keeps
 * requests same-origin and avoids CORS entirely — including when Vite picks a
 * different port because 5173 was taken. Honouring VITE_API_BASE_URL in dev
 * would send the browser cross-origin and defeat that proxy.
 *
 * In a production build the variable is required, since the frontend is served
 * from a different host than the API.
 */
const BASE_URL = import.meta.env.DEV
  ? ''
  : ((import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/+$/, '') ?? '');

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly requestId: string | null;
  readonly details: unknown;

  constructor(body: Partial<ApiErrorBody>, status: number) {
    super(body.message ?? `Request failed with status ${status}`);
    this.name = 'ApiError';
    this.code = body.code ?? 'INTERNAL_ERROR';
    this.status = status;
    this.requestId = body.requestId ?? null;
    this.details = body.details;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as Partial<ApiErrorBody>;
    throw new ApiError(body, response.status);
  }

  return (await response.json()) as T;
}

// ─── Reads ───────────────────────────────────────────────────────────────────

export function fetchHealth(deep = false): Promise<HealthResponse> {
  return request<HealthResponse>(`/api/health${deep ? '?deep=true' : ''}`);
}

export function fetchStats(recentLimit = 25): Promise<StatsResponse> {
  return request<StatsResponse>(`/api/stats?recentLimit=${recentLimit}`);
}

export function fetchSuggestions(count = 2): Promise<DatasetSuggestionsResponse> {
  return request<DatasetSuggestionsResponse>(`/api/dataset/suggestions?count=${count}`);
}

export function runBenchmark(params: {
  sampleSize: number;
  generation: boolean;
  language?: string;
  concurrency?: number;
}): Promise<BenchmarkResult> {
  const query = new URLSearchParams({
    sampleSize: String(params.sampleSize),
    generation: String(params.generation),
    concurrency: String(params.concurrency ?? 2),
  });
  if (params.language) query.set('language', params.language);
  return request<BenchmarkResult>(`/api/benchmark?${query.toString()}`);
}

// ─── Transcription ───────────────────────────────────────────────────────────

export function transcribeAudio(audio: Blob, languageCode?: string): Promise<TranscriptionResult> {
  const form = new FormData();
  form.append('file', audio, filenameFor(audio));
  if (languageCode) form.append('languageCode', languageCode);
  return request<TranscriptionResult>('/api/transcribe', { method: 'POST', body: form });
}

/** ElevenLabs infers the container from the extension, so it must match. */
function filenameFor(audio: Blob): string {
  const type = audio.type.split(';')[0] ?? '';
  const extension =
    type.includes('webm') ? 'webm'
    : type.includes('ogg') ? 'ogg'
    : type.includes('mp4') || type.includes('m4a') ? 'mp4'
    : type.includes('wav') ? 'wav'
    : type.includes('mpeg') || type.includes('mp3') ? 'mp3'
    : 'webm';
  return `recording.${extension}`;
}

// ─── Speech synthesis ────────────────────────────────────────────────────────

/**
 * Synthesise speech server-side (Sarvam AI, falling back to ElevenLabs),
 * returning a playable object URL.
 *
 * Pass `languageCode` — the code the STT reported for the turn — or the answer
 * is spoken with the server's default language's phonetics.
 *
 * Returns `null` when the account cannot use TTS (HTTP 402 on a free plan) so
 * the caller can fall back to the browser's own synthesiser instead of showing
 * an error for something the user cannot fix.
 */
export async function speak(
  text: string,
  voiceId?: string,
  languageCode?: string,
): Promise<string | null> {
  const response = await fetch(`${BASE_URL}/api/speak`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'audio/mpeg' },
    body: JSON.stringify({
      text,
      ...(voiceId ? { voiceId } : {}),
      ...(languageCode ? { languageCode } : {}),
    }),
  });

  if (!response.ok) {
    if (response.status === 402 || response.status === 401) return null;
    const body = (await response.json().catch(() => ({}))) as Partial<ApiErrorBody>;
    throw new ApiError(body, response.status);
  }

  return URL.createObjectURL(await response.blob());
}

// ─── Non-streaming query ─────────────────────────────────────────────────────

export function postQuery(input: {
  query: string;
  history?: ConversationTurn[];
  options?: RetrievalOptions;
}): Promise<QueryResult> {
  return request<QueryResult>('/api/query', {
    method: 'POST',
    body: JSON.stringify({
      query: input.query,
      history: input.history ?? [],
      options: input.options ?? {},
      stream: false,
    }),
  });
}

// ─── Streaming ───────────────────────────────────────────────────────────────

export interface StreamHandlers {
  onEvent: (event: StreamEvent) => void;
  signal?: AbortSignal;
}

/**
 * Read an SSE response body, dispatching each parsed event.
 *
 * Frames are separated by a blank line and may be split across network
 * chunks, so we buffer until a complete frame is available.
 */
async function readEventStream(response: Response, handlers: StreamHandlers): Promise<void> {
  if (!response.body) throw new Error('The response carried no body to stream');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      let separator = buffer.indexOf('\n\n');
      while (separator !== -1) {
        const frame = buffer.slice(0, separator);
        buffer = buffer.slice(separator + 2);

        // A frame is `event: <name>` plus one or more `data:` lines.
        const dataLines = frame
          .split('\n')
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trimStart());

        if (dataLines.length > 0) {
          try {
            handlers.onEvent(JSON.parse(dataLines.join('\n')) as StreamEvent);
          } catch {
            // A malformed frame should not tear down a working stream.
          }
        }
        separator = buffer.indexOf('\n\n');
      }
    }
  } finally {
    reader.releaseLock();
  }
}

async function openStream(path: string, body: BodyInit, handlers: StreamHandlers, isForm: boolean) {
  const response = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers: {
      Accept: 'text/event-stream',
      ...(isForm ? {} : { 'Content-Type': 'application/json' }),
    },
    body,
    ...(handlers.signal ? { signal: handlers.signal } : {}),
  });

  if (!response.ok) {
    const errorBody = (await response.json().catch(() => ({}))) as Partial<ApiErrorBody>;
    throw new ApiError(errorBody, response.status);
  }

  await readEventStream(response, handlers);
}

/** Stream a text query. */
export function streamQuery(
  input: { query: string; history?: ConversationTurn[]; options?: RetrievalOptions },
  handlers: StreamHandlers,
): Promise<void> {
  return openStream(
    '/api/query',
    JSON.stringify({
      query: input.query,
      history: input.history ?? [],
      options: input.options ?? {},
      stream: true,
    }),
    handlers,
    false,
  );
}

/** Stream a voice query: audio in, transcript then answer out. */
export function streamVoiceQuery(
  input: {
    audio: Blob;
    history?: ConversationTurn[];
    options?: RetrievalOptions;
    languageCode?: string;
  },
  handlers: StreamHandlers,
): Promise<void> {
  const form = new FormData();
  form.append('file', input.audio, filenameFor(input.audio));
  form.append('stream', 'true');
  form.append('history', JSON.stringify(input.history ?? []));
  form.append('options', JSON.stringify(input.options ?? {}));
  if (input.languageCode) form.append('languageCode', input.languageCode);

  return openStream('/api/voice-query', form, handlers, true);
}
