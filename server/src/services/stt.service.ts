/**
 * ElevenLabs speech-to-text (Scribe).
 *
 * Scribe returns word-level timings and a per-word log-probability, which we
 * fold into a single transcript confidence. That confidence is genuinely
 * useful downstream: a low-confidence transcript is a common cause of an
 * apparently-bad RAG answer, and surfacing it separates "we misheard you" from
 * "we could not find an answer".
 *
 * Note: an ElevenLabs key scoped only to `speech_to_text` returns 401 on
 * /v1/user, so the health check probes the STT endpoint itself rather than
 * the account endpoint.
 */
import type { TranscriptionResult, TranscriptWord } from '@voxrag/shared';
import { config } from '../config/env.js';
import { logger } from '../utils/logger.js';
import { retry, withTimeout } from '../utils/async.js';
import { errors } from '../utils/errors.js';

interface ScribeWord {
  text?: string;
  start?: number;
  end?: number;
  type?: string;
  logprob?: number;
}

interface ScribeResponse {
  text?: string;
  language_code?: string;
  language_probability?: number;
  audio_duration_secs?: number;
  words?: ScribeWord[];
  detail?: unknown;
}

export interface TranscribeInput {
  audio: Buffer;
  filename: string;
  mimeType: string;
  /** ISO-639-3 hint; omit to auto-detect. */
  languageCode?: string;
  diarize?: boolean;
}

export async function transcribe(input: TranscribeInput): Promise<TranscriptionResult> {
  if (!config.stt.enabled) {
    throw errors.stt(
      'Speech-to-text is not configured. Set ELEVENLABS_API_KEY in .env to enable voice input.',
    );
  }
  if (input.audio.length === 0) {
    throw errors.stt('The uploaded audio file is empty.');
  }

  const started = Date.now();

  const body = await retry(
    async () => {
      const form = new FormData();
      // Node's Blob/FormData produce a spec-compliant multipart body that
      // ElevenLabs accepts directly.
      form.append('file', new Blob([new Uint8Array(input.audio)], { type: input.mimeType }), input.filename);
      form.append('model_id', config.stt.model);
      form.append('diarize', String(Boolean(input.diarize)));
      // scribe_v2 supports word-level timestamps and audio event tagging.
      form.append('timestamps_granularity', 'word');
      form.append('tag_audio_events', 'false');
      if (input.languageCode) form.append('language_code', input.languageCode);

      const response = await withTimeout(
        fetch(`${config.stt.baseUrl.replace(/\/+$/, '')}/speech-to-text`, {
          method: 'POST',
          headers: { 'xi-api-key': config.stt.apiKey, Accept: 'application/json' },
          body: form,
        }),
        config.stt.timeoutMs,
        'ElevenLabs transcription timed out',
      );

      const payload = (await response.json().catch(() => ({}))) as ScribeResponse;

      if (!response.ok) {
        const detail = JSON.stringify(payload.detail ?? payload).slice(0, 400);
        const error = errors.stt(
          response.status === 401
            ? 'ElevenLabs rejected the API key. Check ELEVENLABS_API_KEY and that it has the speech_to_text scope.'
            : `ElevenLabs returned HTTP ${response.status}: ${detail}`,
          { status: response.status },
        );
        (error as unknown as { status: number }).status = response.status;
        throw error;
      }

      return payload;
    },
    {
      retries: 2,
      baseDelayMs: 700,
      shouldRetry: (error) => {
        const status = (error as { status?: number }).status;
        return status === undefined || status === 429 || status >= 500;
      },
      onRetry: (error, attempt, delayMs) =>
        logger.warn({ attempt, delayMs, error: (error as Error).message }, 'Retrying ElevenLabs transcription'),
    },
  );

  const words: TranscriptWord[] = (body.words ?? []).map((word) => ({
    text: word.text ?? '',
    start: word.start ?? 0,
    end: word.end ?? 0,
    type: word.type ?? 'word',
    logprob: typeof word.logprob === 'number' ? word.logprob : null,
  }));

  const text = (body.text ?? '').trim();

  return {
    text,
    languageCode: body.language_code ?? null,
    languageProbability: body.language_probability ?? null,
    durationSeconds: body.audio_duration_secs ?? null,
    words,
    confidence: computeConfidence(words, body.language_probability),
    provider: 'elevenlabs',
    model: config.stt.model,
    latencyMs: Date.now() - started,
  };
}

/**
 * Transcript confidence from per-word log-probabilities.
 *
 * Scribe reports a natural-log probability per word. Averaging over spoken
 * words (ignoring `audio_event` and `spacing` entries, which carry no speech)
 * and exponentiating gives the model's mean per-word probability. We blend
 * that with the language-detection probability, since a confidently
 * transcribed sentence in a misidentified language is still likely wrong.
 */
function computeConfidence(words: readonly TranscriptWord[], languageProbability?: number): number {
  const spoken = words.filter((word) => word.type === 'word' && word.logprob !== null);

  if (spoken.length === 0) {
    // No usable per-word signal — fall back to language probability alone.
    return languageProbability !== undefined ? clamp01(languageProbability) : 0;
  }

  const meanLogprob =
    spoken.reduce((sum, word) => sum + (word.logprob as number), 0) / spoken.length;
  const wordConfidence = clamp01(Math.exp(meanLogprob));

  if (languageProbability === undefined) return wordConfidence;
  return clamp01(0.75 * wordConfidence + 0.25 * clamp01(languageProbability));
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

/**
 * Health probe.
 *
 * Sends a fraction of a second of silence: enough for the API to accept and
 * answer the request, small enough to be effectively free.
 */
export async function sttHealthCheck(): Promise<{ ok: boolean; detail: string }> {
  if (!config.stt.enabled) {
    return { ok: false, detail: 'ELEVENLABS_API_KEY not set — voice input disabled' };
  }
  try {
    const result = await withTimeout(
      transcribe({
        audio: silentWav(0.25),
        filename: 'health.wav',
        mimeType: 'audio/wav',
      }),
      15_000,
      'health check timed out',
    );
    return { ok: true, detail: `${result.model} · ${result.latencyMs}ms` };
  } catch (error) {
    return { ok: false, detail: (error as Error).message };
  }
}

/** Minimal 16kHz mono PCM WAV of silence. */
function silentWav(seconds: number): Buffer {
  const sampleRate = 16_000;
  const samples = Math.floor(sampleRate * seconds);
  const data = Buffer.alloc(samples * 2);
  const header = Buffer.alloc(44);

  header.write('RIFF', 0);
  header.writeUInt32LE(36 + data.length, 4);
  header.write('WAVE', 8);
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20); // PCM
  header.writeUInt16LE(1, 22); // mono
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write('data', 36);
  header.writeUInt32LE(data.length, 40);

  return Buffer.concat([header, data]);
}
