/**
 * ElevenLabs text-to-speech.
 *
 * Used to read grounded answers back to the user, closing the voice loop:
 * speak a question, hear the answer.
 *
 * `eleven_multilingual_v2` is the default model because the corpus is
 * multilingual — an answer generated in Hindi has to be spoken in Hindi, and a
 * monolingual English voice mangles Devanagari.
 *
 * Note: TTS requires a paid ElevenLabs plan. On a free key the API returns
 * HTTP 402, which we surface as a typed, non-retryable error so the client can
 * fall back to the browser's own speech synthesiser rather than failing loudly.
 */
import { createHash } from 'node:crypto';
import pLimit from 'p-limit';
import { config } from '../config/env.js';
import { logger } from '../utils/logger.js';
import { withTimeout } from '../utils/async.js';
import { AppError, errors } from '../utils/errors.js';
import { concatWav } from '../utils/wav.js';

export interface SynthesisResult {
  audio: Buffer;
  contentType: string;
  model: string;
  voiceId: string;
  latencyMs: number;
  provider: 'sarvam' | 'elevenlabs';
}

/** Longest text we will synthesise, to bound cost and latency. */
const MAX_CHARACTERS = 2_500;

/**
 * Sarvam rejects any single input longer than 500 characters, and its latency
 * grows with input length. Splitting below the hard cap lets long answers be
 * synthesised as several clips at once instead of one slow serial request —
 * and, before that, they were simply truncated at the cap and never spoken.
 */
const SARVAM_CHUNK_CHARACTERS = 480;

/** Clips synthesised at once. Enough to cover a long answer in one wave. */
const SARVAM_CONCURRENCY = 6;

/**
 * Split text into synthesis-sized clips, preferring sentence boundaries.
 *
 * Cutting mid-sentence is audible: the two clips are generated independently,
 * so the prosody does not carry across the join. Sentence ends are where a
 * speaker would pause anyway, which makes the seam disappear.
 */
export function chunkForSpeech(text: string, limit = SARVAM_CHUNK_CHARACTERS): string[] {
  if (text.length <= limit) return text ? [text] : [];

  // Keep the delimiter with the sentence it ends. `।` is the Devanagari danda.
  const sentences = text.match(/[^.!?।]+[.!?।]+\s*|[^.!?।]+$/g) ?? [text];
  const chunks: string[] = [];
  let current = '';

  const flush = () => {
    const trimmed = current.trim();
    if (trimmed) chunks.push(trimmed);
    current = '';
  };

  for (const sentence of sentences) {
    if (current && current.length + sentence.length > limit) flush();

    if (sentence.length > limit) {
      // A single sentence over the cap: break it on whitespace instead.
      flush();
      let rest = sentence.trim();
      while (rest.length > limit) {
        const cut = rest.lastIndexOf(' ', limit);
        const at = cut > limit * 0.5 ? cut : limit;
        chunks.push(rest.slice(0, at).trim());
        rest = rest.slice(at).trim();
      }
      current = rest;
      continue;
    }

    current += sentence;
  }
  flush();

  return chunks.filter(Boolean);
}

// ── Response cache ───────────────────────────────────────────────────────────

/**
 * Replaying an answer, or re-asking the same question, is common enough that
 * re-synthesising is pure waste — a cache hit turns a multi-second call into a
 * memory read. Bounded by total bytes because clips are large (a 2,500-char
 * answer is roughly 2 MB of PCM) and an entry-count cap would not bound memory.
 */
const CACHE_MAX_BYTES = 48 * 1024 * 1024;
const audioCache = new Map<string, SynthesisResult>();
let cacheBytes = 0;

function cacheKey(parts: {
  provider: string;
  model: string;
  voice: string;
  language: string;
  text: string;
}): string {
  return createHash('sha1')
    .update(`${parts.provider}|${parts.model}|${parts.voice}|${parts.language}|${parts.text}`)
    .digest('hex');
}

function cacheGet(key: string): SynthesisResult | undefined {
  const hit = audioCache.get(key);
  if (!hit) return undefined;
  // Re-insert so iteration order stays least-recently-used first.
  audioCache.delete(key);
  audioCache.set(key, hit);
  return hit;
}

function cacheSet(key: string, value: SynthesisResult): void {
  if (value.audio.length > CACHE_MAX_BYTES) return;

  audioCache.set(key, value);
  cacheBytes += value.audio.length;

  for (const [oldest, entry] of audioCache) {
    if (cacheBytes <= CACHE_MAX_BYTES) break;
    if (oldest === key) continue;
    audioCache.delete(oldest);
    cacheBytes -= entry.audio.length;
  }
}

/**
 * Speakers Sarvam accepts, per model.
 *
 * The API validates the speaker against the model and returns 400 on a
 * mismatch, so a `bulbul:v2` name silently carried onto `bulbul:v3` kills every
 * request. Resolving against this table first keeps that failure out of the
 * request path.
 */
const SARVAM_SPEAKERS: Record<string, ReadonlySet<string>> = {
  'bulbul:v3': new Set([
    'aditya', 'ritu', 'ashutosh', 'priya', 'neha', 'rahul', 'pooja', 'rohan',
    'simran', 'kavya', 'amit', 'dev', 'ishita', 'shreya', 'ratan', 'varun',
    'manan', 'sumit', 'roopa', 'kabir', 'aayan', 'shubh', 'advait', 'anand',
    'tanya', 'tarun', 'sunny', 'mani', 'gokul', 'vijay', 'shruti', 'suhani',
    'mohit', 'kavitha', 'rehan', 'soham', 'rupali', 'niharika',
  ]),
  'bulbul:v2': new Set([
    'anushka', 'manisha', 'vidya', 'arya', 'abhilash', 'karun', 'hitesh',
  ]),
};

/**
 * Nearest equivalent speaker across models, matched on gender and timbre.
 *
 * The two model generations share no speaker names, so switching
 * `SARVAM_TTS_MODEL` would otherwise invalidate a configured or caller-supplied
 * voice. Translating keeps the voice roughly the same either way.
 */
const V2_TO_V3_SPEAKER: Record<string, string> = {
  anushka: 'priya',
  manisha: 'neha',
  vidya: 'ritu',
  arya: 'shreya',
  abhilash: 'aditya',
  karun: 'rahul',
  hitesh: 'amit',
};

const V3_TO_V2_SPEAKER: Record<string, string> = Object.fromEntries(
  Object.entries(V2_TO_V3_SPEAKER).map(([v2, v3]) => [v3, v2]),
);

/** Speaker used when nothing valid was supplied, per model. */
const DEFAULT_SPEAKER: Record<string, string> = {
  'bulbul:v2': 'anushka',
  'bulbul:v3': 'priya',
};

/** Languages Sarvam can speak. Anything else has to go to ElevenLabs. */
const SARVAM_LANGUAGES = new Set([
  'as-IN', 'bn-IN', 'brx-IN', 'doi-IN', 'en-IN', 'gu-IN', 'hi-IN', 'kn-IN',
  'kok-IN', 'ks-IN', 'mai-IN', 'ml-IN', 'mni-IN', 'mr-IN', 'ne-IN', 'od-IN',
  'pa-IN', 'sa-IN', 'sat-IN', 'sd-IN', 'ta-IN', 'te-IN', 'ur-IN',
]);

/** ISO-639-3 (what the STT returns) and bare ISO-639-1 → Sarvam language tag. */
const LANGUAGE_TO_SARVAM: Record<string, string> = {
  asm: 'as-IN', as: 'as-IN',
  ben: 'bn-IN', bn: 'bn-IN',
  eng: 'en-IN', en: 'en-IN',
  guj: 'gu-IN', gu: 'gu-IN',
  hin: 'hi-IN', hi: 'hi-IN',
  kan: 'kn-IN', kn: 'kn-IN',
  mal: 'ml-IN', ml: 'ml-IN',
  mar: 'mr-IN', mr: 'mr-IN',
  nep: 'ne-IN', ne: 'ne-IN',
  ori: 'od-IN', ory: 'od-IN', or: 'od-IN',
  pan: 'pa-IN', pa: 'pa-IN',
  san: 'sa-IN', sa: 'sa-IN',
  tam: 'ta-IN', ta: 'ta-IN',
  tel: 'te-IN', te: 'te-IN',
  urd: 'ur-IN', ur: 'ur-IN',
};

/**
 * Resolve a caller language to a Sarvam tag, or `null` when Sarvam cannot speak
 * it and the request should go straight to ElevenLabs.
 *
 * Accepts the shapes that actually reach us: `hin_Deva` from the STT, plain
 * `hi`/`hin`, and BCP-47 like `hi-IN`.
 */
export function resolveSarvamLanguage(languageCode?: string): string | null {
  if (!languageCode) return SARVAM_LANGUAGES.has(config.tts.sarvamLanguage) ? config.tts.sarvamLanguage : 'en-IN';

  const raw = languageCode.trim();
  // Already a Sarvam tag (case-insensitively).
  const exact = [...SARVAM_LANGUAGES].find((tag) => tag.toLowerCase() === raw.toLowerCase());
  if (exact) return exact;

  // `hin_Deva` / `hi-IN` / `hi_IN` → the language subtag before the separator.
  const base = (raw.split(/[-_]/)[0] ?? '').toLowerCase();
  return LANGUAGE_TO_SARVAM[base] ?? null;
}

/**
 * Resolve the Sarvam speaker for a request.
 *
 * `voiceId` is shared with ElevenLabs, whose ids look nothing like a Sarvam
 * speaker name — passing one through unchecked is a guaranteed 400, so an
 * unrecognised value falls back to the configured speaker instead.
 */
export function resolveSarvamSpeaker(voiceId?: string): string {
  const model = config.tts.sarvamModel;
  const valid = SARVAM_SPEAKERS[model];
  const aliases = model === 'bulbul:v2' ? V3_TO_V2_SPEAKER : V2_TO_V3_SPEAKER;

  const normalise = (name: string) => {
    const lower = name.trim().toLowerCase();
    if (!valid || valid.has(lower)) return lower;
    // A name from the other model generation still resolves to its counterpart.
    const migrated = aliases[lower];
    return migrated && valid.has(migrated) ? migrated : '';
  };

  return (
    normalise(voiceId ?? '') ||
    normalise(config.tts.sarvamSpeaker) ||
    DEFAULT_SPEAKER[model] ||
    'anushka'
  );
}

/**
 * Strip markup that should not be spoken.
 *
 * Citation markers, Markdown emphasis and code fences are all visual devices —
 * read aloud they become noise ("bracket one") that obscures the answer.
 */
export function prepareForSpeech(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/\[\d+\](\[\d+\])*/g, ' ')
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/[*_~`>#|]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, MAX_CHARACTERS);
}

export async function synthesize(input: {
  text: string;
  voiceId?: string;
  languageCode?: string;
}): Promise<SynthesisResult> {
  if (!config.tts.enabled) {
    throw new AppError(
      'TTS_FAILED',
      'Text-to-speech is not configured. Set SARVAM_API_KEY or ELEVENLABS_API_KEY in .env.',
      { statusCode: 402, retryable: false, details: { provider: 'tts', code: 'not_configured' } },
    );
  }

  const text = prepareForSpeech(input.text);
  if (!text) {
    throw new AppError('VALIDATION_ERROR', 'There is nothing to speak once markup is removed.');
  }

  // 1. Prefer Sarvam AI TTS if configured. It is skipped for languages Sarvam
  //    cannot speak — ElevenLabs is the multilingual path.
  const targetLanguage = resolveSarvamLanguage(input.languageCode);
  let sarvamFailure: string | null = null;

  if (config.tts.sarvamApiKey && targetLanguage) {
    const model = config.tts.sarvamModel;
    const speaker = resolveSarvamSpeaker(input.voiceId);
    const key = cacheKey({ provider: 'sarvam', model, voice: speaker, language: targetLanguage, text });

    const cached = cacheGet(key);
    if (cached) {
      logger.debug({ model, speaker, bytes: cached.audio.length }, 'Served speech from cache');
      return { ...cached, latencyMs: 0 };
    }

    try {
      const started = Date.now();
      const chunks = chunkForSpeech(text);

      /**
       * Synthesise one chunk.
       *
       * Sarvam splits internally and answers with an *array* of clips once the
       * input passes roughly 220 characters. Reading only `audios[0]` silently
       * dropped everything after the first clip, cutting answers off mid
       * sentence — so every entry is returned and joined in order.
       */
      const synthesizeChunk = async (chunk: string): Promise<Buffer[]> => {
        const response = await withTimeout(
          fetch('https://api.sarvam.ai/text-to-speech', {
            method: 'POST',
            headers: {
              'api-subscription-key': config.tts.sarvamApiKey,
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              text: chunk,
              target_language_code: targetLanguage,
              speaker,
              pace: 1.0,
              speech_sample_rate: config.tts.sarvamSampleRate,
              model,
            }),
          }),
          config.tts.timeoutMs,
          'Sarvam AI speech synthesis timed out',
        );

        if (!response.ok) {
          const detail = await response.text().catch(() => '');
          throw new Error(`HTTP ${response.status}: ${detail.slice(0, 300)}`);
        }
        const data = (await response.json()) as { audios?: string[] };
        if (!data.audios?.length) throw new Error('Sarvam AI returned no audio');
        return data.audios.map((clip) => Buffer.from(clip, 'base64'));
      };

      // Chunks are independent, so the wall clock is the slowest one rather
      // than their sum — the difference on a long answer is seconds.
      const limit = pLimit(SARVAM_CONCURRENCY);
      const perChunk = await Promise.all(chunks.map((chunk) => limit(() => synthesizeChunk(chunk))));

      // `Promise.all` preserves input order, so flattening keeps the answer in
      // the order it was written.
      const audio = concatWav(perChunk.flat());
      if (audio) {
        const latencyMs = Date.now() - started;
        logger.debug(
          {
            model,
            speaker,
            language: targetLanguage,
            chunks: chunks.length,
            bytes: audio.length,
            latencyMs,
          },
          'Synthesised speech via Sarvam AI',
        );
        const result: SynthesisResult = {
          audio,
          contentType: 'audio/wav',
          model,
          voiceId: speaker,
          latencyMs,
          provider: 'sarvam',
        };
        cacheSet(key, result);
        return result;
      }
      sarvamFailure = 'Sarvam AI returned audio that could not be joined';
    } catch (err) {
      sarvamFailure = (err as Error).message;
    }

    if (sarvamFailure) {
      logger.warn(
        { model, speaker, language: targetLanguage, detail: sarvamFailure },
        'Sarvam AI TTS failed, falling back to ElevenLabs',
      );
    }
  }

  // 2. ElevenLabs fallback. Without a key there is nothing left to try, so
  //    surface why Sarvam failed rather than a misleading ElevenLabs 401.
  if (!config.tts.apiKey) {
    throw new AppError(
      'TTS_FAILED',
      sarvamFailure
        ? `Sarvam AI text-to-speech failed and no ElevenLabs key is configured to fall back to. ${sarvamFailure}`
        : `Sarvam AI cannot speak language "${input.languageCode}" and no ELEVENLABS_API_KEY is set for the multilingual fallback.`,
      { statusCode: 502, retryable: false, details: { provider: 'sarvam', code: 'no_fallback' } },
    );
  }

  const voiceId = input.voiceId || config.tts.voiceId;
  const started = Date.now();

  // `output_format` is a query parameter — sent in the body it is ignored and
  // the account default applies instead.
  const url = new URL(
    `${config.tts.baseUrl.replace(/\/+$/, '')}/text-to-speech/${encodeURIComponent(voiceId)}`,
  );
  url.searchParams.set('output_format', 'mp3_44100_128');

  const response = await withTimeout(
    fetch(url, {
      method: 'POST',
      headers: {
        'xi-api-key': config.tts.apiKey,
        'Content-Type': 'application/json',
        Accept: 'audio/mpeg',
      },
      body: JSON.stringify({
        text,
        model_id: config.tts.model,
        voice_settings: { stability: 0.5, similarity_boost: 0.75, style: 0.0, use_speaker_boost: true },
      }),
    }),
    config.tts.timeoutMs,
    'ElevenLabs speech synthesis timed out',
  );

  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    let detailJson: Record<string, unknown> = {};
    try { detailJson = JSON.parse(detail); } catch { /* not JSON */ }

    if (response.status === 402) {
      // Free-tier accounts cannot use TTS. This is a permanent condition for
      // this key — return HTTP 402 so the client can fall back to the browser
      // speech synthesiser rather than showing an error the user cannot fix.
      throw new AppError(
        'TTS_FAILED',
        'ElevenLabs text-to-speech requires a paid plan on this account. Using browser voice instead.',
        {
          statusCode: 402,
          retryable: false,
          details: {
            provider: 'elevenlabs',
            code: 'paid_plan_required',
            // Without this the real cause — Sarvam having rejected the request
            // first — is invisible behind the ElevenLabs billing error.
            ...(sarvamFailure ? { sarvamFailure } : {}),
          },
        },
      );
    }
    if (response.status === 401) {
      throw new AppError(
        'TTS_FAILED',
        'ElevenLabs rejected the API key for text-to-speech. The key needs the text_to_speech scope.',
        { statusCode: 401, retryable: false, details: { provider: 'elevenlabs', code: 'unauthorized' } },
      );
    }
    if (response.status === 429) {
      throw new AppError(
        'TTS_FAILED',
        'ElevenLabs TTS rate limit hit. Try again shortly.',
        { statusCode: 429, retryable: true, details: { provider: 'elevenlabs', code: 'rate_limited', raw: detailJson } },
      );
    }
    throw new AppError(
      'TTS_FAILED',
      `ElevenLabs TTS returned HTTP ${response.status}: ${detail.slice(0, 300)}`,
      { statusCode: 502, retryable: false, details: { provider: 'elevenlabs', status: response.status } },
    );
  }

  const audio = Buffer.from(await response.arrayBuffer());
  const latencyMs = Date.now() - started;

  logger.debug({ voiceId, model: config.tts.model, bytes: audio.length, latencyMs }, 'Synthesised speech');

  return {
    audio,
    contentType: response.headers.get('content-type') ?? 'audio/mpeg',
    model: config.tts.model,
    voiceId,
    latencyMs,
    provider: 'elevenlabs',
  };
}

/** Liveness probe for /api/health. */
export async function ttsHealthCheck(): Promise<{ ok: boolean; detail: string }> {
  if (!config.tts.enabled) {
    return { ok: false, detail: 'Neither SARVAM_API_KEY nor ELEVENLABS_API_KEY set — voice playback disabled' };
  }
  try {
    const result = await synthesize({ text: 'ok' });
    return { ok: true, detail: `${result.provider} · ${result.model} · ${result.latencyMs}ms` };
  } catch (error) {
    return { ok: false, detail: (error as Error).message };
  }
}
