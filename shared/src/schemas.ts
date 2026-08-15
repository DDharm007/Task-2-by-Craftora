/**
 * Zod schemas for every request the API accepts.
 *
 * The server validates with these at the edge; the client imports the inferred
 * types so a contract change breaks the build on both sides at once.
 */
import { z } from 'zod';

/**
 * Boolean parsed from a multipart/form-data field.
 *
 * `z.coerce.boolean()` is wrong here: it applies `Boolean(value)`, and
 * `Boolean('false')` is `true` — so every explicit "false" a client sends
 * would silently become true. Multipart fields are always strings, so the
 * common falsy spellings have to be handled explicitly.
 */
const formBoolean = (fallback: boolean) =>
  z
    .union([z.boolean(), z.string()])
    .optional()
    .transform((value) => {
      if (value === undefined) return fallback;
      if (typeof value === 'boolean') return value;
      const normalized = value.trim().toLowerCase();
      if (normalized === '') return fallback;
      return !['false', '0', 'no', 'off', 'null', 'undefined'].includes(normalized);
    });

/** Conversation history turn. */
export const conversationTurnSchema = z.object({
  role: z.enum(['user', 'assistant']),
  content: z.string().min(1).max(8_000),
});

/** Shared knobs that let the UI override retrieval behaviour per request. */
export const retrievalOptionsSchema = z
  .object({
    topK: z.number().int().min(1).max(50).optional(),
    rerankTopN: z.number().int().min(1).max(20).optional(),
    /** Restrict search to these dataset language tags. */
    languages: z.array(z.string().min(2).max(16)).max(24).optional(),
    enableRerank: z.boolean().optional(),
    enableMmr: z.boolean().optional(),
    enableParentExpansion: z.boolean().optional(),
    /** Turn on the model's reasoning mode (much slower, shows chain-of-thought). */
    enableThinking: z.boolean().optional(),
    /** Skip generation and return retrieval only — used by the chunk inspector. */
    retrievalOnly: z.boolean().optional(),
  })
  .strict();

/** `POST /api/query` */
export const queryRequestSchema = z
  .object({
    query: z
      .string()
      .trim()
      .min(2, 'Query must be at least 2 characters')
      .max(2_000, 'Query must be at most 2000 characters'),
    history: z.array(conversationTurnSchema).max(20).default([]),
    options: retrievalOptionsSchema.default({}),
    /** When true the response is streamed as SSE instead of JSON. */
    stream: z.boolean().default(false),
  })
  .strict();

/**
 * `POST /api/transcribe` and `POST /api/voice-query` accept multipart audio.
 * These fields arrive as strings alongside the file, so they are coerced.
 */
export const transcribeOptionsSchema = z
  .object({
    /** ISO-639-3 hint passed to ElevenLabs, e.g. `hin`. Blank = auto-detect. */
    languageCode: z.string().trim().min(2).max(8).optional(),
    /** Ask the STT model to tag distinct speakers. */
    diarize: formBoolean(false),
  })
  .strip();

/** `POST /api/voice-query` — multipart audio plus the same retrieval options. */
export const voiceQueryOptionsSchema = transcribeOptionsSchema.extend({
  history: z
    .string()
    .optional()
    .transform((value, ctx) => {
      if (!value) return [] as Array<z.infer<typeof conversationTurnSchema>>;
      try {
        return z.array(conversationTurnSchema).max(20).parse(JSON.parse(value));
      } catch {
        ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'history must be a JSON array of turns' });
        return z.NEVER;
      }
    }),
  options: z
    .string()
    .optional()
    .transform((value, ctx) => {
      if (!value) return {} as z.infer<typeof retrievalOptionsSchema>;
      try {
        return retrievalOptionsSchema.parse(JSON.parse(value));
      } catch {
        ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'options must be a JSON object' });
        return z.NEVER;
      }
    }),
  stream: formBoolean(false),
});

/** `POST /api/speak` — synthesise an answer with Sarvam AI or ElevenLabs. */
export const speakRequestSchema = z
  .object({
    text: z.string().trim().min(1, 'Nothing to speak').max(5_000),
    /** Override the configured voice (an ElevenLabs voice id or Sarvam speaker). */
    voiceId: z.string().trim().min(1).max(64).optional(),
    /**
     * Language to speak in. Accepts what the STT returns (`hin_Deva`), a bare
     * ISO code (`hi`/`hin`) or BCP-47 (`hi-IN`). Without it the server default
     * applies, which mispronounces answers in any other language.
     */
    languageCode: z.string().trim().min(2).max(16).optional(),
  })
  .strict();

/** `GET /api/benchmark` */
export const benchmarkQuerySchema = z
  .object({
    /** Number of dataset queries to evaluate. */
    sampleSize: z.coerce.number().int().min(1).max(100).default(10),
    /** Include LLM generation in the measurement (slow but end-to-end). */
    generation: z
      .union([z.boolean(), z.string()])
      .default(false)
      .transform((v) => v === true || v === 'true' || v === '1'),
    /** Restrict evaluation to one dataset language. */
    language: z.string().trim().min(2).max(16).optional(),
    /** How many benchmark cases to run at once. */
    concurrency: z.coerce.number().int().min(1).max(8).default(2),
  })
  .strip();

/** `GET /api/stats` */
export const statsQuerySchema = z
  .object({
    /** How many recent requests to include in the rolling window. */
    recentLimit: z.coerce.number().int().min(0).max(200).default(25),
  })
  .strip();

/** `GET /api/chunks/:id` style lookup for the inspector. */
export const chunkLookupSchema = z
  .object({
    id: z.string().min(1).max(200),
  })
  .strip();

export type QueryRequest = z.infer<typeof queryRequestSchema>;
export type RetrievalOptions = z.infer<typeof retrievalOptionsSchema>;
export type TranscribeOptions = z.infer<typeof transcribeOptionsSchema>;
export type VoiceQueryOptions = z.infer<typeof voiceQueryOptionsSchema>;
export type SpeakRequest = z.infer<typeof speakRequestSchema>;
export type BenchmarkQuery = z.infer<typeof benchmarkQuerySchema>;
export type StatsQuery = z.infer<typeof statsQuerySchema>;
export type ConversationTurnInput = z.infer<typeof conversationTurnSchema>;
