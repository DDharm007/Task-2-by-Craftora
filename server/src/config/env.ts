/**
 * Environment loading and validation.
 *
 * Every configurable value in the system funnels through here. Secrets are read
 * from the environment only — nothing is ever hardcoded, and the parsed config
 * is redacted before it is logged or returned by any endpoint.
 */
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';
import { z } from 'zod';

const here = path.dirname(fileURLToPath(import.meta.url));

/** Walk up from this file to find the monorepo root (the dir holding `.env`). */
function findRepoRoot(): string {
  let dir = here;
  for (let i = 0; i < 8; i += 1) {
    if (existsSync(path.join(dir, 'package.json')) && existsSync(path.join(dir, 'shared'))) {
      return dir;
    }
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return process.cwd();
}

export const REPO_ROOT = findRepoRoot();

// Load `.env` from the repo root, then allow a local override file.
dotenv.config({ path: path.join(REPO_ROOT, '.env') });
dotenv.config({ path: path.join(REPO_ROOT, '.env.local'), override: true });

/** Coerce the loose truthy strings people actually put in .env files. */
const booleanish = (fallback: boolean) =>
  z
    .string()
    .optional()
    .transform((v) => {
      if (v === undefined || v.trim() === '') return fallback;
      return ['1', 'true', 'yes', 'on'].includes(v.trim().toLowerCase());
    });

const intWithDefault = (fallback: number, min: number, max: number) =>
  z
    .string()
    .optional()
    .transform((v) => (v === undefined || v.trim() === '' ? fallback : Number(v)))
    .pipe(z.number().int().min(min).max(max));

const floatWithDefault = (fallback: number, min: number, max: number) =>
  z
    .string()
    .optional()
    .transform((v) => (v === undefined || v.trim() === '' ? fallback : Number(v)))
    .pipe(z.number().min(min).max(max));

const stringWithDefault = (fallback: string) =>
  z
    .string()
    .optional()
    .transform((v) => (v === undefined || v.trim() === '' ? fallback : v.trim()));

/** Comma-separated list → trimmed array, empty string → empty array. */
const csv = z
  .string()
  .optional()
  .transform((v) =>
    (v ?? '')
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean),
  );

const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  PORT: intWithDefault(8787, 1, 65535),
  LOG_LEVEL: z.enum(['fatal', 'error', 'warn', 'info', 'debug', 'trace', 'silent']).default('info'),
  CORS_ORIGIN: csv,

  // Groq (LLM generation)
  GROQ_API_KEY: z.string().trim().min(1, 'GROQ_API_KEY is required'),
  GROQ_BASE_URL: stringWithDefault('https://api.groq.com/openai/v1'),
  GROQ_MODEL: stringWithDefault('openai/gpt-oss-120b'),
  // `reasoning_budget` and `chat_template_kwargs.enable_thinking` (sent when
  // this is on — see llm.service.ts) are extensions only some gateways accept.
  // Groq's OpenAI-compatible endpoint rejects them outright with a 400, so this
  // must stay `false` unless GROQ_BASE_URL points at a gateway that supports
  // them.
  LLM_ENABLE_THINKING: booleanish(false),
  LLM_TEMPERATURE: floatWithDefault(0.2, 0, 2),
  LLM_TOP_P: floatWithDefault(0.95, 0, 1),
  LLM_MAX_TOKENS: intWithDefault(2048, 64, 32768),
  LLM_REASONING_BUDGET: intWithDefault(4096, 0, 32768),
  LLM_TIMEOUT_MS: intWithDefault(120_000, 5_000, 600_000),
  LLM_MAX_RETRIES: intWithDefault(2, 0, 5),

  // ElevenLabs / Sarvam
  ELEVENLABS_API_KEY: z.string().trim().default(''),
  SARVAM_API_KEY: z.string().trim().default(''),
  ELEVENLABS_STT_MODEL: stringWithDefault('scribe_v1'),
  ELEVENLABS_TTS_MODEL: stringWithDefault('eleven_multilingual_v2'),
  ELEVENLABS_VOICE_ID: stringWithDefault('21m00Tcm4TlvDq8ikWAM'),
  ELEVENLABS_BASE_URL: stringWithDefault('https://api.elevenlabs.io/v1'),
  // Sarvam speakers are model-specific: a `bulbul:v2` name on `bulbul:v3` is a
  // hard 400. Keep the two in sync when changing either.
  //
  // v2 is the default for latency. Measured on this corpus it returns in about
  // a second regardless of input length, while v3 scales with it — ~3.7s for a
  // sentence and ~10.6s at the 500-character cap. v3 is the better voice; set
  // it here if quality matters more than responsiveness.
  SARVAM_TTS_MODEL: stringWithDefault('bulbul:v2'),
  SARVAM_TTS_SPEAKER: stringWithDefault('anushka'),
  /** Language used when the caller does not supply one. */
  SARVAM_TTS_LANGUAGE: stringWithDefault('en-IN'),
  // Does not affect synthesis time, only payload size — 22.05 kHz is roughly
  // 3× the bytes of 8 kHz for noticeably better speech.
  SARVAM_TTS_SAMPLE_RATE: intWithDefault(22_050, 8_000, 48_000),
  STT_TIMEOUT_MS: intWithDefault(120_000, 5_000, 600_000),
  MAX_AUDIO_UPLOAD_MB: intWithDefault(25, 1, 200),

  // Embeddings
  EMBEDDING_PROVIDER: z.enum(['local']).default('local'),
  EMBEDDING_MODEL: stringWithDefault('BAAI/bge-m3'),
  EMBEDDING_DIMENSIONS: intWithDefault(1024, 64, 8192),
  EMBEDDING_BATCH_SIZE: intWithDefault(8, 1, 256),
  EMBEDDING_MAX_TOKENS: intWithDefault(512, 64, 8192),
  EMBEDDING_QUANTIZATION: z.enum(['fp32', 'fp16', 'int8', 'uint8', 'q4', 'q4f16']).default('int8'),

  // Reranker
  // `heuristic` by default: the cross-encoder costs ~1.55s at p50 on CPU,
  // which dominates the latency budget. See .env.example for measured numbers.
  RERANKER_PROVIDER: z.enum(['local', 'heuristic']).default('heuristic'),
  RERANKER_MODEL: stringWithDefault('Xenova/bge-reranker-base'),
  RERANKER_QUANTIZATION: z.enum(['fp32', 'fp16', 'int8', 'uint8', 'q4', 'q4f16']).default('int8'),
  RERANK_BATCH_SIZE: intWithDefault(4, 1, 32),

  // Vector store
  VECTOR_STORE: z.enum(['qdrant', 'embedded', 'auto']).default('auto'),
  QDRANT_URL: stringWithDefault('http://localhost:6333'),
  QDRANT_API_KEY: z.string().trim().default(''),
  QDRANT_COLLECTION: stringWithDefault('goarag_msmarco_xi'),
  EMBEDDED_STORE_PATH: stringWithDefault('./storage/vectors'),

  // Retrieval
  RETRIEVAL_TOP_K: intWithDefault(10, 1, 100),
  RERANK_TOP_N: intWithDefault(5, 1, 50),
  RRF_K: intWithDefault(60, 1, 1000),
  MMR_LAMBDA: floatWithDefault(0.7, 0, 1),
  ENABLE_MMR: booleanish(true),
  ENABLE_PARENT_EXPANSION: booleanish(true),

  // Chunking
  CHUNK_TARGET_TOKENS: intWithDefault(220, 32, 2048),
  CHUNK_MIN_TOKENS: intWithDefault(48, 8, 512),
  CHUNK_OVERLAP_TOKENS: intWithDefault(48, 0, 512),
  PARENT_CHUNK_TOKENS: intWithDefault(700, 128, 4096),
  SEMANTIC_BREAKPOINT_PERCENTILE: intWithDefault(88, 50, 99),
  SLIDING_WINDOW_TOKENS: intWithDefault(160, 32, 1024),
  SLIDING_WINDOW_STRIDE: intWithDefault(80, 8, 1024),

  // Guardrails
  SIMILARITY_THRESHOLD: floatWithDefault(0.28, 0, 1),
  CONFIDENCE_THRESHOLD: floatWithDefault(0.42, 0, 1),
  GROUNDEDNESS_THRESHOLD: floatWithDefault(0.55, 0, 1),
  ENABLE_GUARDRAILS: booleanish(true),
  ENABLE_HALLUCINATION_CHECK: booleanish(true),

  // Rate limiting
  RATE_LIMIT_WINDOW_MS: intWithDefault(60_000, 1_000, 3_600_000),
  RATE_LIMIT_MAX: intWithDefault(60, 1, 10_000),
  RATE_LIMIT_QUERY_MAX: intWithDefault(20, 1, 10_000),
  RATE_LIMIT_TRANSCRIBE_MAX: intWithDefault(12, 1, 10_000),

  // Dataset
  DATASET_REPO: stringWithDefault('ai4bharat/MSMARCO-XI'),
  DATASET_SPLIT: stringWithDefault('validation'),
  DATASET_SOURCE: z.enum(['parquet', 'api', 'auto']).default('auto'),
  DATASET_MAX_ROWS: intWithDefault(300, 1, 100_000),
  DATASET_LANGUAGES: csv,
  DATASET_INCLUDE_ENGLISH: booleanish(true),
  DATASET_DIR: stringWithDefault('./dataset'),
  INDEX_BATCH_SIZE: intWithDefault(64, 1, 512),
  INDEX_CONCURRENCY: intWithDefault(3, 1, 16),
});

export type RawEnv = z.infer<typeof envSchema>;

function formatIssues(error: z.ZodError): string {
  return error.issues.map((i) => `  • ${i.path.join('.') || '(root)'}: ${i.message}`).join('\n');
}

const parsed = envSchema.safeParse(process.env);

if (!parsed.success) {
  // Fail fast and loudly — a half-configured RAG server is worse than none.
  process.stderr.write(
    `\n✖ Invalid environment configuration:\n${formatIssues(parsed.error)}\n\n` +
      `Copy .env.example to .env and fill in the required values.\n\n`,
  );
  process.exit(1);
}

const env: RawEnv = parsed.data;

/** Resolve a possibly-relative path from the .env against the repo root. */
export function resolveFromRoot(p: string): string {
  return path.isAbsolute(p) ? p : path.resolve(REPO_ROOT, p);
}

export const config = {
  env: env.NODE_ENV,
  isProduction: env.NODE_ENV === 'production',
  isDev: env.NODE_ENV === 'development',
  port: env.PORT,
  logLevel: env.LOG_LEVEL,
  corsOrigins: env.CORS_ORIGIN.length > 0 ? env.CORS_ORIGIN : ['http://localhost:5173'],
  version: '1.0.0',

  llm: {
    apiKey: env.GROQ_API_KEY,
    baseUrl: env.GROQ_BASE_URL,
    model: env.GROQ_MODEL,
    enableThinking: env.LLM_ENABLE_THINKING,
    temperature: env.LLM_TEMPERATURE,
    topP: env.LLM_TOP_P,
    maxTokens: env.LLM_MAX_TOKENS,
    reasoningBudget: env.LLM_REASONING_BUDGET,
    timeoutMs: env.LLM_TIMEOUT_MS,
    maxRetries: env.LLM_MAX_RETRIES,
  },

  stt: {
    apiKey: env.ELEVENLABS_API_KEY,
    model: env.ELEVENLABS_STT_MODEL,
    baseUrl: env.ELEVENLABS_BASE_URL,
    timeoutMs: env.STT_TIMEOUT_MS,
    maxUploadBytes: env.MAX_AUDIO_UPLOAD_MB * 1024 * 1024,
    enabled: env.ELEVENLABS_API_KEY.length > 0,
  },

  tts: {
    apiKey: env.ELEVENLABS_API_KEY,
    sarvamApiKey: env.SARVAM_API_KEY,
    sarvamModel: env.SARVAM_TTS_MODEL,
    sarvamSpeaker: env.SARVAM_TTS_SPEAKER,
    sarvamLanguage: env.SARVAM_TTS_LANGUAGE,
    sarvamSampleRate: env.SARVAM_TTS_SAMPLE_RATE,
    baseUrl: env.ELEVENLABS_BASE_URL,
    model: env.ELEVENLABS_TTS_MODEL,
    voiceId: env.ELEVENLABS_VOICE_ID,
    timeoutMs: env.STT_TIMEOUT_MS,
    enabled: env.ELEVENLABS_API_KEY.length > 0 || env.SARVAM_API_KEY.length > 0,
  },

  embedding: {
    provider: env.EMBEDDING_PROVIDER,
    model: env.EMBEDDING_MODEL,
    dimensions: env.EMBEDDING_DIMENSIONS,
    batchSize: env.EMBEDDING_BATCH_SIZE,
    maxTokens: env.EMBEDDING_MAX_TOKENS,
    quantization: env.EMBEDDING_QUANTIZATION,
    cacheDir: resolveFromRoot('./models'),
  },

  reranker: {
    provider: env.RERANKER_PROVIDER,
    model: env.RERANKER_MODEL,
    quantization: env.RERANKER_QUANTIZATION,
    batchSize: env.RERANK_BATCH_SIZE,
  },

  vectorStore: {
    driver: env.VECTOR_STORE,
    qdrantUrl: env.QDRANT_URL,
    qdrantApiKey: env.QDRANT_API_KEY,
    collection: env.QDRANT_COLLECTION,
    embeddedPath: resolveFromRoot(env.EMBEDDED_STORE_PATH),
  },

  retrieval: {
    topK: env.RETRIEVAL_TOP_K,
    rerankTopN: env.RERANK_TOP_N,
    rrfK: env.RRF_K,
    mmrLambda: env.MMR_LAMBDA,
    enableMmr: env.ENABLE_MMR,
    enableParentExpansion: env.ENABLE_PARENT_EXPANSION,
  },

  chunking: {
    targetTokens: env.CHUNK_TARGET_TOKENS,
    minTokens: env.CHUNK_MIN_TOKENS,
    overlapTokens: env.CHUNK_OVERLAP_TOKENS,
    parentTokens: env.PARENT_CHUNK_TOKENS,
    semanticPercentile: env.SEMANTIC_BREAKPOINT_PERCENTILE,
    windowTokens: env.SLIDING_WINDOW_TOKENS,
    windowStride: env.SLIDING_WINDOW_STRIDE,
  },

  guardrails: {
    similarityThreshold: env.SIMILARITY_THRESHOLD,
    confidenceThreshold: env.CONFIDENCE_THRESHOLD,
    groundednessThreshold: env.GROUNDEDNESS_THRESHOLD,
    enabled: env.ENABLE_GUARDRAILS,
    hallucinationCheck: env.ENABLE_HALLUCINATION_CHECK,
  },

  rateLimit: {
    windowMs: env.RATE_LIMIT_WINDOW_MS,
    max: env.RATE_LIMIT_MAX,
    queryMax: env.RATE_LIMIT_QUERY_MAX,
    transcribeMax: env.RATE_LIMIT_TRANSCRIBE_MAX,
  },

  dataset: {
    repo: env.DATASET_REPO,
    split: env.DATASET_SPLIT,
    source: env.DATASET_SOURCE,
    maxRows: env.DATASET_MAX_ROWS,
    languages: env.DATASET_LANGUAGES,
    includeEnglish: env.DATASET_INCLUDE_ENGLISH,
    dir: resolveFromRoot(env.DATASET_DIR),
    batchSize: env.INDEX_BATCH_SIZE,
    concurrency: env.INDEX_CONCURRENCY,
  },
} as const;

export type AppConfig = typeof config;

/** Config with every secret stripped — safe to log or expose over HTTP. */
export function redactedConfig(): Record<string, unknown> {
  const mask = (secret: string) => (secret ? `${secret.slice(0, 6)}…${secret.slice(-4)}` : '(unset)');
  return {
    env: config.env,
    port: config.port,
    llm: { ...config.llm, apiKey: mask(config.llm.apiKey) },
    stt: { ...config.stt, apiKey: mask(config.stt.apiKey) },
    tts: { ...config.tts, apiKey: mask(config.tts.apiKey) },
    embedding: config.embedding,
    reranker: config.reranker,
    vectorStore: { ...config.vectorStore, qdrantApiKey: mask(config.vectorStore.qdrantApiKey) },
    retrieval: config.retrieval,
    chunking: config.chunking,
    guardrails: config.guardrails,
    dataset: config.dataset,
  };
}
