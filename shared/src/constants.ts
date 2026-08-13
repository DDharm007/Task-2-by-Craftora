/** Constants shared by the server pipeline and the client UI. */

/** Exact refusal string required when retrieval finds no usable evidence. */
export const INSUFFICIENT_EVIDENCE_MESSAGE =
  "I couldn't find enough evidence in the retrieved documents.";

/** Exact refusal string used when confidence falls below the threshold. */
export const LOW_CONFIDENCE_MESSAGE = "I don't have enough information.";

/** Audio MIME types accepted by the transcription endpoints. */
export const SUPPORTED_AUDIO_MIME_TYPES = [
  'audio/webm',
  'audio/ogg',
  'audio/wav',
  'audio/x-wav',
  'audio/wave',
  'audio/mpeg',
  'audio/mp3',
  'audio/mp4',
  'audio/m4a',
  'audio/x-m4a',
  'audio/flac',
  'video/webm',
] as const;

/** Human-readable labels for each pipeline stage, used by the UI. */
export const STAGE_LABELS: Record<string, string> = {
  transcription: 'Transcription',
  guardrails: 'Guardrails',
  embedding: 'Embedding',
  retrieval: 'Retrieval',
  reranking: 'Reranking',
  prompt: 'Prompt build',
  generation: 'Generation',
  verification: 'Verification',
};

/** Display names for guardrail ids. */
export const GUARDRAIL_LABELS: Record<string, string> = {
  prompt_injection: 'Prompt injection',
  jailbreak: 'Jailbreak',
  toxicity: 'Toxic query',
  off_topic: 'Off-topic',
  similarity_threshold: 'Similarity threshold',
  context_verification: 'Context verification',
  hallucination: 'Hallucination',
  confidence: 'Confidence',
};

/**
 * Dataset language tags (FLORES-200 style) mapped to display names.
 * MSMARCO-XI covers Indic languages plus the English source.
 */
export const LANGUAGE_NAMES: Record<string, string> = {
  eng_Latn: 'English',
  asm_Beng: 'Assamese',
  ben_Beng: 'Bengali',
  brx_Deva: 'Bodo',
  doi_Deva: 'Dogri',
  gom_Deva: 'Konkani',
  guj_Gujr: 'Gujarati',
  hin_Deva: 'Hindi',
  kan_Knda: 'Kannada',
  kas_Arab: 'Kashmiri',
  mai_Deva: 'Maithili',
  mal_Mlym: 'Malayalam',
  mar_Deva: 'Marathi',
  mni_Beng: 'Manipuri',
  npi_Deva: 'Nepali',
  ory_Orya: 'Odia',
  pan_Guru: 'Punjabi',
  san_Deva: 'Sanskrit',
  sat_Olck: 'Santali',
  snd_Arab: 'Sindhi',
  tam_Taml: 'Tamil',
  tel_Telu: 'Telugu',
  urd_Arab: 'Urdu',
};

/** Resolve a dataset language tag to a display name, falling back to the tag. */
export function languageName(tag: string): string {
  return LANGUAGE_NAMES[tag] ?? tag;
}

export const API_ROUTES = {
  transcribe: '/api/transcribe',
  query: '/api/query',
  voiceQuery: '/api/voice-query',
  benchmark: '/api/benchmark',
  health: '/api/health',
  stats: '/api/stats',
} as const;
