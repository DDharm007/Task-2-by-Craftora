/**
 * Query session state.
 *
 * Holds everything about the in-flight and most recent query: streaming
 * answer text, retrieved chunks, guardrail report, stage timings and
 * conversation history.
 */
import { create } from 'zustand';
import type {
  ConversationTurn,
  GuardrailReport,
  PipelineStage,
  QueryResult,
  RetrievalOptions,
  RetrievedChunk,
  StreamEvent,
  TranscriptionResult,
} from '@goarag/shared';

export type StageState = 'idle' | 'running' | 'done';

export interface StageProgress {
  state: StageState;
  durationMs: number | null;
  detail: string | null;
}

const STAGES: PipelineStage[] = [
  'transcription',
  'guardrails',
  'embedding',
  'retrieval',
  'reranking',
  'prompt',
  'generation',
  'verification',
];

function emptyStages(): Record<PipelineStage, StageProgress> {
  return Object.fromEntries(
    STAGES.map((stage) => [stage, { state: 'idle', durationMs: null, detail: null }]),
  ) as Record<PipelineStage, StageProgress>;
}

export interface SessionState {
  /** True while a query is streaming. */
  isStreaming: boolean;
  /** Answer text accumulated from token deltas. */
  streamedAnswer: string;
  /** Reasoning text, when thinking mode is on. */
  streamedReasoning: string;
  transcription: TranscriptionResult | null;
  chunks: RetrievedChunk[];
  guardrails: GuardrailReport | null;
  stages: Record<PipelineStage, StageProgress>;
  /** The finished result, available once the stream completes. */
  result: QueryResult | null;
  error: string | null;
  history: ConversationTurn[];
  options: RetrievalOptions;
  /** Chunk currently open in the inspector. */
  inspectedChunkId: string | null;

  beginQuery: (voice: boolean) => void;
  applyEvent: (event: StreamEvent) => void;
  failQuery: (message: string) => void;
  endQuery: () => void;
  reset: () => void;
  clearHistory: () => void;
  setOptions: (options: Partial<RetrievalOptions>) => void;
  inspectChunk: (chunkId: string | null) => void;
}

const DEFAULT_OPTIONS: RetrievalOptions = {
  topK: 10,
  rerankTopN: 5,
  enableRerank: false,
  enableMmr: true,
  enableParentExpansion: true,
  enableThinking: false,
};

export const useSession = create<SessionState>((set, get) => ({
  isStreaming: false,
  streamedAnswer: '',
  streamedReasoning: '',
  transcription: null,
  chunks: [],
  guardrails: null,
  stages: emptyStages(),
  result: null,
  error: null,
  history: [],
  options: DEFAULT_OPTIONS,
  inspectedChunkId: null,

  beginQuery: (voice) =>
    set((state) => ({
      isStreaming: true,
      streamedAnswer: '',
      streamedReasoning: '',
      transcription: null,
      chunks: [],
      guardrails: null,
      result: null,
      error: null,
      inspectedChunkId: null,
      stages: {
        ...emptyStages(),
        // A voice query starts in transcription; a typed one skips it.
        transcription: voice
          ? { state: 'running', durationMs: null, detail: null }
          : { state: 'idle', durationMs: null, detail: null },
      },
      history: state.history,
    })),

  applyEvent: (event) => {
    switch (event.type) {
      case 'stage':
        set((state) => ({
          stages: {
            ...state.stages,
            [event.stage]: {
              state: event.status === 'completed' ? 'done' : 'running',
              durationMs: event.durationMs ?? state.stages[event.stage]?.durationMs ?? null,
              detail: event.detail ?? state.stages[event.stage]?.detail ?? null,
            },
          },
        }));
        break;

      case 'transcript':
        set((state) => ({
          transcription: event.transcription,
          stages: {
            ...state.stages,
            transcription: {
              state: 'done',
              durationMs: event.transcription.latencyMs,
              detail: event.transcription.languageCode,
            },
          },
        }));
        break;

      case 'chunks':
        set({ chunks: event.chunks });
        break;

      case 'guardrails':
        // Merge: the report arrives in stages, and later stages add results.
        set((state) => ({
          guardrails: mergeGuardrails(state.guardrails, event.report),
        }));
        break;

      case 'reasoning':
        set((state) => ({ streamedReasoning: state.streamedReasoning + event.delta }));
        break;

      case 'token':
        set((state) => ({ streamedAnswer: state.streamedAnswer + event.delta }));
        break;

      case 'done': {
        const { history } = get();
        set({
          result: event.result,
          // The server's final answer is authoritative: the confidence gate may
          // have replaced the streamed text with a refusal.
          streamedAnswer: event.result.answer,
          chunks: event.result.chunks,
          guardrails: event.result.guardrails,
          transcription: event.result.transcription,
          isStreaming: false,
          history: [
            ...history,
            { role: 'user' as const, content: event.result.query },
            { role: 'assistant' as const, content: event.result.answer },
          ].slice(-12),
        });
        break;
      }

      case 'error':
        set({ error: event.error.message, isStreaming: false });
        break;

      case 'start':
        break;
    }
  },

  failQuery: (message) => set({ error: message, isStreaming: false }),
  endQuery: () => set({ isStreaming: false }),

  reset: () =>
    set({
      isStreaming: false,
      streamedAnswer: '',
      streamedReasoning: '',
      transcription: null,
      chunks: [],
      guardrails: null,
      stages: emptyStages(),
      result: null,
      error: null,
      inspectedChunkId: null,
    }),

  clearHistory: () => set({ history: [] }),

  setOptions: (options) => set((state) => ({ options: { ...state.options, ...options } })),

  inspectChunk: (chunkId) => set({ inspectedChunkId: chunkId }),
}));

/** Union two partial reports, keeping the latest verdict for each guardrail. */
function mergeGuardrails(
  previous: GuardrailReport | null,
  incoming: GuardrailReport,
): GuardrailReport {
  if (!previous) return incoming;

  const byId = new Map(previous.results.map((result) => [result.id, result]));
  for (const result of incoming.results) byId.set(result.id, result);
  const results = [...byId.values()];
  const blocking = results.find((result) => result.verdict === 'block');

  return {
    passed: !blocking,
    blocked: Boolean(blocking),
    blockedBy: blocking?.id ?? null,
    results,
    totalDurationMs: results.reduce((sum, result) => sum + result.durationMs, 0),
  };
}

export { STAGES };
