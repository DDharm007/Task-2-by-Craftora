/**
 * Pipeline stage tracker.
 *
 * Shows each stage's state and measured duration as the request progresses.
 * This is the clearest signal of where time actually goes — and it makes an
 * apparently-slow request legible: a 30s wait is obviously the LLM, not
 * retrieval, when the retrieval row reads 40ms.
 */
import { Check, Loader2, Minus } from 'lucide-react';
import { STAGE_LABELS } from '@goarag/shared';
import { cn, formatMs } from '@/lib/utils';
import { STAGES, useSession } from '@/store/session';

export function PipelineTimeline() {
  const stages = useSession((state) => state.stages);
  const isStreaming = useSession((state) => state.isStreaming);
  const transcription = useSession((state) => state.transcription);

  // The transcription row is meaningless for a typed query.
  const visible = STAGES.filter(
    (stage) => stage !== 'transcription' || transcription || stages.transcription.state !== 'idle',
  );

  const total = visible.reduce((sum, stage) => sum + (stages[stage]?.durationMs ?? 0), 0);

  return (
    <div>
      <ol className="space-y-0.5">
        {visible.map((stage) => {
          const progress = stages[stage];
          const share = total > 0 ? (progress.durationMs ?? 0) / total : 0;

          return (
            <li key={stage} className="group relative flex items-center gap-2.5 py-1">
              <span className="flex size-4 shrink-0 items-center justify-center">
                {progress.state === 'done' ? (
                  <Check className="size-3.5 text-success" />
                ) : progress.state === 'running' ? (
                  <Loader2 className="size-3.5 animate-spin text-ink" />
                ) : (
                  <Minus className="size-3 text-ink-tertiary" />
                )}
              </span>

              <span
                className={cn(
                  'w-24 shrink-0 text-xs',
                  progress.state === 'idle' ? 'text-ink-tertiary' : 'text-ink',
                )}
              >
                {STAGE_LABELS[stage] ?? stage}
              </span>

              {/* Proportional bar: the visual weight is the share of total time. */}
              <span className="hidden h-1 min-w-0 flex-1 overflow-hidden rounded-full bg-border/50 sm:block">
                <span
                  className={cn(
                    'block h-full rounded-full transition-[width] duration-300',
                    progress.state === 'done' ? 'bg-ink/70' : 'bg-transparent',
                  )}
                  style={{ width: `${Math.max(share * 100, progress.state === 'done' ? 2 : 0)}%` }}
                />
              </span>

              <span className="w-14 shrink-0 text-right font-mono text-2xs text-ink-secondary">
                {progress.durationMs !== null ? formatMs(progress.durationMs) : '—'}
              </span>
            </li>
          );
        })}
      </ol>

      {stages.generation.detail || stages.retrieval.detail ? (
        <p className="mt-2 border-t border-border pt-2 text-2xs text-ink-secondary">
          {[stages.retrieval.detail, stages.reranking.detail, stages.prompt.detail]
            .filter(Boolean)
            .join(' · ')}
        </p>
      ) : null}

      {isStreaming ? null : total > 0 ? (
        <p className="mt-2 border-t border-border pt-2 text-2xs text-ink-secondary">
          Stages sum to {formatMs(total)}; retrieval arms run concurrently, so wall-clock is lower.
        </p>
      ) : null}
    </div>
  );
}
