/**
 * Pipeline stage tracker — drawn as an actual glass tube with liquid in it,
 * rather than a progress bar or a row of beads.
 *
 * Two earlier versions of this panel reported the same numbers, first as a
 * disconnected list and then as a rail of dots joined by a line; the dots
 * ended up reading as generic "AI-built progress UI" — a repeated shape
 * standing in for state rather than anything that looks like it belongs to
 * a pipeline. This one leans on the metaphor already in the name: the
 * request is a fluid, the stages sit against the vessel it's filling, and
 * finished means "poured all the way down".
 *
 * What actually sells the liquid — kept deliberately quiet, nothing flashing:
 *
 *   · the **tube** is shaded like glass — dark rims with a bright band just
 *     inside them, the way light behaves on a curved surface;
 *   · the **fluid** is a single flat neutral grey, no gradient, no sheen;
 *   · the **meniscus** is a genuine curved cap at the leading edge of the
 *     fluid, half above the fill line and half below it — the one shape
 *     that turns "a bar that has grown" into "something that was poured".
 *
 * The fill level does the same job the row list used to: it stops at the
 * level of whichever stage currently holds the request, so progress is a
 * physical position on the tube rather than something read off text. A
 * **wash behind each row**, sized to that stage's share of the total time,
 * turns the row list into a waterfall — the clearest signal of where time
 * actually goes, since a 30s wait is obviously the LLM, not retrieval, when
 * the retrieval row is a sliver and generation fills its own row.
 *
 * Each row's own status is a plain ink dot: filled for done, pulsing for
 * running, hollow-toned for idle. `success` stays reserved elsewhere for
 * "passed a check" (guardrails, verification), not "finished running".
 */
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

  // Where the fluid has reached. Rows are a uniform height, so a stage's
  // graduation mark sits at ((index + 0.5) / count) of the tube — filling to
  // exactly that fraction puts the surface at the mark of whichever stage
  // currently holds the request, and 100% once every stage has let go of it.
  const count = visible.length;
  const doneCount = visible.filter((stage) => stages[stage].state === 'done').length;
  const runningIndex = visible.findIndex((stage) => stages[stage].state === 'running');
  const allDone = count > 0 && doneCount === count;
  const frontier = runningIndex >= 0 ? runningIndex : doneCount - 1;
  const fillPercent = allDone ? 100 : frontier < 0 ? 0 : ((frontier + 0.5) / count) * 100;
  const showMeniscus = !allDone && fillPercent > 0;

  return (
    <div>
      <div className="relative">
        {/* ── The tube ──────────────────────────────────────────────────── */}
        <span aria-hidden className="pointer-events-none absolute inset-y-1 left-0 w-2.5">
          <span className="tube-glass absolute inset-0 overflow-hidden rounded-full">
            <span
              className="tube-fluid absolute inset-x-0 top-0 transition-[height] duration-500 ease-out"
              style={{ height: `${fillPercent}%` }}
            />
            {/* Meniscus: half above the fill line, half below — an ellipse
                straddling the edge rather than a flat cut-off. */}
            {showMeniscus ? (
              <span
                className="tube-meniscus absolute left-1/2 h-1.5 w-2.5 -translate-x-1/2 -translate-y-1/2 transition-[top] duration-500 ease-out"
                style={{ top: `${fillPercent}%` }}
              />
            ) : null}
          </span>
        </span>

        <ol className="relative">
          {visible.map((stage) => {
            const progress = stages[stage];
            const share = total > 0 ? (progress.durationMs ?? 0) / total : 0;
            const done = progress.state === 'done';
            const running = progress.state === 'running';

            return (
              <li key={stage} className="relative flex items-center gap-2 py-[3px] pl-6">
                {/* Waterfall wash — percentage is of this track, not of the
                    row, so a tiny share still resolves to a visible sliver
                    instead of a negative width. */}
                {done && share > 0 ? (
                  <span aria-hidden className="absolute inset-y-px left-6 right-0">
                    <span
                      className="block h-full rounded-[3px] bg-ink/[0.055] transition-[width] duration-300"
                      style={{ width: `${Math.max(share * 100, 2).toFixed(1)}%` }}
                    />
                  </span>
                ) : null}

                {/* State dot — sits by the label rather than on the tube, so
                    the tube stays a clean instrument and this stays a plain
                    status readout. */}
                <span
                  className={cn(
                    'relative z-10 size-1.5 shrink-0 rounded-full',
                    done ? 'bg-ink' : running ? 'animate-pulse bg-ink' : 'bg-border-strong',
                  )}
                />

                <span
                  className={cn(
                    'relative z-10 min-w-0 flex-1 truncate text-xs',
                    progress.state === 'idle' ? 'text-ink-tertiary' : 'text-ink',
                  )}
                >
                  {STAGE_LABELS[stage] ?? stage}
                </span>

                <span
                  className={cn(
                    'relative z-10 shrink-0 font-mono text-2xs tabular-nums',
                    done ? 'text-ink-secondary' : 'text-ink-tertiary',
                  )}
                >
                  {progress.durationMs !== null ? formatMs(progress.durationMs) : '—'}
                </span>
              </li>
            );
          })}
        </ol>
      </div>

      {stages.generation.detail || stages.retrieval.detail ? (
        <p className="mt-2.5 border-t border-border pt-2 text-2xs leading-relaxed text-ink-secondary">
          {[stages.retrieval.detail, stages.reranking.detail, stages.prompt.detail]
            .filter(Boolean)
            .join(' · ')}
        </p>
      ) : null}

      {isStreaming ? null : total > 0 ? (
        <p className="mt-2.5 border-t border-border pt-2 text-2xs leading-relaxed text-ink-secondary">
          Stages sum to {formatMs(total)}; retrieval arms run concurrently, so wall-clock is lower.
        </p>
      ) : null}
    </div>
  );
}
