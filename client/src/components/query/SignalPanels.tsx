/**
 * Trust signals: guardrail verdicts, confidence breakdown and latency.
 *
 * These sit next to the answer rather than behind a tab, because the whole
 * point of a grounded RAG system is that you can see *why* it answered the way
 * it did without going looking.
 */
import { Activity, Check, Gauge, ShieldCheck, TriangleAlert, X, Timer } from 'lucide-react';
import type {
  ConfidenceBreakdown,
  ConfidenceFactorKey,
  GuardrailReport,
  GuardrailResult,
  LatencyBreakdown,
} from '@goarag/shared';
import { CONFIDENCE_WEIGHTS, GUARDRAIL_LABELS } from '@goarag/shared';
import { cn, formatMs, formatPercent } from '@/lib/utils';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Badge, KeyValue, Tooltip } from '@/components/ui/primitives';

// ─── Guardrails ──────────────────────────────────────────────────────────────

const STAGE_ORDER = ['pre_retrieval', 'post_retrieval', 'post_generation'] as const;
const STAGE_TITLES: Record<string, string> = {
  pre_retrieval: 'Input',
  post_retrieval: 'Evidence',
  post_generation: 'Answer',
};

function VerdictIcon({ verdict }: { verdict: GuardrailResult['verdict'] }) {
  if (verdict === 'block') return <X className="size-3.5 text-danger" />;
  if (verdict === 'warn') return <TriangleAlert className="size-3.5 text-warning" />;
  return <Check className="size-3.5 text-success" />;
}

export function GuardrailPanel({ report }: { report: GuardrailReport | null }) {
  const results = report?.results ?? [];

  return (
    <Card>
      <CardHeader
        title="Guardrails"
        icon={ShieldCheck}
        description={
          results.length > 0
            ? `${results.filter((r) => r.verdict === 'pass').length}/${results.length} passed`
            : 'Eight checks across three stages'
        }
        action={
          report ? (
            <Badge tone={report.blocked ? 'danger' : report.passed ? 'success' : 'warning'}>
              {report.blocked ? 'Blocked' : report.passed ? 'Passed' : 'Warnings'}
            </Badge>
          ) : null
        }
      />
      <CardContent className="space-y-3">
        {results.length === 0 ? (
          <p className="text-xs text-ink-tertiary">
            Injection, jailbreak, toxicity, similarity, off-topic, context verification,
            hallucination and confidence checks run on every query.
          </p>
        ) : (
          STAGE_ORDER.map((stage) => {
            const staged = results.filter((result) => result.stage === stage);
            if (staged.length === 0) return null;

            return (
              <div key={stage}>
                <p className="label mb-1.5">{STAGE_TITLES[stage]}</p>
                <ul className="space-y-1">
                  {staged.map((result) => (
                    <li key={result.id}>
                      <Tooltip content={result.reason} side="left">
                        <div className="flex items-center gap-2 rounded px-1 py-0.5 hover:bg-subtle">
                          <VerdictIcon verdict={result.verdict} />
                          <span
                            className={cn(
                              'flex-1 truncate text-xs',
                              result.verdict === 'pass' ? 'text-ink-secondary' : 'text-ink',
                            )}
                          >
                            {GUARDRAIL_LABELS[result.id] ?? result.id}
                          </span>
                          <span className="font-mono text-2xs text-ink-tertiary">
                            {result.score.toFixed(4)}
                          </span>
                        </div>
                      </Tooltip>
                    </li>
                  ))}
                </ul>
              </div>
            );
          })
        )}
      </CardContent>
    </Card>
  );
}

// ─── Confidence ──────────────────────────────────────────────────────────────

/**
 * Per-factor explanations. The labels and weights themselves come from
 * `CONFIDENCE_WEIGHTS` in shared, so this panel can never describe a different
 * blend from the one the server actually computed.
 */
const FACTOR_HINTS: Record<ConfidenceFactorKey, string> = {
  topScore: 'Best dense cosine similarity between the query and retrieved context',
  groundedness: 'Share of answer sentences whose content words appear in the retrieved context',
  meanScore: 'Average dense cosine similarity across the chunks sent to the model',
  retrievalAgreement: 'Overlap between what dense search and keyword search independently found',
  contextCoverage: 'How much of the question’s vocabulary appears in the context',
};

const SEGMENT_COLORS = [
  'rgb(var(--brand))',
  'rgb(var(--accent))',
  'rgb(var(--brand) / 0.65)',
  'rgb(var(--accent) / 0.65)',
  'rgb(var(--brand) / 0.3)',
];

/** A factor's value, what it contributed, and what it left on the table. */
interface FactorRow {
  key: ConfidenceFactorKey;
  label: string;
  weight: number;
  value: number;
  /** weight × value — the points this factor actually added to `overall`. */
  contribution: number;
  /** weight − contribution — the points it could have added but didn't. */
  shortfall: number;
}

function factorRows(confidence: ConfidenceBreakdown): FactorRow[] {
  return CONFIDENCE_WEIGHTS.map(({ key, label, weight }) => {
    const value = confidence[key];
    const contribution = weight * value;
    return { key, label, weight, value, contribution, shortfall: weight - contribution };
  });
}

export function ConfidencePanel({ confidence }: { confidence: ConfidenceBreakdown | null }) {
  const rows = confidence ? factorRows(confidence) : [];
  // The single factor costing the most points. Naming it turns the panel from
  // "here are five numbers" into "here is why the score isn't higher".
  const worst = rows.reduce<FactorRow | null>(
    (lowest, row) => (!lowest || row.shortfall > lowest.shortfall ? row : lowest),
    null,
  );
  const headroom = confidence ? confidence.overall - confidence.threshold : 0;

  return (
    <Card>
      <CardHeader
        title="Confidence"
        icon={Gauge}
        description={
          confidence
            ? `Threshold ${formatPercent(confidence.threshold)} · ${
                confidence.sufficient
                  ? `cleared by ${(headroom * 100).toFixed(1)} pts`
                  : `short by ${(-headroom * 100).toFixed(1)} pts`
              }`
            : 'Weighted blend of retrieval and grounding signals'
        }
        action={
          confidence ? (
            <Badge tone={confidence.sufficient ? 'success' : 'warning'}>
              {formatPercent(confidence.overall, 1)}
            </Badge>
          ) : null
        }
      />
      <CardContent className="space-y-3">
        {!confidence ? (
          <p className="text-xs text-ink-tertiary">
            Below the threshold the answer is replaced with an explicit refusal rather than
            hedged.
          </p>
        ) : (
          <>
            <div>
              <div className="mb-1.5 flex items-baseline justify-between">
                <span className="text-xs text-ink-secondary">Overall</span>
                <span className="font-mono text-lg font-semibold">
                  {formatPercent(confidence.overall, 1)}
                </span>
              </div>

              {/*
                Stacked by contribution rather than a single fill, because the
                thing worth seeing is *where the score came from*. Each segment
                is weight × value, so the segments sum to exactly `overall` and
                the empty remainder is the score that was available but not
                earned. A flat bar shows the same total while hiding that a
                third of it is groundedness alone.
              */}
              <div
                className="relative flex h-2 w-full overflow-hidden rounded-full bg-border/50"
                role="img"
                aria-label={`Confidence ${formatPercent(confidence.overall, 1)}, threshold ${formatPercent(confidence.threshold)}`}
              >
                {rows.map((row, index) => (
                  <Tooltip
                    key={row.key}
                    side="top"
                    content={`${row.label} contributed ${(row.contribution * 100).toFixed(1)} of a possible ${(row.weight * 100).toFixed(0)} pts`}
                  >
                    <span
                      className="h-full border-r border-card/70 last:border-r-0"
                      style={{
                        width: `${row.contribution * 100}%`,
                        backgroundColor: SEGMENT_COLORS[index] ?? 'rgb(var(--ink))',
                      }}
                    />
                  </Tooltip>
                ))}
                {/* Threshold marker — turns "cleared" from a claim into
                    something you can see the margin of. */}
                <span
                  className="absolute top-0 h-full w-px bg-danger"
                  style={{ left: `${confidence.threshold * 100}%` }}
                  aria-hidden
                />
              </div>
            </div>

            <div className="space-y-2 border-t border-border pt-2">
              {rows.map((row, index) => (
                <Tooltip key={row.key} content={FACTOR_HINTS[row.key]} side="left">
                  <div>
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="flex min-w-0 items-baseline gap-1.5">
                        {/* Swatch ties the row to its segment above — the only
                            thing making the stacked bar readable. */}
                        <span
                          className="size-1.5 shrink-0 translate-y-px rounded-full"
                          style={{
                            backgroundColor: SEGMENT_COLORS[index] ?? 'rgb(var(--ink))',
                          }}
                          aria-hidden
                        />
                        <span className="truncate text-2xs text-ink-secondary">{row.label}</span>
                      </span>
                      <span className="shrink-0 font-mono text-2xs tabular-nums">
                        <span className="text-ink">{row.value.toFixed(3)}</span>
                        <span className="text-ink-tertiary">
                          {' '}
                          → {(row.contribution * 100).toFixed(1)}/{(row.weight * 100).toFixed(0)}
                        </span>
                      </span>
                    </div>
                    {/*
                      Scaled to this factor's weight, not to 0-1. A 10%-weight
                      factor can never fill more than a tenth of the track, so
                      bar lengths are comparable across rows as contributions —
                      which is what the overall score is made of. Drawn on 0-1
                      instead, arm agreement's 0.143 looked like a catastrophe
                      when it costs 8.6 points out of 100.
                    */}
                    <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-border/40">
                      <div
                        className="h-full rounded-full transition-[width] duration-300"
                        style={{
                          width: `${row.contribution * 100}%`,
                          backgroundColor: SEGMENT_COLORS[index] ?? 'rgb(var(--ink))',
                        }}
                      />
                    </div>
                  </div>
                </Tooltip>
              ))}
            </div>

            {worst && worst.shortfall > 0.02 ? (
              <p className="border-t border-border pt-2 text-2xs leading-relaxed text-ink-secondary">
                Biggest drag: <span className="text-ink">{worst.label}</span> gave up{' '}
                <span className="font-mono">{(worst.shortfall * 100).toFixed(1)}</span> of its{' '}
                <span className="font-mono">{(worst.weight * 100).toFixed(0)}</span> points.
              </p>
            ) : null}
          </>
        )}
      </CardContent>
    </Card>
  );
}

// ─── Latency ─────────────────────────────────────────────────────────────────

export function LatencyPanel({
  latency,
  usage,
}: {
  latency: LatencyBreakdown | null;
  usage: { promptTokens: number; completionTokens: number; totalTokens: number } | null;
}) {
  return (
    <Card>
      <CardHeader
        title="Latency & tokens"
        icon={Timer}
        description={latency ? `${formatMs(latency.total)} end to end` : 'Per-stage timings'}
        action={
          latency?.timeToFirstToken ? (
            <Tooltip content="Time from request start to the first answer token">
              <Badge tone="neutral">
                <Activity className="size-3" />
                TTFT {formatMs(latency.timeToFirstToken)}
              </Badge>
            </Tooltip>
          ) : null
        }
      />
      <CardContent>
        {!latency ? (
          <p className="text-xs text-ink-tertiary">
            Timings for every stage appear here once a query completes.
          </p>
        ) : (
          <dl className="divide-y divide-border">
            {latency.transcription !== null ? (
              <KeyValue label="Transcription" value={formatMs(latency.transcription)} mono />
            ) : null}
            <KeyValue label="Guardrails (in)" value={formatMs(latency.guardrailsPre)} mono />
            <KeyValue label="Embedding" value={formatMs(latency.embedding)} mono />
            <KeyValue label="Dense search" value={formatMs(latency.denseRetrieval)} mono />
            <KeyValue label="Keyword search" value={formatMs(latency.sparseRetrieval)} mono />
            <KeyValue label="Fusion" value={formatMs(latency.fusion)} mono />
            <KeyValue label="Reranking" value={formatMs(latency.reranking)} mono />
            <KeyValue label="Prompt build" value={formatMs(latency.promptBuilding)} mono />
            <KeyValue label="Generation" value={formatMs(latency.generation)} mono />
            <KeyValue label="Guardrails (out)" value={formatMs(latency.guardrailsPost)} mono />
            <KeyValue
              label="Total"
              value={<span className="font-medium">{formatMs(latency.total)}</span>}
              mono
            />
            {usage ? (
              <>
                <KeyValue
                  label="Prompt tokens"
                  value={usage.promptTokens.toLocaleString()}
                  mono
                />
                <KeyValue
                  label="Completion tokens"
                  value={usage.completionTokens.toLocaleString()}
                  mono
                />
                <KeyValue
                  label="Total tokens"
                  value={<span className="font-medium">{usage.totalTokens.toLocaleString()}</span>}
                  mono
                />
              </>
            ) : null}
          </dl>
        )}
      </CardContent>
    </Card>
  );
}
