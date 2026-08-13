/**
 * Trust signals: guardrail verdicts, confidence breakdown and latency.
 *
 * These sit next to the answer rather than behind a tab, because the whole
 * point of a grounded RAG system is that you can see *why* it answered the way
 * it did without going looking.
 */
import { Activity, Check, Gauge, ShieldCheck, TriangleAlert, X, Timer } from 'lucide-react';
import type { ConfidenceBreakdown, GuardrailReport, GuardrailResult, LatencyBreakdown } from '@voxrag/shared';
import { GUARDRAIL_LABELS } from '@voxrag/shared';
import { cn, formatMs, formatPercent, scoreTone } from '@/lib/utils';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Badge, KeyValue, Meter, Tooltip } from '@/components/ui/primitives';

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
                            {result.score.toFixed(2)}
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

const CONFIDENCE_FACTORS: Array<{
  key: keyof ConfidenceBreakdown;
  label: string;
  weight: string;
  hint: string;
}> = [
  {
    key: 'topScore',
    label: 'Top chunk',
    weight: '35%',
    hint: 'Cross-encoder score of the single best chunk',
  },
  {
    key: 'groundedness',
    label: 'Groundedness',
    weight: '30%',
    hint: 'Share of answer sentences whose content words appear in the retrieved context',
  },
  {
    key: 'meanScore',
    label: 'Mean chunk',
    weight: '15%',
    hint: 'Average score across the chunks sent to the model',
  },
  {
    key: 'retrievalAgreement',
    label: 'Arm agreement',
    weight: '10%',
    hint: 'Overlap between what dense search and keyword search independently found',
  },
  {
    key: 'contextCoverage',
    label: 'Query coverage',
    weight: '10%',
    hint: 'How much of the question’s vocabulary appears in the context',
  },
];

export function ConfidencePanel({ confidence }: { confidence: ConfidenceBreakdown | null }) {
  const tone = confidence ? scoreTone(confidence.overall) : 'neutral';

  return (
    <Card>
      <CardHeader
        title="Confidence"
        icon={Gauge}
        description={
          confidence
            ? `Threshold ${formatPercent(confidence.threshold)} · ${confidence.sufficient ? 'cleared' : 'not met'}`
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
              <Meter value={confidence.overall} tone={tone === 'neutral' ? 'success' : tone} />
            </div>

            <div className="space-y-2 border-t border-border pt-2">
              {CONFIDENCE_FACTORS.map((factor) => {
                const value = confidence[factor.key] as number;
                return (
                  <Tooltip key={factor.key} content={factor.hint} side="left">
                    <div>
                      <div className="flex items-baseline justify-between gap-2">
                        <span className="text-2xs text-ink-secondary">
                          {factor.label}
                          <span className="ml-1 text-ink-tertiary">{factor.weight}</span>
                        </span>
                        <span className="font-mono text-2xs text-ink">{value.toFixed(3)}</span>
                      </div>
                      <Meter value={value} tone="neutral" className="mt-1 h-1" />
                    </div>
                  </Tooltip>
                );
              })}
            </div>
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
