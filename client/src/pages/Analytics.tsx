/**
 * Analytics — latency distribution and benchmark quality.
 *
 * Percentiles rather than averages: a mean latency hides the tail, and the
 * tail is what users actually experience as "slow".
 */
import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as ChartTooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Activity, Play, Target, Timer, Coins, Gauge } from 'lucide-react';
import type {
  BenchmarkResult,
  LatencyPercentiles,
  RequestLogEntry,
  StageLatencyStats,
} from '@goarag/shared';
import { fetchStats, runBenchmark } from '@/lib/api';
import { cn, formatMs, formatNumber, formatPercent, themeColor } from '@/lib/utils';
import { useTheme } from '@/hooks/useTheme';
import { Card, CardHeader, CardContent, StatCard } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge, EmptyState, KeyValue, Meter, Skeleton, Switch, Tooltip } from '@/components/ui/primitives';

const PERCENTILE_KEYS: Array<keyof LatencyPercentiles> = ['p50', 'p70', 'p95', 'p99', 'p100'];

const STAGE_ROWS: Array<{ key: keyof StageLatencyStats; label: string }> = [
  { key: 'transcription', label: 'Transcription' },
  { key: 'embedding', label: 'Embedding' },
  { key: 'retrieval', label: 'Retrieval' },
  { key: 'reranking', label: 'Reranking' },
  // The 50ms-budget window, sitting between the stages it sums and the
  // generation that is deliberately outside it.
  { key: 'retrievalPath', label: 'Retrieval path' },
  { key: 'generation', label: 'Generation' },
  { key: 'total', label: 'Total (+LLM)' },
];

/** The task spec's budget for query text in → ranked context out. */
const LATENCY_BUDGET_MS = 50;

/** One point on the trace, as handed to the chart. */
interface TracePoint {
  index: number;
  value: number;
  total: number;
  at: string;
}

/** Tooltip for the trace. Styled with app tokens rather than inline colours. */
function TraceTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: TracePoint }>;
}) {
  const point = payload?.[0]?.payload;
  if (!active || !point) return null;
  const over = point.value > LATENCY_BUDGET_MS;

  return (
    <div className="rounded-md border border-border bg-card px-2.5 py-2 shadow-popover">
      <p className="mb-1 text-2xs text-ink-tertiary">{point.at}</p>
      <p className="flex items-baseline gap-1.5">
        <span
          className={cn('font-mono text-sm font-medium tabular-nums', over ? 'text-danger' : 'text-ink')}
        >
          {formatMs(point.value)}
        </span>
        <span className="text-2xs text-ink-secondary">retrieval path</span>
      </p>
      <p className="mt-0.5 font-mono text-2xs text-ink-tertiary tabular-nums">
        {formatMs(point.total)} end to end
      </p>
    </div>
  );
}

/**
 * Retrieval-path latency over recent requests, drawn as a trace rather than as
 * percentile bars.
 *
 * The bar chart this replaced plotted P50/P70/P95/P99/P100 side by side, which
 * is only meaningful once there are enough samples for those to differ — on a
 * fresh server it drew five identical bars from one observation and implied a
 * distribution that did not exist. A time series says something true at every
 * sample count: with one request it is one point, and it gets more informative
 * as requests accumulate rather than less honest.
 *
 * It plots the *retrieval path*, not total latency, because that is the number
 * under a budget. Total is dominated by the LLM — a 1.4s generation against a
 * ~20ms retrieval path would flatten the series it is supposed to show.
 */
function LatencyTrace({ entries }: { entries: RequestLogEntry[] }) {
  // Oldest first: the analytics store hands back newest-first for the table.
  const data = [...entries]
    .reverse()
    .map((entry, index) => ({
      index,
      value: entry.retrievalPathMs,
      total: entry.totalLatencyMs,
      at: new Date(entry.createdAt).toLocaleTimeString(),
    }));

  if (data.length === 0) {
    return (
      <div className="flex h-44 items-center justify-center text-xs text-ink-tertiary">
        No requests yet — run a query on the Console.
      </div>
    );
  }

  // Recharts takes literal colour values, not Tailwind classes, so these are
  // read straight from the active theme's CSS variables. Re-reading happens
  // on every render of this component; AnalyticsPage subscribes to
  // `useTheme()` precisely so a theme switch forces that render.
  const gridColor = themeColor('border');
  const tickColor = themeColor('ink-secondary');
  const inkColor = themeColor('ink');
  const budgetColor = themeColor('danger');

  const peak = Math.max(...data.map((point) => point.value), LATENCY_BUDGET_MS);

  return (
    <ResponsiveContainer width="100%" height={180}>
      {/* No negative left margin: it was pulling the plot area past the
          YAxis's own reserved width, clipping the leading digit off every
          tick label ("50ms" rendering as ")0ms"). `width` on YAxis already
          controls how much gutter the labels get. */}
      <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        <defs>
          {/* The gradient is what makes this read as a trace rather than a
              plotted line — opaque at the baseline, gone by the top. */}
          <linearGradient id="latency-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={inkColor} stopOpacity={0.22} />
            <stop offset="100%" stopColor={inkColor} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="2 4" stroke={gridColor} vertical={false} />
        <XAxis dataKey="index" tick={false} axisLine={false} tickLine={false} height={4} />
        <YAxis
          tick={{ fontSize: 11, fill: tickColor }}
          axisLine={false}
          tickLine={false}
          width={46}
          domain={[0, Math.ceil((peak * 1.15) / 10) * 10]}
          tickFormatter={(value: number) => (value >= 1000 ? `${(value / 1000).toFixed(1)}s` : `${value}ms`)}
        />
        {/* The line the whole pipeline is judged against, so it belongs on the
            chart rather than in prose beside it. */}
        <ReferenceLine
          y={LATENCY_BUDGET_MS}
          stroke={budgetColor}
          strokeDasharray="4 4"
          strokeWidth={1}
          label={{
            value: `${LATENCY_BUDGET_MS}ms budget`,
            position: 'insideTopRight',
            fill: budgetColor,
            fontSize: 10,
          }}
        />
        {/* A component rather than recharts' `contentStyle` props: it inherits
            the app's card tokens directly, so it themes with everything else
            instead of needing its colours passed in one at a time. */}
        <ChartTooltip
          cursor={{ stroke: tickColor, strokeWidth: 1, strokeDasharray: '3 3' }}
          content={<TraceTooltip />}
        />
        <Area
          type="monotone"
          dataKey="value"
          stroke={inkColor}
          strokeWidth={1.6}
          fill="url(#latency-fill)"
          // Dots only once the series is sparse enough for them to read as
          // data points rather than as noise along the line.
          dot={data.length <= 20 ? { r: 2, fill: inkColor, strokeWidth: 0 } : false}
          activeDot={{ r: 3.5, fill: inkColor, strokeWidth: 0 }}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/**
 * Percentiles as horizontal bars against the budget.
 *
 * Used for benchmark results, where there genuinely is a distribution — a
 * 155-query run has a real tail worth seeing. Horizontal rather than vertical
 * because the labels (P50…P100) and the values both read inline, and because
 * the budget becomes a single vertical rule every bar is measured against
 * instead of a number you have to hold in your head.
 */
function PercentileStrip({ stats }: { stats: LatencyPercentiles }) {
  if (stats.count === 0) {
    return <div className="py-6 text-center text-xs text-ink-tertiary">No samples</div>;
  }

  // Scale to the budget *or* the worst value, whichever is larger, so a run
  // that blows the budget still shows how far past it went.
  const scale = Math.max(stats.p100, LATENCY_BUDGET_MS) * 1.05;
  const budgetLeft = (LATENCY_BUDGET_MS / scale) * 100;

  return (
    <div className="space-y-2">
      <div className="relative">
        <span
          className="absolute -top-1 bottom-0 z-10 w-px bg-danger/70"
          style={{ left: `${budgetLeft}%` }}
          aria-hidden
        />
        <div className="space-y-2">
          {PERCENTILE_KEYS.map((key) => {
            const value = stats[key];
            const over = value > LATENCY_BUDGET_MS;
            return (
              <div key={key} className="flex items-center gap-2">
                <span className="w-9 shrink-0 font-mono text-2xs text-ink-secondary">
                  {key.toUpperCase()}
                </span>
                <div className="h-2 min-w-0 flex-1 overflow-hidden rounded-full bg-border/40">
                  <div
                    className={cn('h-full rounded-full', over ? 'bg-danger/70' : 'bg-ink/75')}
                    style={{ width: `${Math.min(100, (value / scale) * 100)}%` }}
                  />
                </div>
                <span
                  className={cn(
                    'w-14 shrink-0 text-right font-mono text-2xs tabular-nums',
                    over ? 'text-danger' : 'text-ink',
                  )}
                >
                  {formatMs(value)}
                </span>
              </div>
            );
          })}
        </div>
      </div>
      <p className="text-2xs text-ink-tertiary">
        {stats.count} queries · red rule marks the {LATENCY_BUDGET_MS}ms budget
      </p>
    </div>
  );
}

/** Ticker-style readout: latest value and how it sits against the median. */
function TraceTicker({ entries }: { entries: RequestLogEntry[] }) {
  if (entries.length === 0) return null;
  const values = entries.map((entry) => entry.retrievalPathMs);
  const latest = values[0] as number;
  const sorted = [...values].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)] as number;
  const delta = latest - median;
  // Below the median is the good direction here, unlike a share price.
  const better = delta <= 0;

  return (
    <div className="flex items-baseline gap-2">
      <span className="font-mono text-lg font-semibold tabular-nums">{formatMs(latest)}</span>
      {entries.length > 1 ? (
        <span
          className={cn(
            'font-mono text-2xs tabular-nums',
            better ? 'text-success' : 'text-warning',
          )}
        >
          {better ? '▼' : '▲'} {formatMs(Math.abs(delta))} vs median
        </span>
      ) : null}
    </div>
  );
}

function QualityCard({ result }: { result: BenchmarkResult }) {
  const metrics = [
    { label: 'Hit rate', value: result.quality.hitRate, hint: 'Queries where a labelled-relevant passage appeared anywhere in the results' },
    { label: 'Recall@5', value: result.quality.recallAt5, hint: 'Share of relevant passages found in the top 5' },
    { label: 'Recall@10', value: result.quality.recallAt10, hint: 'Share of relevant passages found in the top 10' },
    { label: 'Precision@5', value: result.quality.precisionAt5, hint: 'Share of the top 5 that are relevant' },
    { label: 'MRR', value: result.quality.mrr, hint: 'Mean reciprocal rank of the first relevant passage' },
    { label: 'nDCG@5', value: result.quality.ndcgAt5, hint: 'Ranking quality — rewards putting relevant passages first' },
  ];

  return (
    <Card>
      <CardHeader
        title="Retrieval quality"
        icon={Target}
        description={`${result.quality.evaluatedQueries} queries scored against the dataset's is_selected labels`}
        action={<Badge tone="neutral">{result.generationEnabled ? 'with generation' : 'retrieval only'}</Badge>}
      />
      <CardContent className="space-y-3">
        {metrics.map((metric) => (
          <Tooltip key={metric.label} content={metric.hint} side="left">
            <div>
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-xs text-ink-secondary">{metric.label}</span>
                <span className="font-mono text-xs font-medium text-ink">
                  {formatPercent(metric.value, 1)}
                </span>
              </div>
              <Meter value={metric.value} className="mt-1 h-1" />
            </div>
          </Tooltip>
        ))}
      </CardContent>
    </Card>
  );
}

export function AnalyticsPage() {
  const [sampleSize, setSampleSize] = useState(10);
  const [withGeneration, setWithGeneration] = useState(false);
  // Forces a re-render on theme change — see the comment in LatencyTrace.
  useTheme();

  const { data: stats, isLoading } = useQuery({
    queryKey: ['stats', 25],
    queryFn: () => fetchStats(25),
    refetchInterval: 15_000,
  });

  const benchmark = useMutation({
    mutationFn: () =>
      runBenchmark({ sampleSize, generation: withGeneration, concurrency: withGeneration ? 2 : 4 }),
  });

  const analytics = stats?.analytics;
  const sampleCount = analytics?.latency.retrievalPath.count ?? 0;

  return (
    <div className="mx-auto w-full max-w-[1400px] space-y-6 p-4 lg:p-6">
      {/* Live request metrics */}
      <section>
        <div className="mb-3 flex items-baseline justify-between">
          <div>
            <h2 className="text-sm font-medium">Live requests</h2>
            <p className="text-xs text-ink-secondary">
              Rolling window of requests served by this instance.
            </p>
          </div>
          {analytics ? (
            <span className="text-2xs text-ink-tertiary">
              uptime {Math.floor(analytics.uptimeSeconds / 60)}m
            </span>
          ) : null}
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {isLoading ? (
            Array.from({ length: 4 }, (_, i) => <Skeleton key={i} className="h-24" />)
          ) : (
            <>
              <StatCard
                label="Requests"
                value={formatNumber(analytics?.totalRequests ?? 0)}
                hint={`${analytics?.successfulRequests ?? 0} answered · ${analytics?.blockedRequests ?? 0} blocked`}
                icon={Activity}
              />
              {/* Reports the budget window rather than total latency, and at
                  p100 rather than the mean — "under 50ms" is only a claim
                  worth making if the slowest query also clears it. */}
              <StatCard
                label="Retrieval path p100"
                value={formatMs(analytics?.latency.retrievalPath.p100 ?? 0)}
                hint={`p50 ${formatMs(analytics?.latency.retrievalPath.p50 ?? 0)} · p70 ${formatMs(
                  analytics?.latency.retrievalPath.p70 ?? 0,
                )} · budget ${LATENCY_BUDGET_MS}ms`}
                icon={Timer}
                tone={
                  (analytics?.latency.retrievalPath.count ?? 0) === 0
                    ? 'neutral'
                    : (analytics?.latency.retrievalPath.p100 ?? 0) <= LATENCY_BUDGET_MS
                      ? 'success'
                      : 'danger'
                }
              />
              <StatCard
                label="Avg confidence"
                value={formatPercent(analytics?.averageConfidence ?? 0)}
                hint={`${analytics?.lowConfidenceRequests ?? 0} below threshold`}
                icon={Gauge}
              />
              <StatCard
                label="Tokens used"
                value={formatNumber(analytics?.tokensUsed.totalTokens ?? 0)}
                hint={`${formatNumber(analytics?.tokensUsed.promptTokens ?? 0)} prompt · ${formatNumber(analytics?.tokensUsed.completionTokens ?? 0)} completion`}
                icon={Coins}
              />
            </>
          )}
        </div>
      </section>

      {/* Percentiles */}
      <section className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader
            title="Retrieval path"
            icon={Timer}
            description={
              sampleCount === 0
                ? 'Query text in → ranked context out'
                : `Last ${sampleCount} ${sampleCount === 1 ? 'request' : 'requests'} · budget ${LATENCY_BUDGET_MS}ms`
            }
            action={analytics ? <TraceTicker entries={analytics.recent} /> : null}
          />
          <CardContent>
            {analytics ? <LatencyTrace entries={analytics.recent} /> : <Skeleton className="h-44" />}
          </CardContent>
        </Card>

        <Card>
          <CardHeader
            title="Per-stage percentiles"
            description={
              // With one observation every percentile *is* that observation, so
              // a row of six identical numbers is noise dressed as a
              // distribution. Say so rather than letting it be misread.
              sampleCount === 1
                ? 'One request — every percentile is that same observation'
                : 'All values in milliseconds'
            }
          />
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border text-ink-secondary">
                    <th className="px-4 py-2 text-left font-medium">Stage</th>
                    {PERCENTILE_KEYS.map((key) => (
                      <th key={key} className="px-2 py-2 text-right font-medium uppercase">
                        {key}
                      </th>
                    ))}
                    <th className="px-4 py-2 text-right font-medium">Mean</th>
                  </tr>
                </thead>
                <tbody>
                  {STAGE_ROWS.map((row) => {
                    const stage = analytics?.latency[row.key];
                    return (
                      <tr key={row.key} className="border-b border-border last:border-0">
                        <td className="px-4 py-2 text-ink">{row.label}</td>
                        {PERCENTILE_KEYS.map((key) => (
                          <td key={key} className="px-2 py-2 text-right font-mono text-ink-secondary">
                            {stage && stage.count > 0 ? formatMs(stage[key]) : '—'}
                          </td>
                        ))}
                        <td className="px-4 py-2 text-right font-mono text-ink">
                          {stage && stage.count > 0 ? formatMs(stage.mean) : '—'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Benchmark */}
      <section>
        <div className="mb-3">
          <h2 className="text-sm font-medium">Benchmark</h2>
          <p className="text-xs text-ink-secondary">
            Scores retrieval against the dataset&rsquo;s own relevance labels — not just speed.
          </p>
        </div>

        <div className="grid gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
          <Card>
            <CardHeader title="Run" icon={Play} />
            <CardContent className="space-y-4">
              <label className="flex items-center justify-between gap-3 text-xs">
                <span className="font-medium text-ink">Sample size</span>
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={sampleSize}
                  onChange={(event) => setSampleSize(Number(event.target.value))}
                  className="w-20 rounded border border-border bg-canvas px-2 py-1 text-right font-mono text-xs"
                />
              </label>

              <Switch
                label="Include generation"
                description="Runs the LLM per query — accurate but slow"
                checked={withGeneration}
                onCheckedChange={setWithGeneration}
              />

              <Button
                variant="primary"
                className="w-full"
                onClick={() => benchmark.mutate()}
                loading={benchmark.isPending}
              >
                {benchmark.isPending ? 'Running…' : 'Run benchmark'}
              </Button>

              {benchmark.isError ? (
                <p className="text-xs text-danger">{(benchmark.error as Error).message}</p>
              ) : null}

              {benchmark.data ? (
                <dl className="divide-y divide-border border-t border-border pt-1">
                  <KeyValue label="Duration" value={formatMs(benchmark.data.durationMs)} mono />
                  <KeyValue label="Queries" value={String(benchmark.data.sampleSize)} mono />
                  <KeyValue
                    label="Avg confidence"
                    value={formatPercent(benchmark.data.averageConfidence, 1)}
                    mono
                  />
                  <KeyValue label="Embeddings" value={benchmark.data.config.embeddingModel} mono />
                  <KeyValue label="Reranker" value={benchmark.data.config.rerankerProvider} mono />
                  <KeyValue label="Store" value={benchmark.data.config.vectorStore} mono />
                </dl>
              ) : null}
            </CardContent>
          </Card>

          {benchmark.data ? (
            <div className="grid gap-4 sm:grid-cols-2">
              <QualityCard result={benchmark.data} />
              <Card>
                <CardHeader
                  title="Benchmark latency"
                  icon={Timer}
                  description="Retrieval path across the sampled queries"
                  action={
                    <Badge
                      tone={
                        benchmark.data.latency.retrievalPath.p100 <= LATENCY_BUDGET_MS
                          ? 'success'
                          : 'warning'
                      }
                    >
                      {benchmark.data.latency.retrievalPath.p100 <= LATENCY_BUDGET_MS
                        ? 'Within budget'
                        : 'Over at p100'}
                    </Badge>
                  }
                />
                <CardContent>
                  <PercentileStrip stats={benchmark.data.latency.retrievalPath} />
                </CardContent>
              </Card>
              <Card className="sm:col-span-2">
                <CardHeader title="Cases" description="Per-query outcome, worst first" />
                <CardContent className="p-0">
                  <div className="max-h-80 overflow-auto">
                    <table className="w-full text-xs">
                      <thead className="sticky top-0 bg-card">
                        <tr className="border-b border-border text-ink-secondary">
                          <th className="px-4 py-2 text-left font-medium">Query</th>
                          <th className="px-2 py-2 text-left font-medium">Lang</th>
                          <th className="px-2 py-2 text-right font-medium">RR</th>
                          <th className="px-2 py-2 text-right font-medium">Latency</th>
                          <th className="px-4 py-2 text-right font-medium">Hit</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[...benchmark.data.cases]
                          .sort((a, b) => a.reciprocalRank - b.reciprocalRank)
                          .map((item, index) => (
                            <tr key={index} className="border-b border-border last:border-0">
                              <td className="max-w-xs truncate px-4 py-2 text-ink" title={item.query}>
                                {item.query}
                              </td>
                              <td className="px-2 py-2 font-mono text-ink-tertiary">{item.language}</td>
                              <td className="px-2 py-2 text-right font-mono text-ink-secondary">
                                {item.reciprocalRank.toFixed(2)}
                              </td>
                              <td className="px-2 py-2 text-right font-mono text-ink-secondary">
                                {formatMs(item.latencyMs)}
                              </td>
                              <td className="px-4 py-2 text-right">
                                <Badge tone={item.hit ? 'success' : 'danger'}>
                                  {item.hit ? 'hit' : 'miss'}
                                </Badge>
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : (
            <Card>
              <CardContent className="p-0">
                <EmptyState
                  icon={Target}
                  title="No benchmark run yet"
                  description="Run one to measure recall, MRR and nDCG against the dataset's ground-truth relevance labels, alongside per-stage latency percentiles."
                />
              </CardContent>
            </Card>
          )}
        </div>
      </section>
    </div>
  );
}
