/**
 * Analytics — latency distribution and benchmark quality.
 *
 * Percentiles rather than averages: a mean latency hides the tail, and the
 * tail is what users actually experience as "slow".
 */
import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip as ChartTooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Activity, Play, Target, Timer, Coins, Gauge } from 'lucide-react';
import type { BenchmarkResult, LatencyPercentiles, StageLatencyStats } from '@voxrag/shared';
import { fetchStats, runBenchmark } from '@/lib/api';
import { formatMs, formatNumber, formatPercent, themeColor } from '@/lib/utils';
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
  { key: 'generation', label: 'Generation' },
  { key: 'total', label: 'Total' },
];

/** Percentile bars for one stage. */
function PercentileChart({ stats }: { stats: LatencyPercentiles }) {
  const data = PERCENTILE_KEYS.map((key) => ({ name: key.toUpperCase(), value: stats[key] }));

  if (stats.count === 0) {
    return (
      <div className="flex h-40 items-center justify-center text-xs text-ink-tertiary">
        No samples yet
      </div>
    );
  }

  // Recharts takes literal colour values, not Tailwind classes, so these are
  // read straight from the active theme's CSS variables. Re-reading happens
  // on every render of this component; AnalyticsPage subscribes to
  // `useTheme()` precisely so a theme switch forces that render.
  const gridColor = themeColor('border');
  const tickColor = themeColor('ink-secondary');
  const cardColor = themeColor('card');
  const inkColor = themeColor('ink');
  const tailColor = themeColor('warning');

  return (
    <ResponsiveContainer width="100%" height={180}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid strokeDasharray="2 4" stroke={gridColor} vertical={false} />
        <XAxis dataKey="name" tick={{ fontSize: 11, fill: tickColor }} axisLine={false} tickLine={false} />
        <YAxis
          tick={{ fontSize: 11, fill: tickColor }}
          axisLine={false}
          tickLine={false}
          tickFormatter={(value: number) => (value >= 1000 ? `${(value / 1000).toFixed(1)}s` : `${value}`)}
        />
        <ChartTooltip
          cursor={{ fill: gridColor }}
          contentStyle={{
            background: cardColor,
            border: `1px solid ${gridColor}`,
            borderRadius: 6,
            fontSize: 12,
            color: inkColor,
            boxShadow: '0 8px 24px -4px rgb(0 0 0 / 0.24)',
          }}
          labelStyle={{ color: inkColor }}
          formatter={(value: number) => [formatMs(value), 'Latency']}
        />
        <Bar dataKey="value" radius={[2, 2, 0, 0]} maxBarSize={48}>
          {data.map((entry) => (
            // The tail is what matters — shade p95+ to make it stand out.
            <Cell key={entry.name} fill={entry.name === 'P99' || entry.name === 'P100' ? tailColor : inkColor} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
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
              <Meter value={metric.value} tone="neutral" className="mt-1 h-1" />
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
  // Forces a re-render on theme change — see the comment in PercentileChart.
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
              <StatCard
                label="Avg latency"
                value={formatMs(analytics?.averageLatencyMs ?? 0)}
                hint={`p95 ${formatMs(analytics?.latency.total.p95 ?? 0)}`}
                icon={Timer}
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
            title="Total latency distribution"
            icon={Timer}
            description={`${analytics?.latency.total.count ?? 0} samples · P99 and P100 highlighted`}
          />
          <CardContent>
            {analytics ? (
              <PercentileChart stats={analytics.latency.total} />
            ) : (
              <Skeleton className="h-40" />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader title="Per-stage percentiles" description="All values in milliseconds" />
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
                  description="Percentiles across the sampled queries"
                />
                <CardContent>
                  <PercentileChart stats={benchmark.data.latency.total} />
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
