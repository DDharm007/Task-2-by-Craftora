/**
 * Dashboard — the state of the index and the health of the system.
 *
 * Answers "what is in the corpus and is everything up?" without requiring a
 * terminal.
 */
import { useQuery } from '@tanstack/react-query';
import {
  Boxes,
  Circle,
  Database,
  FileStack,
  Languages,
  Layers,
  RefreshCw,
  Ruler,
  Server,
  Activity,
} from 'lucide-react';
import type { ComponentHealth } from '@voxrag/shared';
import { languageName } from '@voxrag/shared';
import { fetchHealth, fetchStats } from '@/lib/api';
import { cn, formatCompact, formatMs, formatNumber, formatRelative } from '@/lib/utils';
import { Card, CardHeader, CardContent, StatCard } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge, EmptyState, Meter, Skeleton } from '@/components/ui/primitives';

function HealthRow({ component }: { component: ComponentHealth }) {
  const tone =
    component.status === 'ok'
      ? 'text-success'
      : component.status === 'degraded'
        ? 'text-warning'
        : 'text-danger';

  return (
    <div className="flex items-start gap-2.5 border-b border-border py-2.5 last:border-0">
      <Circle className={cn('mt-1 size-2 shrink-0 fill-current', tone)} />
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium text-ink">{component.name.replace(/_/g, ' ')}</p>
        <p className="truncate font-mono text-2xs text-ink-secondary" title={component.detail}>
          {component.detail}
        </p>
      </div>
      {component.latencyMs !== null ? (
        <span className="shrink-0 font-mono text-2xs text-ink-tertiary">
          {formatMs(component.latencyMs)}
        </span>
      ) : null}
    </div>
  );
}

/** Horizontal distribution list used for languages and chunk strategies. */
function Distribution({
  items,
  total,
  labelOf,
}: {
  items: Array<{ label: string; count: number }>;
  total: number;
  labelOf?: (label: string) => string;
}) {
  if (items.length === 0) {
    return <p className="text-xs text-ink-tertiary">No data yet.</p>;
  }

  return (
    <ul className="space-y-2.5">
      {items.map((item) => (
        <li key={item.label}>
          <div className="flex items-baseline justify-between gap-2">
            <span className="truncate text-xs text-ink">
              {labelOf ? labelOf(item.label) : item.label}
            </span>
            <span className="shrink-0 font-mono text-2xs text-ink-secondary">
              {formatNumber(item.count)}
              <span className="ml-1 text-ink-tertiary">
                {total > 0 ? `${((item.count / total) * 100).toFixed(0)}%` : ''}
              </span>
            </span>
          </div>
          <Meter value={total > 0 ? item.count / total : 0} tone="neutral" className="mt-1 h-1" />
        </li>
      ))}
    </ul>
  );
}

export function DashboardPage() {
  const statsQuery = useQuery({
    queryKey: ['stats', 15],
    queryFn: () => fetchStats(15),
    refetchInterval: 20_000,
  });

  const healthQuery = useQuery({
    queryKey: ['health', 'deep'],
    queryFn: () => fetchHealth(false),
    refetchInterval: 30_000,
  });

  const index = statsQuery.data?.index;
  const analytics = statsQuery.data?.analytics;
  const totalVectors = index?.vectors ?? 0;

  return (
    <div className="mx-auto w-full max-w-[1400px] space-y-6 p-4 lg:p-6">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-medium">Index overview</h2>
          <p className="text-xs text-ink-secondary">
            {index?.indexed
              ? `${index.collection} · ${index.vectorStore} · last indexed ${index.lastIndexedAt ? formatRelative(index.lastIndexedAt) : '—'}`
              : 'No vectors indexed yet'}
          </p>
        </div>
        <Button
          variant="secondary"
          size="sm"
          onClick={() => {
            void statsQuery.refetch();
            void healthQuery.refetch();
          }}
          loading={statsQuery.isRefetching}
        >
          <RefreshCw />
          Refresh
        </Button>
      </div>

      {statsQuery.isLoading ? (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
          {Array.from({ length: 6 }, (_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
      ) : !index?.indexed ? (
        <Card>
          <CardContent className="p-0">
            <EmptyState
              icon={Database}
              title="The index is empty"
              description={
                <>
                  Run <code className="font-mono">npm run dataset:download</code> then{' '}
                  <code className="font-mono">npm run index</code> to populate the vector store.
                </>
              }
            />
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
          <StatCard
            label="Documents"
            value={formatCompact(index.documents)}
            hint="source passages"
            icon={FileStack}
          />
          <StatCard
            label="Vectors"
            value={formatCompact(index.vectors)}
            hint={`${index.embeddingDimensions}-dim`}
            icon={Boxes}
          />
          <StatCard
            label="Chunks"
            value={formatCompact(index.chunks)}
            hint="retrievable"
            icon={Layers}
          />
          <StatCard
            label="Avg chunk"
            value={`${index.averageChunkTokens}t`}
            hint={`${formatNumber(index.averageChunkSizeChars)} chars`}
            icon={Ruler}
          />
          <StatCard
            label="Requests"
            value={formatCompact(analytics?.totalRequests ?? 0)}
            hint={`${analytics?.blockedRequests ?? 0} blocked`}
            icon={Activity}
          />
          <StatCard
            label="Avg latency"
            value={formatMs(analytics?.averageLatencyMs ?? 0)}
            hint={`p95 ${formatMs(analytics?.latency.total.p95 ?? 0)}`}
            icon={Server}
          />
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-3">
        <Card>
          <CardHeader
            title="Languages"
            icon={Languages}
            description="Vectors per language in the corpus"
          />
          <CardContent>
            <Distribution
              items={(index?.languages ?? []).map((entry) => ({
                label: entry.language,
                count: entry.count,
              }))}
              total={totalVectors}
              labelOf={languageName}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader
            title="Chunking strategies"
            icon={Layers}
            description="How each vector was produced"
          />
          <CardContent>
            <Distribution
              items={(index?.strategies ?? []).map((entry) => ({
                label: entry.strategy,
                count: entry.count,
              }))}
              total={totalVectors}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader
            title="System health"
            icon={Server}
            description={healthQuery.data ? `status: ${healthQuery.data.status}` : 'checking…'}
            action={
              healthQuery.data ? (
                <Badge
                  tone={
                    healthQuery.data.status === 'ok'
                      ? 'success'
                      : healthQuery.data.status === 'degraded'
                        ? 'warning'
                        : 'danger'
                  }
                >
                  {healthQuery.data.status}
                </Badge>
              ) : null
            }
          />
          <CardContent className="py-0">
            {healthQuery.data ? (
              healthQuery.data.components.map((component) => (
                <HealthRow key={component.name} component={component} />
              ))
            ) : (
              <div className="space-y-2 py-3">
                {Array.from({ length: 4 }, (_, i) => (
                  <Skeleton key={i} className="h-8" />
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader
          title="Recent requests"
          icon={Activity}
          description={`${analytics?.recent.length ?? 0} most recent, newest first`}
        />
        <CardContent className="p-0">
          {!analytics || analytics.recent.length === 0 ? (
            <EmptyState
              icon={Activity}
              title="No requests yet"
              description="Ask a question in the console and it will appear here."
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border text-ink-secondary">
                    <th className="px-4 py-2 text-left font-medium">Query</th>
                    <th className="px-2 py-2 text-left font-medium">Status</th>
                    <th className="px-2 py-2 text-right font-medium">Confidence</th>
                    <th className="px-2 py-2 text-right font-medium">Chunks</th>
                    <th className="px-2 py-2 text-right font-medium">Tokens</th>
                    <th className="px-2 py-2 text-right font-medium">Latency</th>
                    <th className="px-4 py-2 text-right font-medium">When</th>
                  </tr>
                </thead>
                <tbody>
                  {analytics.recent.map((entry) => (
                    <tr key={entry.requestId} className="border-b border-border last:border-0">
                      <td className="max-w-sm truncate px-4 py-2 text-ink" title={entry.query}>
                        {entry.voice ? (
                          <Badge tone="neutral" className="mr-1.5">
                            voice
                          </Badge>
                        ) : null}
                        {entry.query}
                      </td>
                      <td className="px-2 py-2">
                        <Badge
                          tone={
                            entry.status === 'answered'
                              ? 'success'
                              : entry.status === 'blocked'
                                ? 'danger'
                                : 'warning'
                          }
                        >
                          {entry.status.replace(/_/g, ' ')}
                        </Badge>
                      </td>
                      <td className="px-2 py-2 text-right font-mono text-ink-secondary">
                        {(entry.confidence * 100).toFixed(0)}%
                      </td>
                      <td className="px-2 py-2 text-right font-mono text-ink-secondary">
                        {entry.chunkCount}
                      </td>
                      <td className="px-2 py-2 text-right font-mono text-ink-secondary">
                        {formatNumber(entry.tokensUsed)}
                      </td>
                      <td className="px-2 py-2 text-right font-mono text-ink-secondary">
                        {formatMs(entry.totalLatencyMs)}
                      </td>
                      <td className="px-4 py-2 text-right text-ink-tertiary">
                        {formatRelative(entry.createdAt)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
