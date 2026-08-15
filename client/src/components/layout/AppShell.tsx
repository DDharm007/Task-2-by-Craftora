/**
 * Application shell: fixed sidebar, top bar, scrolling content area.
 *
 * The sidebar collapses to a slide-over below `lg`, so the console remains
 * usable on a phone without a separate mobile layout.
 */
import { useEffect, useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { AudioLines, BarChart3, LayoutDashboard, Menu, X, Circle } from 'lucide-react';
import type { HealthResponse } from '@goarag/shared';
import { fetchHealth } from '@/lib/api';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Tooltip } from '@/components/ui/primitives';
import { ThemeToggle } from '@/components/layout/ThemeToggle';
import { AmbientGlow } from '@/components/layout/AmbientGlow';

const NAV_ITEMS = [
  { to: '/', label: 'Console', icon: AudioLines, description: 'Ask by voice or text' },
  { to: '/analytics', label: 'Analytics', icon: BarChart3, description: 'Latency percentiles' },
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard, description: 'Index and corpus' },
];

/** Drop an `org/` prefix for display; the full id stays in the tooltip. */
function shortModelName(model: string): string {
  return model.split('/').pop() || model;
}

function StatusDot({ health }: { health: HealthResponse | undefined }) {
  const status = health?.status ?? 'down';
  const tone =
    status === 'ok' ? 'text-success' : status === 'degraded' ? 'text-warning' : 'text-danger';
  const label =
    status === 'ok'
      ? 'All systems operational'
      : status === 'degraded'
        ? health?.components.find((c) => c.status === 'degraded')?.detail ?? 'Degraded'
        : health?.components.find((c) => c.status === 'down')?.detail ?? 'Unreachable';

  return (
    <Tooltip content={label}>
      <span className="flex items-center gap-1.5 text-xs text-ink-secondary">
        <Circle className={cn('size-2 fill-current', tone)} />
        <span className="hidden sm:inline">{status === 'ok' ? 'Operational' : status}</span>
      </span>
    </Tooltip>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  // Close the slide-over whenever navigation happens.
  useEffect(() => setMobileOpen(false), [location.pathname]);

  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: () => fetchHealth(false),
    refetchInterval: 30_000,
    staleTime: 15_000,
  });

  return (
    <div className="flex h-full">
      {/* `fixed`, so its DOM position doesn't affect layout — but a
          positioned element paints *above* plain non-positioned content
          regardless of z-index, so without the `relative z-10` down on the
          main column below, this would sit on top of the page instead of
          behind it. Renders nothing outside the Exclusive theme. */}
      <AmbientGlow />

      {/* Sidebar */}
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-40 flex w-60 flex-col border-r border-border bg-subtle transition-transform duration-150 lg:static lg:translate-x-0',
          mobileOpen ? 'translate-x-0' : '-translate-x-full',
        )}
      >
        <div className="flex h-16 items-center justify-between border-b border-border px-4">
          <div className="flex items-center gap-2.5">
            {/* The mark is white artwork, so it needs the saturated tile behind
                it — on the light sidebar it would be invisible. This is the one
                place the evidence ramp is decorative rather than a readout. */}
            <div className="spectrum-tile flex size-11 shrink-0 items-center justify-center rounded-lg">
              <img
                src="/goarag-logo.png"
                alt=""
                aria-hidden="true"
                className="size-8 object-contain"
              />
            </div>
            <div className="flex items-baseline">
              <span style={{ fontFamily: "'Yatra One', cursive" }} className="text-2xl font-bold tracking-wide mr-0.5">गोवा</span>
              <span className="text-base font-semibold tracking-tight">RAG</span>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon-sm"
            className="lg:hidden"
            onClick={() => setMobileOpen(false)}
            aria-label="Close navigation"
          >
            <X />
          </Button>
        </div>

        <nav className="flex-1 space-y-0.5 p-2">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                cn(
                  'flex items-start gap-2.5 rounded px-2.5 py-2 text-sm transition-colors',
                  isActive
                    ? 'bg-card font-medium text-ink shadow-xs'
                    : 'text-ink-secondary hover:bg-card/70 hover:text-ink',
                )
              }
            >
              <item.icon className="mt-0.5 size-4 shrink-0" />
              <span className="min-w-0">
                <span className="block truncate">{item.label}</span>
                <span className="block truncate text-2xs text-ink-tertiary">{item.description}</span>
              </span>
            </NavLink>
          ))}
        </nav>

        <div className="border-t border-border p-3">
          <dl className="space-y-1 text-2xs text-ink-secondary">
            <div className="flex justify-between gap-2">
              <dt>Model</dt>
              <dd
                className="truncate font-mono"
                title={health?.models.llm}
              >
                {health ? shortModelName(health.models.llm) : '—'}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt>Embeddings</dt>
              <dd
                className="truncate font-mono"
                title={health?.models.embedding}
              >
                {health ? shortModelName(health.models.embedding) : '—'}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt>Version</dt>
              <dd className="font-mono">{health?.version ?? '—'}</dd>
            </div>
          </dl>
        </div>
      </aside>

      {mobileOpen ? (
        <button
          type="button"
          aria-label="Close navigation overlay"
          className="fixed inset-0 z-30 bg-ink/20 lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      ) : null}

      {/* Main column */}
      <div className="relative z-10 flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex h-16 shrink-0 items-center justify-between gap-3 border-b border-border bg-canvas/95 px-4 backdrop-blur-sm">
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon-sm"
              className="lg:hidden"
              onClick={() => setMobileOpen(true)}
              aria-label="Open navigation"
            >
              <Menu />
            </Button>
            <h1 className="text-sm font-medium">
              {NAV_ITEMS.find((item) =>
                item.to === '/' ? location.pathname === '/' : location.pathname.startsWith(item.to),
              )?.label ?? 'GoaRAG'}
            </h1>
          </div>

          <div className="flex items-center gap-2">
            <StatusDot health={health} />
            <div className="mx-1 h-4 w-px bg-border" />
            <ThemeToggle />
          </div>
        </header>

        <main className="min-h-0 flex-1 overflow-y-auto">{children}</main>
      </div>
    </div>
  );
}
