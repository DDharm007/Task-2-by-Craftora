/**
 * Application shell: fixed sidebar, top bar, scrolling content area.
 *
 * The sidebar collapses to a slide-over below `lg`, so the console remains
 * usable on a phone without a separate mobile layout. Independently, on `lg`
 * and up it can be toggled down to a narrow icon rail — a desktop preference
 * (more room for the console's two-column layout) rather than a responsive
 * fallback, so the two collapse behaviours are driven by separate state and
 * the rail only ever applies at the `lg:` breakpoint.
 */
import { useEffect, useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  AudioLines,
  BarChart3,
  LayoutDashboard,
  Menu,
  PanelLeftClose,
  PanelLeftOpen,
  X,
  Circle,
} from 'lucide-react';
import type { HealthResponse } from '@goarag/shared';
import { fetchHealth } from '@/lib/api';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Tooltip } from '@/components/ui/primitives';
import { ThemeToggle } from '@/components/layout/ThemeToggle';
import { AmbientGlow } from '@/components/layout/AmbientGlow';
import { useTheme } from '@/hooks/useTheme';

const COLLAPSED_KEY = 'goarag:sidebar-collapsed';
const RAIL_WIDTH = 'lg:w-[76px]';
const FULL_WIDTH = 'lg:w-60';

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
  const { theme } = useTheme();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(
    () => typeof window !== 'undefined' && window.localStorage.getItem(COLLAPSED_KEY) === '1',
  );
  const location = useLocation();

  // Close the slide-over whenever navigation happens.
  useEffect(() => setMobileOpen(false), [location.pathname]);

  useEffect(() => {
    window.localStorage.setItem(COLLAPSED_KEY, collapsed ? '1' : '0');
  }, [collapsed]);

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
          // White sidebar against the warm paper canvas, rather than the other
          // way round — it gives the tinted active row something to sit on.
          // Width is `w-60` unconditionally below `lg` (the collapse toggle is
          // a desktop-only affordance, see the button below) and switches
          // between the rail and full widths at `lg` based on `collapsed`.
          'fixed inset-y-0 left-0 z-40 flex w-60 flex-col overflow-hidden border-r border-border bg-card transition-[transform,width] duration-150 lg:static lg:translate-x-0',
          collapsed ? RAIL_WIDTH : FULL_WIDTH,
          mobileOpen ? 'translate-x-0' : '-translate-x-full',
        )}
      >
        {/* `relative` + an absolutely-positioned close button, rather than
            `justify-between` with a `lg:hidden` sibling: with only one
            visible child, `justify-between` collapses to flex-start and the
            lockup pins to the left instead of centring — which is exactly
            what was happening. Positioning the (mobile-only) close button out
            of flow lets the wordmark center in the row unconditionally. */}
        <div className="relative flex h-16 shrink-0 items-center justify-center border-b border-border px-4">
          <span
            className={cn(
              'flex items-center justify-center',
              collapsed ? 'lg:flex-col lg:leading-[1.15]' : 'gap-0.5 leading-none',
            )}
          >
            <span
              className={cn(
                'font-bold tracking-tight',
                theme === 'exclusive' ? 'text-[#FF0080]' : 'text-ink',
                collapsed ? 'text-2xl lg:text-xl' : 'text-[1.35rem]',
              )}
              style={theme === 'exclusive' ? { WebkitTextStroke: '1px white' } : undefined}
            >
              Goa
            </span>
            <span
              className={cn(
                'font-bold tracking-tight text-brand',
                collapsed ? 'text-2xl lg:text-xl' : 'text-[1.35rem]',
              )}
            >
              RAG
            </span>
          </span>
          <Button
            variant="ghost"
            size="icon-sm"
            className="absolute right-4 top-1/2 -translate-y-1/2 lg:hidden"
            onClick={() => setMobileOpen(false)}
            aria-label="Close navigation"
          >
            <X />
          </Button>
        </div>

        <nav className="flex-1 space-y-0.5 p-2">
          {NAV_ITEMS.map((item) => {
            const link = (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                // The active row is the one place the warm brand colour
                // appears: a soft tint behind it and the icon picked out in
                // full strength. Everything else in the sidebar stays
                // neutral, so "you are here" is the only thing colour says.
                className={({ isActive }) =>
                  cn(
                    'group flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm transition-colors',
                    collapsed && 'justify-center',
                    isActive
                      ? 'bg-brand-subtle font-medium text-ink'
                      : 'text-ink-secondary hover:bg-subtle hover:text-ink',
                  )
                }
              >
                {({ isActive }: { isActive: boolean }) => (
                  <>
                    <item.icon
                      className={cn('size-4 shrink-0', isActive && 'text-brand')}
                    />
                    {!collapsed ? (
                      <span className="min-w-0">
                        <span className="block truncate">{item.label}</span>
                        <span className="block truncate text-2xs text-ink-tertiary">
                          {item.description}
                        </span>
                      </span>
                    ) : null}
                  </>
                )}
              </NavLink>
            );
            // The label moves from inline text to a tooltip when collapsed —
            // it has to keep existing somewhere, or the rail becomes a row of
            // unlabelled icons no one can identify without clicking through.
            return collapsed ? (
              <Tooltip key={item.to} content={item.label} side="right">
                {link}
              </Tooltip>
            ) : (
              link
            );
          })}
        </nav>

        {/* Collapse toggle — desktop only; the mobile slide-over always opens
            full width, so this control has no meaning below `lg`. */}
        <div className="hidden border-t border-border p-2 lg:block">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setCollapsed((value) => !value)}
            aria-pressed={collapsed}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            className={cn('w-full', collapsed ? 'justify-center px-0' : 'justify-start')}
          >
            {collapsed ? <PanelLeftOpen /> : <PanelLeftClose />}
            {!collapsed ? 'Collapse' : null}
          </Button>
        </div>

        {!collapsed ? (
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
        ) : null}
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
            {/* A location label rather than display copy, so it stays on the
                UI face while card and page headings take the serif. */}
            <h1 className="font-sans text-sm font-medium tracking-normal">
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
