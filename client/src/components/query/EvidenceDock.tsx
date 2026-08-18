/**
 * Sticky bottom dock for the two panels you want visible *while* reading an
 * answer rather than after scrolling past it: guardrail verdicts and the
 * latency/token breakdown.
 *
 * Collapsed by default, and that is the point. Both panels together run to
 * roughly 800px; pinned open they would leave almost no room for the answer
 * they are meant to sit alongside. The collapsed bar carries the three facts
 * worth glancing at continuously — did the guardrails pass, how long did it
 * take, what did it cost — and the full detail is one click away.
 *
 * `sticky bottom-0` rather than `fixed`: the dock belongs to the page's scroll
 * container, so it pins to the bottom of the content area without having to
 * know the sidebar's width or the header's height, and it cannot end up
 * overlaying the navigation on a narrow screen.
 */
import { useEffect, useState } from 'react';
import { Activity, ChevronDown, Coins, ShieldCheck, Timer, TriangleAlert, X } from 'lucide-react';
import type { GuardrailReport, LatencyBreakdown, TokenUsage } from '@goarag/shared';
import { cn, formatMs, formatNumber } from '@/lib/utils';
import { Badge } from '@/components/ui/primitives';
import { Button } from '@/components/ui/button';
import { GuardrailPanel, LatencyPanel } from '@/components/query/SignalPanels';

const OPEN_KEY = 'goarag:evidence-dock-open';

interface EvidenceDockProps {
  report: GuardrailReport | null;
  latency: LatencyBreakdown | null;
  usage: TokenUsage | null;
}

/** One summary stat in the collapsed bar. */
function Stat({
  icon: Icon,
  label,
  value,
  tone = 'neutral',
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: React.ReactNode;
  tone?: 'neutral' | 'success' | 'warning' | 'danger';
}) {
  const toneClass = {
    neutral: 'text-ink-tertiary',
    success: 'text-success',
    warning: 'text-warning',
    danger: 'text-danger',
  }[tone];

  return (
    <span className="flex min-w-0 items-center gap-1.5">
      <Icon className={cn('size-3.5 shrink-0', toneClass)} />
      <span className="hidden text-2xs text-ink-secondary sm:inline">{label}</span>
      <span className="truncate font-mono text-xs text-ink">{value}</span>
    </span>
  );
}

export function EvidenceDock({ report, latency, usage }: EvidenceDockProps) {
  const [open, setOpen] = useState(
    () => typeof window !== 'undefined' && window.localStorage.getItem(OPEN_KEY) === '1',
  );

  useEffect(() => {
    window.localStorage.setItem(OPEN_KEY, open ? '1' : '0');
  }, [open]);

  // Escape closes the dock, matching every other transient surface in the app.
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open]);

  // Nothing measured yet — a dock reporting zeroes is worse than no dock.
  if (!report && !latency) return null;

  const passed = report?.results.filter((result) => result.verdict === 'pass').length ?? 0;
  const total = report?.results.length ?? 0;
  const guardTone = report?.blocked ? 'danger' : report?.passed ? 'success' : 'warning';

  return (
    <div
      className={cn(
        // Bleeds past the page's own padding so the bar spans the full content
        // width and reads as chrome rather than as another card in the column.
        'sticky bottom-0 z-20 -mx-4 -mb-4 mt-4 border-t border-border lg:-mx-6 lg:-mb-6',
        'bg-canvas/95 backdrop-blur-sm',
      )}
    >
      {open ? (
        <div className="max-h-[52vh] overflow-y-auto border-b border-border px-4 py-4 lg:px-6">
          <div className="grid gap-4 lg:grid-cols-2">
            <GuardrailPanel report={report} />
            <LatencyPanel latency={latency} usage={usage} />
          </div>
        </div>
      ) : null}

      <div className="flex items-center gap-3 px-4 py-2 lg:px-6">
        <div className="flex min-w-0 flex-1 items-center gap-4 overflow-hidden">
          {report ? (
            <Stat
              icon={report.blocked || !report.passed ? TriangleAlert : ShieldCheck}
              label="Guardrails"
              value={`${passed}/${total}`}
              tone={guardTone}
            />
          ) : null}

          {latency ? (
            <Stat icon={Timer} label="Latency" value={formatMs(latency.total)} />
          ) : null}

          {latency?.timeToFirstToken ? (
            <Stat icon={Activity} label="TTFT" value={formatMs(latency.timeToFirstToken)} />
          ) : null}

          {usage ? (
            <Stat icon={Coins} label="Tokens" value={formatNumber(usage.totalTokens)} />
          ) : null}
        </div>

        {report ? (
          <Badge tone={guardTone === 'danger' ? 'danger' : guardTone === 'warning' ? 'warning' : 'success'}>
            {report.blocked ? 'Blocked' : report.passed ? 'Passed' : 'Warnings'}
          </Badge>
        ) : null}

        <Button
          variant="ghost"
          size="sm"
          onClick={() => setOpen((value) => !value)}
          aria-expanded={open}
          aria-label={open ? 'Hide guardrails and latency detail' : 'Show guardrails and latency detail'}
        >
          {open ? (
            <>
              <X />
              Hide
            </>
          ) : (
            <>
              <ChevronDown className="rotate-180" />
              Detail
            </>
          )}
        </Button>
      </div>
    </div>
  );
}
