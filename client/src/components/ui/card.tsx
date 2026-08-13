import { forwardRef } from 'react';
import { cn } from '@/lib/utils';

export const Card = forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn('rounded-lg border border-border bg-card', className)} {...props} />
  ),
);
Card.displayName = 'Card';

/** Header row. `action` sits flush right for controls like copy/export. */
export function CardHeader({
  title,
  description,
  icon: Icon,
  action,
  className,
}: {
  title: React.ReactNode;
  description?: React.ReactNode;
  icon?: React.ComponentType<{ className?: string }>;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        'flex items-start justify-between gap-3 border-b border-border px-4 py-3',
        className,
      )}
    >
      <div className="flex min-w-0 items-start gap-2.5">
        {Icon ? <Icon className="mt-0.5 size-4 shrink-0 text-ink-tertiary" /> : null}
        <div className="min-w-0">
          <h2 className="truncate text-sm font-medium text-ink">{title}</h2>
          {description ? (
            <p className="mt-0.5 text-xs text-ink-secondary">{description}</p>
          ) : null}
        </div>
      </div>
      {action ? <div className="flex shrink-0 items-center gap-1">{action}</div> : null}
    </div>
  );
}

export function CardContent({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('p-4', className)} {...props} />;
}

/** Compact metric tile used across the dashboard and analytics pages. */
export function StatCard({
  label,
  value,
  hint,
  icon: Icon,
  tone = 'neutral',
}: {
  label: string;
  value: React.ReactNode;
  hint?: React.ReactNode;
  icon?: React.ComponentType<{ className?: string }>;
  tone?: 'neutral' | 'success' | 'warning' | 'danger';
}) {
  const toneClass = {
    neutral: 'text-ink',
    success: 'text-success',
    warning: 'text-warning',
    danger: 'text-danger',
  }[tone];

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between gap-2">
        <span className="label">{label}</span>
        {Icon ? <Icon className="size-4 text-ink-tertiary" /> : null}
      </div>
      <div className={cn('mt-2 text-2xl font-semibold tracking-tight', toneClass)}>{value}</div>
      {hint ? <div className="mt-1 text-xs text-ink-secondary">{hint}</div> : null}
    </Card>
  );
}
