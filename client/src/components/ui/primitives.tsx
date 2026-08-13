import { forwardRef } from 'react';
import * as TooltipPrimitive from '@radix-ui/react-tooltip';
import * as SwitchPrimitive from '@radix-ui/react-switch';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

// ─── Badge ───────────────────────────────────────────────────────────────────

const badgeVariants = cva(
  'inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-2xs font-medium [&_svg]:size-3',
  {
    variants: {
      tone: {
        neutral: 'border-border bg-subtle text-ink-secondary',
        success: 'border-success-border bg-success-subtle text-success',
        warning: 'border-warning-border bg-warning-subtle text-warning',
        danger: 'border-danger-border bg-danger-subtle text-danger',
        solid: 'border-transparent bg-accent text-white',
      },
    },
    defaultVariants: { tone: 'neutral' },
  },
);

/**
 * forwardRef because Badge is used as a Radix Tooltip trigger via `asChild`.
 * Radix passes a ref to position the popper against the trigger; a plain
 * function component silently drops it and the tooltip misplaces itself.
 */
export const Badge = forwardRef<
  HTMLSpanElement,
  React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>
>(({ className, tone, ...props }, ref) => (
  <span ref={ref} className={cn(badgeVariants({ tone }), className)} {...props} />
));
Badge.displayName = 'Badge';

// ─── Tooltip ─────────────────────────────────────────────────────────────────

export const TooltipProvider = TooltipPrimitive.Provider;

export function Tooltip({
  content,
  children,
  side = 'top',
}: {
  content: React.ReactNode;
  children: React.ReactNode;
  side?: 'top' | 'right' | 'bottom' | 'left';
}) {
  if (!content) return <>{children}</>;
  return (
    <TooltipPrimitive.Root delayDuration={200}>
      <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
      <TooltipPrimitive.Portal>
        <TooltipPrimitive.Content
          side={side}
          sideOffset={6}
          className="z-50 max-w-xs animate-fade-in rounded border border-border bg-card px-2 py-1 text-xs text-ink shadow-popover"
        >
          {content}
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    </TooltipPrimitive.Root>
  );
}

// ─── Switch ──────────────────────────────────────────────────────────────────

export function Switch({
  checked,
  onCheckedChange,
  disabled,
  label,
  description,
}: {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  disabled?: boolean;
  label: string;
  description?: string;
}) {
  return (
    <label className={cn('flex items-start justify-between gap-3', disabled && 'opacity-50')}>
      <span className="min-w-0">
        <span className="block text-xs font-medium text-ink">{label}</span>
        {description ? (
          <span className="mt-0.5 block text-2xs text-ink-secondary">{description}</span>
        ) : null}
      </span>
      <SwitchPrimitive.Root
        checked={checked}
        onCheckedChange={onCheckedChange}
        disabled={disabled}
        className="relative mt-0.5 h-4 w-7 shrink-0 rounded-full border border-border bg-border transition-colors data-[state=checked]:border-accent data-[state=checked]:bg-accent"
      >
        <SwitchPrimitive.Thumb className="block size-3 translate-x-0.5 rounded-full bg-white transition-transform data-[state=checked]:translate-x-3.5" />
      </SwitchPrimitive.Root>
    </label>
  );
}

// ─── Meter ───────────────────────────────────────────────────────────────────

/** Horizontal bar for a 0-1 value. Colour encodes the tone, never decoration. */
export function Meter({
  value,
  tone = 'neutral',
  className,
}: {
  value: number;
  tone?: 'neutral' | 'success' | 'warning' | 'danger';
  className?: string;
}) {
  const width = `${Math.max(0, Math.min(1, value)) * 100}%`;
  const fill = {
    neutral: 'bg-ink',
    success: 'bg-success',
    warning: 'bg-warning',
    danger: 'bg-danger',
  }[tone];

  return (
    <div className={cn('h-1.5 w-full overflow-hidden rounded-full bg-border/60', className)}>
      <div className={cn('h-full rounded-full transition-[width] duration-300', fill)} style={{ width }} />
    </div>
  );
}

// ─── Loading bar ─────────────────────────────────────────────────────────────

/** Indeterminate progress used while a request is in flight. */
export function LoadingBar({ active }: { active: boolean }) {
  if (!active) return null;
  return (
    <div className="h-0.5 w-full overflow-hidden bg-border/50">
      <div className="h-full w-1/4 animate-indeterminate bg-accent" />
    </div>
  );
}

// ─── Skeleton ────────────────────────────────────────────────────────────────

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('animate-pulse rounded bg-border/60', className)} />;
}

// ─── Empty state ─────────────────────────────────────────────────────────────

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
}: {
  icon?: React.ComponentType<{ className?: string }>;
  title: string;
  description?: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-12 text-center">
      {Icon ? <Icon className="size-6 text-ink-tertiary" /> : null}
      <p className="mt-3 text-sm font-medium text-ink">{title}</p>
      {description ? (
        <p className="mt-1 max-w-sm text-xs leading-relaxed text-ink-secondary">{description}</p>
      ) : null}
      {action ? <div className="mt-4">{action}</div> : null}
    </div>
  );
}

// ─── Key/value row ───────────────────────────────────────────────────────────

export function KeyValue({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5">
      <dt className="shrink-0 text-xs text-ink-secondary">{label}</dt>
      <dd className={cn('min-w-0 truncate text-right text-xs text-ink', mono && 'font-mono')}>{value}</dd>
    </div>
  );
}
