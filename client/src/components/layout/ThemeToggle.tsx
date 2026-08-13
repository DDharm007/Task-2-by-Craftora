/**
 * Theme picker: Light, Dark, and "HHGoa'26 Exclusive".
 *
 * The first two are ordinary preference switches. Exclusive is deliberately
 * not — it's a novelty, and the whole point of it is that people want to
 * click it, so its row gets a shimmering tri-colour swatch, a bouncing
 * sparkle badge, and a confetti burst on selection instead of the restrained
 * treatment everything else in this app gets. That imbalance is intentional.
 */
import { useCallback, useEffect, useId, useRef, useState } from 'react';
import * as Popover from '@radix-ui/react-popover';
import { Check, Moon, Palette, PartyPopper, Sparkles, Sun } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { useTheme, type Theme } from '@/hooks/useTheme';

const THEME_META: Record<
  Theme,
  { label: string; description: string; icon: React.ComponentType<{ className?: string }> }
> = {
  light: { label: 'Light', description: 'Bright surfaces, high contrast', icon: Sun },
  dark: { label: 'Dark', description: 'Low-glare, easy at night', icon: Moon },
  exclusive: { label: "HHGoa'26 Exclusive", description: 'Limited-run festival colours', icon: PartyPopper },
};

const CONFETTI_COLORS = ['#036735', '#FEE101', '#FF0080'];
const CONFETTI_COUNT = 14;
const HINT_SEEN_KEY = 'goarag:theme-hint-seen';

/** A tiny pulsing tri-colour dot on the trigger, gone for good after the
    first open — a "there's something here" nudge, not a permanent nag. */
function HintBadge() {
  return (
    <span
      className="absolute -right-0.5 -top-0.5 flex size-2.5"
      aria-hidden="true"
    >
      <span
        className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-75"
        style={{
          backgroundImage: `conic-gradient(${CONFETTI_COLORS[0]}, ${CONFETTI_COLORS[1]}, ${CONFETTI_COLORS[2]}, ${CONFETTI_COLORS[0]})`,
        }}
      />
      <span
        className="relative inline-flex size-2.5 rounded-full border border-white/40"
        style={{
          backgroundImage: `conic-gradient(${CONFETTI_COLORS[0]}, ${CONFETTI_COLORS[1]}, ${CONFETTI_COLORS[2]}, ${CONFETTI_COLORS[0]})`,
        }}
      />
    </span>
  );
}

interface Fleck {
  id: number;
  color: string;
  x: number;
  y: number;
  spin: number;
  delayMs: number;
}

function makeFlecks(): Fleck[] {
  return Array.from({ length: CONFETTI_COUNT }, (_, id) => {
    const angle = (Math.PI * 2 * id) / CONFETTI_COUNT + (Math.random() - 0.5) * 0.6;
    const distance = 26 + Math.random() * 30;
    return {
      id,
      color: CONFETTI_COLORS[id % CONFETTI_COLORS.length] as string,
      x: Math.cos(angle) * distance,
      y: Math.sin(angle) * distance - 10, // biased upward — a burst, not a puddle
      spin: (Math.random() - 0.5) * 540,
      delayMs: Math.random() * 60,
    };
  });
}

/** Confetti flecks launched from the trigger button, self-removing after flight. */
function ConfettiBurst({ onDone }: { onDone: () => void }) {
  const [flecks] = useState(makeFlecks);

  useEffect(() => {
    const timer = setTimeout(onDone, 800);
    return () => clearTimeout(timer);
  }, [onDone]);

  return (
    <div className="pointer-events-none absolute inset-0 overflow-visible" aria-hidden="true">
      {flecks.map((fleck) => (
        <span
          key={fleck.id}
          className="absolute left-1/2 top-1/2 size-1.5 animate-confetti-burst rounded-[1px]"
          style={
            {
              backgroundColor: fleck.color,
              animationDelay: `${fleck.delayMs}ms`,
              '--confetti-x': `${fleck.x}px`,
              '--confetti-y': `${fleck.y}px`,
              '--confetti-spin': `${fleck.spin}deg`,
            } as React.CSSProperties
          }
        />
      ))}
    </div>
  );
}

/** The tri-colour preview swatch — a static conic split with a sheen sweeping across it. */
function ExclusiveSwatch({ className }: { className?: string }) {
  return (
    <span
      className={cn('relative block overflow-hidden rounded-full', className)}
      style={{
        backgroundImage: `conic-gradient(from 90deg, ${CONFETTI_COLORS[0]} 0deg 120deg, ${CONFETTI_COLORS[1]} 120deg 240deg, ${CONFETTI_COLORS[2]} 240deg 360deg)`,
      }}
    >
      <span
        className="absolute inset-0 animate-shimmer"
        style={{
          backgroundImage:
            'linear-gradient(115deg, transparent 30%, rgb(255 255 255 / 0.55) 48%, transparent 66%)',
          backgroundSize: '250% 100%',
        }}
      />
    </span>
  );
}

function ThemeSwatch({ theme }: { theme: Theme }) {
  if (theme === 'exclusive') return <ExclusiveSwatch className="size-8" />;
  const tone = theme === 'dark' ? 'bg-[#0B0D10]' : 'bg-white';
  return (
    <span className={cn('block size-8 rounded-full border border-border-strong', tone)}>
      <span
        className={cn(
          'block size-full rounded-full',
          theme === 'dark' ? 'bg-[radial-gradient(circle_at_35%_35%,#2F6FED_0%,transparent_55%)]' : '',
        )}
      />
    </span>
  );
}

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [open, setOpen] = useState(false);
  const [bursting, setBursting] = useState(false);
  const [hintSeen, setHintSeen] = useState(
    () => typeof window !== 'undefined' && window.localStorage.getItem(HINT_SEEN_KEY) === '1',
  );
  const headingId = useId();
  const clearBurst = useRef(() => setBursting(false));

  const select = useCallback(
    (next: Theme) => {
      if (next === 'exclusive' && theme !== 'exclusive') setBursting(true);
      setTheme(next);
      setOpen(false);
    },
    [theme, setTheme],
  );

  const handleOpenChange = useCallback((next: boolean) => {
    setOpen(next);
    if (next) {
      window.localStorage.setItem(HINT_SEEN_KEY, '1');
      setHintSeen(true);
    }
  }, []);

  const ActiveIcon = THEME_META[theme].icon;

  return (
    <Popover.Root open={open} onOpenChange={handleOpenChange}>
      <Popover.Trigger asChild>
        <Button
          variant="ghost"
          size="icon-sm"
          className={cn('relative', theme === 'exclusive' && 'text-[#FF0080] hover:text-[#FF0080]')}
          aria-label={`Theme: ${THEME_META[theme].label}. Click to change.${hintSeen ? '' : ' A new Exclusive theme is available.'}`}
        >
          <ActiveIcon className={theme === 'exclusive' ? 'animate-pulse' : undefined} />
          {!hintSeen && !bursting ? <HintBadge /> : null}
          {bursting ? <ConfettiBurst onDone={clearBurst.current} /> : null}
        </Button>
      </Popover.Trigger>

      <Popover.Portal>
        <Popover.Content
          align="end"
          sideOffset={8}
          className="z-50 w-64 animate-fade-in rounded-lg border border-border bg-card p-1.5 shadow-popover"
          aria-labelledby={headingId}
        >
          <p id={headingId} className="flex items-center gap-1.5 px-2 py-1.5 text-2xs font-medium uppercase tracking-wider text-ink-secondary">
            <Palette className="size-3" />
            Theme
          </p>

          <div className="space-y-0.5">
            {(Object.keys(THEME_META) as Theme[]).map((key) => {
              const meta = THEME_META[key];
              const selected = theme === key;
              const isExclusive = key === 'exclusive';
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => select(key)}
                  className={cn(
                    'group flex w-full items-center gap-2.5 rounded-md p-2 text-left transition-all duration-150',
                    selected ? 'bg-subtle' : 'hover:bg-subtle hover:scale-[1.02]',
                    isExclusive &&
                      !selected &&
                      'ring-1 ring-inset ring-[#FF0080]/30 hover:ring-[#FF0080]/60 hover:shadow-[0_0_16px_-4px_#FF0080]',
                  )}
                >
                  <ThemeSwatch theme={key} />
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-1.5">
                      <span className="truncate text-xs font-medium text-ink">{meta.label}</span>
                      {isExclusive ? (
                        <Sparkles className="size-3 shrink-0 animate-pulse text-[#FEE101]" />
                      ) : null}
                    </span>
                    <span className="block truncate text-2xs text-ink-secondary">{meta.description}</span>
                  </span>
                  {selected ? (
                    <Check
                      className={cn('size-4 shrink-0', isExclusive ? 'text-[#FF0080]' : 'text-ink')}
                    />
                  ) : null}
                </button>
              );
            })}
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
