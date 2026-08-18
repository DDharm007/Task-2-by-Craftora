/**
 * Two-position lever toggle: HHGoa'26 Exclusive ← → Light.
 *
 * Feels like a real physical switch — the thumb squishes on press, then
 * stretches through the travel arc and snaps back with a soft bounce.
 * All animation is CSS; no JS timers or RAF loops.
 */
import { useId, useState } from 'react';
import { useTheme } from '@/hooks/useTheme';
import { cn } from '@/lib/utils';

/* ─── keyframes injected once as a plain <style> tag ─── */
const LEVER_STYLES = `
@keyframes lever-squish {
  0%   { transform: scaleX(1)   scaleY(1); }
  20%  { transform: scaleX(0.72) scaleY(1.28); }
  55%  { transform: scaleX(1.18) scaleY(0.88); }
  75%  { transform: scaleX(0.94) scaleY(1.06); }
  100% { transform: scaleX(1)   scaleY(1); }
}
`;

let styleInjected = false;
function injectStyle() {
  if (styleInjected || typeof document === 'undefined') return;
  const el = document.createElement('style');
  el.textContent = LEVER_STYLES;
  document.head.appendChild(el);
  styleInjected = true;
}

/* ─── colours ─── */
const HHG_GREEN  = '#036834';
const HHG_PINK   = '#FF0080';
const HHG_YELLOW = '#FEE101';

export function ThemeToggle() {
  injectStyle();

  const { theme, setTheme } = useTheme();
  const id = useId();
  const [clickCount, setClickCount] = useState(0);

  /* only two positions: exclusive (left) and light (right) */
  const isExclusive = theme === 'exclusive';

  function toggle() {
    setTheme(isExclusive ? 'light' : 'exclusive');
    setClickCount((n) => n + 1);
  }

  return (
    <div className="flex items-center gap-2" role="group" aria-label="Theme switcher">

      {/* Left label — HHGoa'26 */}
      <span
        className={cn(
          'select-none text-[10px] font-semibold tracking-wide transition-all duration-300',
          isExclusive ? 'opacity-100' : 'opacity-35',
        )}
        style={{ color: isExclusive ? HHG_PINK : undefined }}
        aria-hidden="true"
      >
        HHGoa&apos;26
      </span>

      {/* ── Track ── */}
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={!isExclusive}
        aria-label={isExclusive ? "Switch to Light theme" : "Switch to HHGoa'26 theme"}
        onClick={toggle}
        className={cn(
          'relative h-6 w-11 shrink-0 cursor-pointer rounded-full border transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2',
        )}
        style={{
          /* track colour shifts between the two themes */
          backgroundColor: isExclusive ? HHG_GREEN : 'hsl(0 0% 90%)',
          borderColor:     isExclusive ? '#024f28' : 'hsl(0 0% 78%)',
          boxShadow: isExclusive
            ? `inset 0 1px 4px rgba(0,0,0,0.45), 0 0 0 1px ${HHG_GREEN}`
            : 'inset 0 1px 3px rgba(0,0,0,0.15)',
          focusVisibleRingColor: isExclusive ? HHG_PINK : undefined,
        } as React.CSSProperties}
      >
        {/* ── Thumb — re-keyed on every click to restart the CSS animation ── */}
        <span
          key={clickCount}
          aria-hidden="true"
          className={cn(
            'absolute top-0.5 flex size-5 items-center justify-center rounded-full shadow-md transition-all',
          )}
          style={{
            /* slide: left=exclusive, right=light */
            left: isExclusive ? '2px' : 'calc(100% - 22px)',
            /* spring-eased travel */
            transitionProperty: 'left, background-color, box-shadow',
            transitionDuration: '320ms',
            transitionTimingFunction: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
            background: isExclusive
              ? `radial-gradient(circle at 38% 38%, ${HHG_YELLOW}, #d4a500)`
              : 'radial-gradient(circle at 38% 38%, #ffffff, #e2e2e2)',
            boxShadow: isExclusive
              ? `0 1px 4px rgba(0,0,0,0.5), 0 0 6px 1px ${HHG_YELLOW}55`
              : '0 1px 3px rgba(0,0,0,0.25)',
            /* squish-stretch plays on every click */
            animation: 'lever-squish 420ms cubic-bezier(0.34, 1.56, 0.64, 1)',
            /* re-trigger by forcing new animation on each render — we key
               the animation name so React always writes it fresh */
          }}
        >
          {/* tiny icon inside thumb */}
          {isExclusive ? (
            /* festival spark */
            <svg viewBox="0 0 12 12" width="9" height="9" fill="none" aria-hidden="true">
              <path
                d="M6 1 L6.6 4.5 L10 4.5 L7.2 6.8 L8.1 10 L6 7.8 L3.9 10 L4.8 6.8 L2 4.5 L5.4 4.5 Z"
                fill={HHG_GREEN}
                opacity="0.85"
              />
            </svg>
          ) : (
            /* sun rays */
            <svg viewBox="0 0 12 12" width="9" height="9" fill="none" aria-hidden="true">
              <circle cx="6" cy="6" r="2.4" fill="#f59e0b" />
              {[0,45,90,135,180,225,270,315].map((deg) => (
                <line
                  key={deg}
                  x1="6" y1="6"
                  x2={6 + Math.cos((deg * Math.PI) / 180) * 4.5}
                  y2={6 + Math.sin((deg * Math.PI) / 180) * 4.5}
                  stroke="#f59e0b"
                  strokeWidth="1"
                  strokeLinecap="round"
                />
              ))}
            </svg>
          )}
        </span>

        {/* festival-colour inner glow when exclusive */}
        {isExclusive && (
          <span
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 rounded-full"
            style={{
              background: `linear-gradient(90deg, ${HHG_PINK}22 0%, transparent 60%)`,
            }}
          />
        )}
      </button>

      {/* Right label — Light */}
      <span
        className={cn(
          'select-none text-[10px] font-semibold tracking-wide transition-all duration-300',
          !isExclusive ? 'opacity-100' : 'opacity-35',
        )}
        aria-hidden="true"
      >
        Light
      </span>
    </div>
  );
}
