import type { Config } from 'tailwindcss';

/**
 * Design tokens.
 *
 * A warm, editorial workspace palette: paper neutrals, black primary actions,
 * and colour reserved for meaningful system state.
 *
 * There is exactly one gradient in the system — the evidence spectrum defined
 * in styles/index.css — and it is a readout, not an ornament: every 0-1
 * confidence the app reports is painted on that shared ramp, so a colour means
 * the same thing in every panel. Outside of it, surfaces are flat. If
 * something is coloured here, it means something.
 */
/**
 * Every colour below is `rgb(var(--x) / <alpha-value>)`, not a literal hex —
 * the actual R/G/B values live as CSS custom properties in styles/index.css,
 * one triplet per theme (`:root` for Light, `[data-theme="dark"]`,
 * `[data-theme="exclusive"]`). Tailwind still generates the same `bg-canvas`,
 * `text-ink-secondary`, `border-danger-border` utilities as before; which
 * theme's values they resolve to is decided at paint time by the
 * `data-theme` attribute on `<html>`, not at build time. `<alpha-value>` is
 * what lets opacity modifiers (`bg-canvas/95`) keep working — a plain
 * `var(--canvas)` hex string can't be sliced like that.
 */
function themedColor(name: string) {
  return `rgb(var(--${name}) / <alpha-value>)`;
}

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: themedColor('canvas'),
        subtle: themedColor('subtle'),
        card: themedColor('card'),
        border: {
          DEFAULT: themedColor('border'),
          strong: themedColor('border-strong'),
        },
        ink: {
          DEFAULT: themedColor('ink'),
          secondary: themedColor('ink-secondary'),
          tertiary: themedColor('ink-tertiary'),
        },
        accent: {
          DEFAULT: themedColor('accent'),
          hover: themedColor('accent-hover'),
          /** Text/icon colour that sits *on* the accent. Light puts white on
              near-black, Dark puts near-black on white — same component, and
              neither has to branch on the theme. */
          ink: themedColor('on-accent'),
        },
        /** The single warm colour: active navigation and the icon inside it.
            Never a button — that's what `accent` is for. */
        brand: {
          DEFAULT: themedColor('brand'),
          subtle: themedColor('brand-subtle'),
        },
        success: {
          DEFAULT: themedColor('success'),
          subtle: themedColor('success-subtle'),
          border: themedColor('success-border'),
        },
        warning: {
          DEFAULT: themedColor('warning'),
          subtle: themedColor('warning-subtle'),
          border: themedColor('warning-border'),
        },
        danger: {
          DEFAULT: themedColor('danger'),
          subtle: themedColor('danger-subtle'),
          border: themedColor('danger-border'),
        },
        /** Fixed brand swatches for the theme picker itself — never resolved
            through a CSS variable, since all three need to render at once
            regardless of which theme is currently active. */
        exclusive: {
          green: '#036735',
          yellow: '#FEE101',
          pink: '#FF0080',
        },
      },
      fontFamily: {
        // The pairing: geometric grotesque for UI and body, serif for display.
        sans: ['Figtree', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        serif: ['Noto Serif', 'Georgia', 'serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
      },
      // 8px spacing system.
      spacing: {
        '4.5': '1.125rem',
        '18': '4.5rem',
      },
      borderRadius: {
        // Softer than before: Indus rounds everything generously and takes
        // controls all the way to a pill (`rounded-full` on buttons).
        DEFAULT: '10px',
        md: '12px',
        lg: '16px',
        xl: '20px',
      },
      boxShadow: {
        /**
         * Effectively flat.
         *
         * Indus separates surfaces with a hairline border and whitespace, not
         * with lift — a page of stacked cards stays calm because nothing is
         * competing to float. The tokens survive so call sites don't churn,
         * but they are barely-there; only `popover` keeps real depth, because
         * a menu genuinely does hover above the page it covers.
         */
        xs: '0 1px 0 0 rgb(20 20 20 / 0.02)',
        sm: '0 1px 2px 0 rgb(20 20 20 / 0.03)',
        md: '0 2px 6px -4px rgb(20 20 20 / 0.06)',
        popover: '0 12px 32px -18px rgb(20 20 20 / 0.22)',
      },
      keyframes: {
        'fade-in': {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
        'fade-up': {
          from: { opacity: '0', transform: 'translateY(4px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-in': {
          from: { opacity: '0', transform: 'translateX(-4px)' },
          to: { opacity: '1', transform: 'translateX(0)' },
        },
        // Used only for the indeterminate loading bar.
        indeterminate: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(400%)' },
        },
        // A slow diagonal sweep across the Exclusive swatch in the theme
        // picker — the one deliberately playful animation in an otherwise
        // restrained UI, there to make that option look worth clicking.
        shimmer: {
          '0%': { backgroundPosition: '200% 0' },
          '100%': { backgroundPosition: '-200% 0' },
        },
        // One confetti fleck's flight when Exclusive is selected: launch,
        // tumble, fall away. Each fleck reuses this with a random angle,
        // distance and delay set inline, so ten pieces never move in lockstep.
        'confetti-burst': {
          '0%': { transform: 'translate(0, 0) rotate(0deg)', opacity: '1' },
          '70%': { opacity: '1' },
          '100%': {
            transform: 'translate(var(--confetti-x), var(--confetti-y)) rotate(var(--confetti-spin))',
            opacity: '0',
          },
        },
        // Slow elliptical drift for the Exclusive theme's ambient glow orbs.
        // Two different paths (and durations) so the blue and red lights
        // never move in lockstep. This is the *only* thing animating each
        // orb's position — its scale is driven by JS reading the live TTS
        // level, on a separate nested element, so the two never fight over
        // the same `transform`.
        'drift-1': {
          '0%, 100%': { transform: 'translate(0, 0)' },
          '25%': { transform: 'translate(6%, 8%)' },
          '50%': { transform: 'translate(-4%, 14%)' },
          '75%': { transform: 'translate(-8%, 4%)' },
        },
        'drift-2': {
          '0%, 100%': { transform: 'translate(0, 0)' },
          '30%': { transform: 'translate(-7%, -6%)' },
          '60%': { transform: 'translate(3%, -12%)' },
          '85%': { transform: 'translate(8%, -3%)' },
        },
      },
      animation: {
        'fade-in': 'fade-in 150ms ease-out',
        shimmer: 'shimmer 2.5s linear infinite',
        'confetti-burst': 'confetti-burst 700ms cubic-bezier(0.2, 0.8, 0.4, 1) forwards',
        'fade-up': 'fade-up 180ms ease-out',
        'slide-in': 'slide-in 150ms ease-out',
        indeterminate: 'indeterminate 1.2s ease-in-out infinite',
        'drift-1': 'drift-1 26s ease-in-out infinite',
        'drift-2': 'drift-2 32s ease-in-out infinite',
      },
    },
  },
  plugins: [],
} satisfies Config;
