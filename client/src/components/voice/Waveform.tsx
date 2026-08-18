/**
 * The live recording waveform — a heads-up scope rather than a bar chart.
 *
 * The bars this replaced had two problems: packed tight enough to be
 * interesting they optically shimmer, and between words they collapse into a
 * dead row of dots. A continuous mirrored envelope fixes both. What is drawn,
 * back to front:
 *
 *   · a centre spine and a fine tick grid — the instrument, always present,
 *     so the strip has structure even in silence;
 *   · a filled envelope, mirrored top and bottom, smoothed through its
 *     points with quadratic segments so the trace curves instead of kinking;
 *   · a second inner trace at reduced amplitude, which is what gives the
 *     thing depth without reaching for a second colour;
 *   · a scan band sweeping left to right, composited *onto what was already
 *     drawn* so it lights the trace rather than washing the background.
 *
 * Two shades of grey, not a colour: a faint resting ring while the mic is
 * closed, ink while recording. Accent colours here — violet, then the mic
 * button's own danger red — were tried and read as noise competing with the
 * shape rather than as useful signal, so the wave stays inside the same
 * restrained, colour-only-when-it-means-something language as the rest of the
 * UI. Within a tone, alpha is the only thing that varies; height already
 * reports loudness, and a second channel saying the same thing only adds
 * glare.
 *
 * Two behaviours keep the motion smooth. The envelope eases toward the
 * amplitudes it is given, fast on the way up and slow on the way down, so a
 * syllable swells and falls away instead of flickering; and a pair of slow
 * sines at unrelated rates keeps a travelling swell underneath everything, so
 * the gaps in real speech still breathe. The loop is a plain
 * `requestAnimationFrame` independent of React, so motion stays smooth even
 * when amplitude updates arrive unevenly.
 */
import { useEffect, useRef } from 'react';
import { cn } from '@/lib/utils';
import { useTheme } from '@/hooks/useTheme';

interface WaveformProps {
  /** Rolling amplitudes in [0,1], oldest first. */
  data: number[];
  /** True while the microphone is actually open. */
  active: boolean;
  className?: string;
}

/** Envelope: how fast the trace rises to a new peak, and how slowly it falls. */
const ATTACK = 0.26;
const RELEASE = 0.07;
/** Horizontal distance between envelope points, in CSS pixels. */
const PITCH = 9;
/** Fraction of the half-height the loudest possible signal may fill. */
const HEIGHT_RATIO = 0.86;
/** Swell that stays under a live signal, so silence between words still moves. */
const BREATH = 0.13;
/** Resting ripple with the mic closed — small enough to read as "waiting". */
const IDLE_AMPLITUDE = 0.1;
/** Fraction of the width each end fades out over. */
const EDGE_FADE = 0.08;
/** Amplitude of the inner trace, as a fraction of the outer one. */
const INNER_RATIO = 0.52;
/** Seconds for the scan band to cross the strip once. */
const SCAN_PERIOD = 3.4;
/** Spacing of the tick grid along the centre line, in CSS pixels. */
const TICK_SPACING = 26;
/**
 * Display curve. Speech amplitude clusters low, so drawn linearly most of a
 * normal sentence sits in the bottom fifth of the strip. The exponent lifts
 * quiet passages into view; it is monotonic, so louder still reads as taller.
 */
const GAIN_CURVE = 0.7;

/** Read a theme token as an [r, g, b] triple so alpha can be applied to it. */
function readRgb(token: string, fallback: [number, number, number]): [number, number, number] {
  if (typeof window === 'undefined') return fallback;
  const raw = getComputedStyle(document.documentElement).getPropertyValue(`--${token}`).trim();
  const parts = raw.split(/[\s,]+/).map(Number);
  return parts.length === 3 && parts.every(Number.isFinite)
    ? (parts as [number, number, number])
    : fallback;
}

function readColors() {
  return {
    live: readRgb('waveform', [20, 20, 20]),
    idle: readRgb('waveform-idle', readRgb('border-strong', [213, 209, 201])),
  };
}

const rgba = ([r, g, b]: [number, number, number], alpha: number) =>
  `rgba(${r}, ${g}, ${b}, ${alpha})`;

/** Sample the source buffer at a fractional index, interpolating between neighbours. */
function sampleAt(source: number[], position: number): number {
  const low = Math.floor(position);
  const high = Math.min(source.length - 1, low + 1);
  const fraction = position - low;
  const a = source[low] ?? 0;
  const b = source[high] ?? 0;
  return a + (b - a) * fraction;
}

/**
 * Lay a smooth path through `ys`, left to right (or right to left when
 * `reverse`). Each control point is the sample itself and each segment ends
 * at the midpoint between neighbours — the standard trick for a curve that
 * passes near every point without the overshoot a spline would give on a
 * signal that can jump between frames.
 */
function tracePath(
  ctx: CanvasRenderingContext2D,
  ys: number[],
  step: number,
  reverse: boolean,
): void {
  const count = ys.length;
  const xAt = (i: number) => i * step;

  if (reverse) {
    ctx.lineTo(xAt(count - 1), ys[count - 1] as number);
    for (let i = count - 1; i > 0; i -= 1) {
      const midX = (xAt(i) + xAt(i - 1)) / 2;
      const midY = ((ys[i] as number) + (ys[i - 1] as number)) / 2;
      ctx.quadraticCurveTo(xAt(i), ys[i] as number, midX, midY);
    }
    ctx.lineTo(xAt(0), ys[0] as number);
    return;
  }

  ctx.lineTo(xAt(0), ys[0] as number);
  for (let i = 0; i < count - 1; i += 1) {
    const midX = (xAt(i) + xAt(i + 1)) / 2;
    const midY = ((ys[i] as number) + (ys[i + 1] as number)) / 2;
    ctx.quadraticCurveTo(xAt(i), ys[i] as number, midX, midY);
  }
  ctx.lineTo(xAt(count - 1), ys[count - 1] as number);
}

export function Waveform({ data, active, className }: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // The animation loop reads everything that changes through refs. Amplitude
  // arrives ~60 times a second; rebuilding the loop on each of those updates
  // would cost far more than the drawing itself.
  const dataRef = useRef(data);
  const activeRef = useRef(active);
  const smoothRef = useRef<number[]>([]);
  const colorsRef = useRef<ReturnType<typeof readColors> | null>(null);

  dataRef.current = data;
  activeRef.current = active;

  // Canvas can't respond to a CSS variable changing, so the literal colours
  // are re-read whenever the theme does change.
  const { theme } = useTheme();
  useEffect(() => {
    colorsRef.current = readColors();
  }, [theme]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Everything decorative here is motion, so honour the OS setting: no
    // breathing, no scan band, and no easing.
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    let frame = 0;
    const topY: number[] = [];
    const bottomY: number[] = [];
    const innerTopY: number[] = [];
    const innerBottomY: number[] = [];

    const draw = (time: number) => {
      frame = requestAnimationFrame(draw);

      const { width, height } = canvas.getBoundingClientRect();
      if (width < 1 || height < 1) return;

      const ratio = window.devicePixelRatio || 1;
      const pixelWidth = Math.round(width * ratio);
      const pixelHeight = Math.round(height * ratio);
      // Resizing the backing store clears it, so only touch it when the
      // element has actually changed size.
      if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
      }
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      const source = dataRef.current;
      if (source.length === 0) return;

      // Sampled at a fixed pitch rather than at the source resolution, so the
      // trace has the same detail on a phone as in a wide desktop column.
      const points = Math.max(12, Math.min(120, Math.round(width / PITCH)));
      if (smoothRef.current.length !== points) smoothRef.current = new Array(points).fill(0);
      const smooth = smoothRef.current;

      const live = activeRef.current;
      const seconds = time / 1000;
      const step = points > 1 ? (source.length - 1) / (points - 1) : 0;

      // ── Advance the envelope ────────────────────────────────────────────
      for (let i = 0; i < points; i += 1) {
        // Two sines at unrelated rates, each drifting along the strip. Beating
        // against one another they never visibly repeat, which is what keeps
        // the resting state from looking like a loading animation.
        const swellA = 0.5 + 0.5 * Math.sin(seconds * 1.5 - i * 0.24);
        const swellB = 0.5 + 0.5 * Math.sin(seconds * 0.9 + i * 0.12);
        const swell = swellA * 0.6 + swellB * 0.4;

        let target: number;
        if (live) {
          const amplitude = Math.pow(
            Math.min(1, Math.max(0, sampleAt(source, i * step))),
            GAIN_CURVE,
          );
          // The floor recedes as the signal grows, so a shout is the signal
          // alone and a pause is the swell alone.
          target = reduced ? amplitude : amplitude + BREATH * swell * (1 - amplitude);
        } else {
          target = reduced ? 0 : IDLE_AMPLITUDE * (0.3 + 0.7 * swell);
        }

        const current = smooth[i] ?? 0;
        const rate = reduced ? 1 : target > current ? ATTACK : RELEASE;
        smooth[i] = current + (target - current) * rate;
      }

      // ── Geometry ────────────────────────────────────────────────────────
      const centreY = height / 2;
      const halfHeight = (height / 2) * HEIGHT_RATIO;
      const pointStep = points > 1 ? width / (points - 1) : width;

      topY.length = points;
      bottomY.length = points;
      innerTopY.length = points;
      innerBottomY.length = points;
      for (let i = 0; i < points; i += 1) {
        const offset = Math.min(1, smooth[i] ?? 0) * halfHeight;
        topY[i] = centreY - offset;
        bottomY[i] = centreY + offset;
        innerTopY[i] = centreY - offset * INNER_RATIO;
        innerBottomY[i] = centreY + offset * INNER_RATIO;
      }

      const colors = colorsRef.current ?? (colorsRef.current = readColors());
      const tone = live ? colors.live : colors.idle;

      // ── The instrument: spine and tick grid ─────────────────────────────
      ctx.strokeStyle = rgba(tone, 0.16);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, centreY + 0.5);
      ctx.lineTo(width, centreY + 0.5);
      ctx.stroke();

      ctx.strokeStyle = rgba(tone, 0.2);
      ctx.beginPath();
      for (let x = TICK_SPACING / 2; x < width; x += TICK_SPACING) {
        ctx.moveTo(Math.round(x) + 0.5, centreY - 3);
        ctx.lineTo(Math.round(x) + 0.5, centreY + 3);
      }
      ctx.stroke();

      // ── Envelope body ───────────────────────────────────────────────────
      ctx.fillStyle = rgba(tone, live ? 0.08 : 0.09);
      ctx.beginPath();
      tracePath(ctx, topY, pointStep, false);
      tracePath(ctx, bottomY, pointStep, true);
      ctx.closePath();
      ctx.fill();

      // Inner trace: a quieter echo of the same curve, which is what makes
      // the envelope read as having volume rather than as a flat blob.
      ctx.strokeStyle = rgba(tone, live ? 0.22 : 0.24);
      ctx.lineWidth = 1;
      ctx.beginPath();
      tracePath(ctx, innerTopY, pointStep, false);
      ctx.stroke();
      ctx.beginPath();
      tracePath(ctx, innerBottomY, pointStep, false);
      ctx.stroke();

      // ── Outline ─────────────────────────────────────────────────────────
      // Lighter than a solid line would suggest: the outline is what carries
      // most of the trace's visual weight, so this is the value that decides
      // whether the whole thing reads as a bold line or a light one.
      ctx.strokeStyle = rgba(tone, live ? 0.5 : 0.6);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      tracePath(ctx, topY, pointStep, false);
      ctx.stroke();
      ctx.beginPath();
      tracePath(ctx, bottomY, pointStep, false);
      ctx.stroke();

      // ── Scan band ───────────────────────────────────────────────────────
      // `source-atop` confines it to pixels already drawn, so it lights the
      // trace and the grid without laying a rectangle over the card behind.
      if (!reduced) {
        const progress = (seconds % SCAN_PERIOD) / SCAN_PERIOD;
        const centreX = -0.2 * width + progress * 1.4 * width;
        const bandWidth = Math.max(48, width * 0.16);
        const band = ctx.createLinearGradient(centreX - bandWidth, 0, centreX + bandWidth, 0);
        band.addColorStop(0, rgba(tone, 0));
        band.addColorStop(0.5, rgba(tone, live ? 0.5 : 0.3));
        band.addColorStop(1, rgba(tone, 0));
        ctx.globalCompositeOperation = 'source-atop';
        ctx.fillStyle = band;
        ctx.fillRect(0, 0, width, height);
        ctx.globalCompositeOperation = 'source-over';
      }

      // ── Fade both ends ──────────────────────────────────────────────────
      // Cut out of what was drawn rather than painting the canvas colour over
      // it, so the strip works on any surface — including the Exclusive
      // theme's translucent cards, where an opaque mask would show as a bar.
      const fadeWidth = Math.min(64, width * EDGE_FADE);
      if (fadeWidth > 1) {
        ctx.globalCompositeOperation = 'destination-out';

        const left = ctx.createLinearGradient(0, 0, fadeWidth, 0);
        left.addColorStop(0, 'rgba(0,0,0,1)');
        left.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = left;
        ctx.fillRect(0, 0, fadeWidth, height);

        const right = ctx.createLinearGradient(width - fadeWidth, 0, width, 0);
        right.addColorStop(0, 'rgba(0,0,0,0)');
        right.addColorStop(1, 'rgba(0,0,0,1)');
        ctx.fillStyle = right;
        ctx.fillRect(width - fadeWidth, 0, fadeWidth, height);

        ctx.globalCompositeOperation = 'source-over';
      }
    };

    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, []);

  return <canvas ref={canvasRef} className={cn('h-12 w-full', className)} aria-hidden />;
}
