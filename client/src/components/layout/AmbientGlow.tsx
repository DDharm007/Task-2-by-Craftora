/**
 * Ambient background lighting for the HHGoa'26 Exclusive theme.
 *
 * Two soft lights — blue and red — drift slowly behind the glass-morphic
 * cards (the blur that makes them visible through the cards lives in
 * index.css, on `[data-theme='exclusive'] .bg-card`; the two only look right
 * shipped together). While TTS is speaking, both breathe in time with the
 * actual output level: real amplitude on the server-audio path, a plausible
 * synthesized pulse on the `speechSynthesis` fallback — see the doc comment
 * on `getAudioLevel` in useVoiceTTS for why that split exists.
 *
 * Each light is two nested elements so its two motions never fight over the
 * same CSS property: the outer element runs the slow positional drift as a
 * CSS keyframe animation, the inner element's `transform: scale(...)` is set
 * directly from a requestAnimationFrame loop reading the live TTS level.
 * Driving that loop through refs instead of React state matters here — it
 * runs up to 60 times a second, and a `setState` at that rate would
 * re-render the whole page for a value nothing else consumes.
 */
import { useEffect, useRef } from 'react';
import { useTheme } from '@/hooks/useTheme';
import { useTTS } from '@/components/voice/TTSProvider';

export function AmbientGlow() {
  const { theme } = useTheme();
  const tts = useTTS();
  const blueRef = useRef<HTMLDivElement>(null);
  const redRef = useRef<HTMLDivElement>(null);
  const levelRef = useRef(0);

  useEffect(() => {
    if (theme !== 'exclusive') return undefined;

    let raf = requestAnimationFrame(tick);

    function tick() {
      const target = tts.getAudioLevel();
      // Ease toward the target instead of snapping to it — a raw analyser
      // reading is jittery frame to frame, and this smoothing is what makes
      // the result read as "breathing" rather than "flickering".
      levelRef.current += (target - levelRef.current) * 0.15;
      const level = levelRef.current;

      if (blueRef.current) {
        blueRef.current.style.transform = `scale(${1 + level * 0.35})`;
        blueRef.current.style.opacity = String(Math.min(1, 0.5 + level * 0.5));
      }
      if (redRef.current) {
        redRef.current.style.transform = `scale(${1 + level * 0.5})`;
        redRef.current.style.opacity = String(Math.min(1, 0.42 + level * 0.45));
      }

      raf = requestAnimationFrame(tick);
    }

    return () => cancelAnimationFrame(raf);
  }, [theme, tts]);

  if (theme !== 'exclusive') return null;

  return (
    <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden" aria-hidden="true">
      <div className="absolute -left-[8%] -top-[10%] size-[560px] animate-drift-1">
        <div
          ref={blueRef}
          className="size-full rounded-full opacity-50 transition-transform duration-150 ease-out will-change-transform"
          style={{
            background: 'radial-gradient(circle, rgb(56 130 246 / 0.55) 0%, rgb(56 130 246 / 0) 70%)',
          }}
        />
      </div>
      <div className="absolute -bottom-[12%] -right-[6%] size-[520px] animate-drift-2">
        <div
          ref={redRef}
          className="size-full rounded-full opacity-40 transition-transform duration-150 ease-out will-change-transform"
          style={{
            background: 'radial-gradient(circle, rgb(239 68 68 / 0.5) 0%, rgb(239 68 68 / 0) 70%)',
          }}
        />
      </div>
    </div>
  );
}
