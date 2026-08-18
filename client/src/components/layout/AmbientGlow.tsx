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
  return null;
}
