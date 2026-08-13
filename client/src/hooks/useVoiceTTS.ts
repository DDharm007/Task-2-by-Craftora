/**
 * Text-to-speech for grounded answers.
 *
 * Two tiers, in order:
 *
 *   1. `POST /api/speak` — Sarvam AI `bulbul:v3`, falling back server-side to
 *      ElevenLabs. Real neural voices, and the only tier that pronounces Indic
 *      scripts properly.
 *   2. The browser's own `speechSynthesis`, used when the server cannot
 *      synthesise (no key, or a free ElevenLabs plan returning 402) or when the
 *      request fails. Voice selection walks a female-preference ladder:
 *      exact-language female → any female → exact language → first available.
 *
 * The hook's returned object is memoised. It is used as an effect dependency by
 * callers, and a fresh identity each render made their cleanups re-run mid
 * playback — which cancelled the utterance, re-rendered, and cancelled again.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { speak as speakViaApi } from '@/lib/api';

export type TTSStatus = 'idle' | 'loading' | 'speaking' | 'paused' | 'error' | 'unsupported';

export interface VoiceTTSState {
  status: TTSStatus;
  error: string | null;
  isSupported: boolean;
  isSpeaking: boolean;
  /** Synthesis is in flight — the server call has not returned audio yet. */
  isLoading: boolean;
  availableLanguages: string[];
  speak: (text: string, language?: string) => void;
  stop: () => void;
  pause: () => void;
  resume: () => void;
  unlock: () => void;
  /**
   * Current output level, 0-1, read imperatively rather than as React state —
   * a visual reacting to it (the Exclusive theme's ambient glow) needs up to
   * 60 readings a second, and running that many `setState` calls through
   * React would re-render the whole tree at frame rate for a value nothing
   * else needs. Callers drive their own `requestAnimationFrame` loop and
   * poll this instead.
   *
   * Real analysis (Web Audio `AnalyserNode`) on the server-audio path;
   * `speechSynthesis` exposes no waveform at all, so that path is a plausible
   * synthesized pulse instead — reacting to *that* it isn't lying about, it's
   * the honest ceiling of what the browser API gives us.
   */
  getAudioLevel: () => number;
}

/** Map language codes the STT returns (ISO-639-3) to BCP-47 tags for TTS. */
const LANG_MAP: Record<string, string> = {
  // Indian languages
  hin_Deva: 'hi-IN',
  hin: 'hi-IN',
  ben_Beng: 'bn-IN',
  ben: 'bn-IN',
  tam_Taml: 'ta-IN',
  tam: 'ta-IN',
  tel_Telu: 'te-IN',
  tel: 'te-IN',
  mar_Deva: 'mr-IN',
  mar: 'mr-IN',
  guj_Gujr: 'gu-IN',
  guj: 'gu-IN',
  kan_Knda: 'kn-IN',
  kan: 'kn-IN',
  mal_Mlym: 'ml-IN',
  mal: 'ml-IN',
  pan_Guru: 'pa-IN',
  pan: 'pa-IN',
  urd_Arab: 'ur-PK',
  urd: 'ur-PK',
  ori_Orya: 'or-IN',
  asm_Beng: 'as-IN',
  // Global
  eng_Latn: 'en-US',
  eng: 'en-US',
  spa: 'es-ES',
  fra: 'fr-FR',
  deu: 'de-DE',
  por: 'pt-BR',
  rus: 'ru-RU',
  ara: 'ar-SA',
  zho: 'zh-CN',
  jpn: 'ja-JP',
  kor: 'ko-KR',
  ita: 'it-IT',
  nld: 'nl-NL',
  pol: 'pl-PL',
  tur: 'tr-TR',
  vie: 'vi-VN',
  tha: 'th-TH',
  swa: 'sw-KE',
  hau: 'ha-NG',
};

const FEMALE_KEYWORDS = [
  'female', 'woman', 'girl',
  'zira', 'hazel', 'heather', 'linda', 'susan', 'eva',
  'heera', 'kalpana', 'priya', 'veena',
  'samantha', 'victoria', 'karen', 'moira', 'fiona',
  'allison', 'ava', 'alex', 'nicky', 'tessa', 'serena',
  'google us english female', 'google uk english female',
  'google español', 'google français', 'google italiano',
  'google deutsch', 'google 日本語', 'google 한국의',
  'google हिन्दी', 'google বাংলা', 'google தமிழ்',
  'aria', 'jenny', 'emma', 'isabella', 'amber', 'ana', 'ashley',
  'cora', 'sara', 'jane', 'nancy', 'michelle', 'leah', 'mia',
  'neerja', 'swara', 'aarav', 'ananya',
];

function isFemale(voice: SpeechSynthesisVoice): boolean {
  const name = voice.name.toLowerCase();
  return (
    FEMALE_KEYWORDS.some((kw) => name.includes(kw)) ||
    name.includes('zira') ||
    name.includes('heera') ||
    name.includes('priya') ||
    name.includes('kalpana') ||
    name.includes('female')
  );
}

function pickVoice(voices: SpeechSynthesisVoice[], bcp47: string): SpeechSynthesisVoice | null {
  if (!voices || voices.length === 0) return null;

  const targetLang = bcp47.toLowerCase().replace('_', '-');
  const langBase = targetLang.split('-')[0] ?? '';

  // 1. Exact female voice for language (e.g. hi-IN or en-US)
  const exactFemale = voices.find(
    (v) => (v.lang.toLowerCase().replace('_', '-') === targetLang || v.lang.toLowerCase().startsWith(langBase)) && isFemale(v),
  );
  if (exactFemale) return exactFemale;

  // 2. Any female voice
  const anyFemale = voices.find((v) => isFemale(v));
  if (anyFemale) return anyFemale;

  // 3. Exact language, any voice
  const exactAny = voices.find((v) => v.lang.toLowerCase().replace('_', '-') === targetLang || v.lang.toLowerCase().startsWith(langBase));
  if (exactAny) return exactAny;

  // 4. Default voice
  return voices[0] ?? null;
}

/** Strip Markdown and citation markers so they are not read aloud. */
function cleanForSpeech(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/\[\d+\](\[\d+\])*/g, ' ')
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/[*_~`>#|]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function useVoiceTTS(): VoiceTTSState {
  const [status, setStatus] = useState<TTSStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  /** Object URL currently held, so it can be revoked instead of leaking. */
  const objectUrlRef = useRef<string | null>(null);
  /**
   * Incremented by every `speak`/`stop`. An in-flight request whose token no
   * longer matches has been superseded and must not start playing over the one
   * that replaced it.
   */
  const requestRef = useRef(0);
  /**
   * Latched once the server reports it cannot synthesise. That verdict is a
   * property of the deployment's API keys, not of the request, so re-asking on
   * every answer would just add a wasted round trip before each fallback.
   */
  const serverUnavailableRef = useRef(false);
  /** Web Audio analyser for the current `<audio>` clip, when one exists. */
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserDataRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  /** Which source `getAudioLevel` should read from right now. */
  const playbackModeRef = useRef<'idle' | 'audio' | 'utterance'>('idle');
  const utteranceStartedAtRef = useRef(0);

  // Playback works either way: the server path needs no browser support, and
  // `speechSynthesis` covers the fallback.
  const isSupported = typeof window !== 'undefined';

  useEffect(() => {
    if (!isSupported) {
      setStatus('unsupported');
      return;
    }

    const loadVoices = () => {
      if ('speechSynthesis' in window) {
        const available = window.speechSynthesis.getVoices();
        if (available.length > 0) {
          setVoices(available);
        }
      }
    };

    loadVoices();
    if (typeof window !== 'undefined' && 'speechSynthesis' in window && window.speechSynthesis.onvoiceschanged !== undefined) {
      window.speechSynthesis.onvoiceschanged = loadVoices;
    }

    return () => {
      if (typeof window !== 'undefined' && 'speechSynthesis' in window && window.speechSynthesis.onvoiceschanged !== undefined) {
        window.speechSynthesis.onvoiceschanged = null;
      }
    };
  }, [isSupported]);

  const stop = useCallback(() => {
    // Invalidate any request still in flight so its audio never starts.
    requestRef.current += 1;

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current = null;
    }
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    utteranceRef.current = null;
    analyserRef.current = null;
    playbackModeRef.current = 'idle';
    setStatus('idle');
  }, []);

  /** Instantaneous 0-1 output level — see the doc comment on the interface. */
  const getAudioLevel = useCallback((): number => {
    const mode = playbackModeRef.current;

    if (mode === 'audio' && analyserRef.current) {
      let data = analyserDataRef.current;
      if (!data || data.length !== analyserRef.current.frequencyBinCount) {
        // Constructed from an explicit ArrayBuffer rather than `new
        // Uint8Array(n)` — the latter types as `Uint8Array<ArrayBufferLike>`,
        // which `getByteFrequencyData` (typed for `Uint8Array<ArrayBuffer>`
        // specifically) rejects.
        data = new Uint8Array(new ArrayBuffer(analyserRef.current.frequencyBinCount));
        analyserDataRef.current = data;
      }
      analyserRef.current.getByteFrequencyData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i += 1) sum += data[i] as number;
      return data.length > 0 ? sum / data.length / 255 : 0;
    }

    if (mode === 'utterance') {
      // No waveform access for speechSynthesis — a slow breathing pulse reads
      // as "speaking" without pretending to track the actual audio.
      const elapsed = performance.now() - utteranceStartedAtRef.current;
      return 0.45 + 0.35 * Math.sin(elapsed / 220);
    }

    return 0;
  }, []);

  const unlock = useCallback(() => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    window.speechSynthesis.resume();
    const utterance = new SpeechSynthesisUtterance('');
    utterance.volume = 0;
    window.speechSynthesis.speak(utterance);
  }, []);

  const pause = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
    }
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.pause();
    }
    setStatus('paused');
  }, []);

  const resume = useCallback(() => {
    if (audioRef.current) {
      void audioRef.current.play();
    }
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.resume();
    }
    setStatus('speaking');
  }, []);

  const fallbackSpeak = useCallback(
    (text: string, language?: string) => {
      if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
        setError('Speech synthesis is not supported in this browser.');
        setStatus('error');
        return;
      }

      // Clear previous audio queue & resume Chrome speech synth
      window.speechSynthesis.cancel();
      window.speechSynthesis.resume();

      setError(null);

      // Dynamically fetch latest voices if state was empty
      const currentVoices = voices.length > 0 ? voices : window.speechSynthesis.getVoices();

      const bcp47 = language
        ? (LANG_MAP[language] ?? (language.includes('-') ? language : 'en-US'))
        : 'en-US';

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = bcp47;
      utterance.rate = 1.0;
      utterance.pitch = 1.1;
      utterance.volume = 1;

      const voice = pickVoice(currentVoices, bcp47);
      if (voice) {
        utterance.voice = voice;
        if (voice.lang) utterance.lang = voice.lang;
      }

      utterance.onstart = () => {
        utteranceStartedAtRef.current = performance.now();
        playbackModeRef.current = 'utterance';
        setStatus('speaking');
      };
      utterance.onend = () => {
        utteranceRef.current = null;
        playbackModeRef.current = 'idle';
        setStatus('idle');
      };
      utterance.onerror = (event) => {
        // `interrupted` and `canceled` are what our own `stop()` produces —
        // expected control flow, not a failure worth logging.
        if (event.error !== 'interrupted' && event.error !== 'canceled') {
          console.warn('SpeechSynthesis error:', event.error);
          setError(`Speech synthesis failed: ${event.error}`);
        }
        utteranceRef.current = null;
        playbackModeRef.current = 'idle';
        setStatus('idle');
      };

      utteranceRef.current = utterance;

      // Small delay to ensure cancel/resume has processed in Chrome
      setTimeout(() => {
        window.speechSynthesis.resume();
        window.speechSynthesis.speak(utterance);
      }, 20);
    },
    [voices],
  );

  const speak = useCallback(
    (rawText: string, language?: string) => {
      const text = cleanForSpeech(rawText);
      if (!text) return;

      stop();
      setError(null);

      // `stop` bumped the token; this call owns everything from here on.
      const token = requestRef.current;
      const superseded = () => requestRef.current !== token;

      if (serverUnavailableRef.current) {
        fallbackSpeak(text, language);
        return;
      }

      setStatus('loading');
      speakViaApi(text, undefined, language)
        .then((url) => {
          if (superseded()) {
            if (url) URL.revokeObjectURL(url);
            return;
          }
          // `null` means the server has no usable TTS provider — a standing
          // condition, so stop asking and use the browser from here on.
          if (!url) {
            serverUnavailableRef.current = true;
            fallbackSpeak(text, language);
            return;
          }

          const audio = new Audio(url);
          audioRef.current = audio;
          objectUrlRef.current = url;

          // Real amplitude data for the ambient glow. Best-effort: some
          // browsers cap the number of live AudioContexts, and Safari can be
          // picky about routing a MediaElementSource — losing this must never
          // cost us the actual playback, so failures here are swallowed and
          // `getAudioLevel` simply falls back to 0 (mode stays 'audio' with
          // no analyser, read as silent rather than a crash).
          try {
            const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
            if (Ctx) {
              if (!audioContextRef.current) audioContextRef.current = new Ctx();
              const ctx = audioContextRef.current;
              if (ctx.state === 'suspended') void ctx.resume();
              // `createMediaElementSource` reroutes the element's output through
              // the graph — skipping the connection to `destination` below
              // would make the clip play silently.
              const source = ctx.createMediaElementSource(audio);
              const analyser = ctx.createAnalyser();
              analyser.fftSize = 64;
              analyser.smoothingTimeConstant = 0.75;
              source.connect(analyser);
              analyser.connect(ctx.destination);
              analyserRef.current = analyser;
            }
          } catch (err) {
            console.warn('Audio analyser unavailable, glow will stay static:', err);
          }
          playbackModeRef.current = 'audio';

          const release = () => {
            if (objectUrlRef.current === url) {
              URL.revokeObjectURL(url);
              objectUrlRef.current = null;
            }
            if (audioRef.current === audio) audioRef.current = null;
            analyserRef.current = null;
            playbackModeRef.current = 'idle';
          };

          audio.onended = () => {
            release();
            if (!superseded()) setStatus('idle');
          };
          audio.onerror = () => {
            release();
            if (!superseded()) fallbackSpeak(text, language);
          };

          setStatus('speaking');
          void audio.play().catch(() => {
            // Autoplay refused until the user has interacted with the page.
            release();
            if (!superseded()) fallbackSpeak(text, language);
          });
        })
        .catch((err: unknown) => {
          if (superseded()) return;
          console.warn('Server TTS failed, falling back to Web Speech API:', err);
          fallbackSpeak(text, language);
        });
    },
    [stop, fallbackSpeak],
  );

  // Release the last object URL and audio graph if the component goes away
  // mid-playback — this only ever runs once, at provider teardown, since the
  // shared TTS context lives for the app's lifetime otherwise.
  useEffect(
    () => () => {
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
      void audioContextRef.current?.close();
    },
    [],
  );

  const availableLanguages = useMemo(
    () => Array.from(new Set(voices.map((v) => v.lang))),
    [voices],
  );

  // Stable identity — callers list this object in effect dependency arrays.
  return useMemo(
    () => ({
      status,
      error,
      isSupported,
      isSpeaking: status === 'speaking',
      isLoading: status === 'loading',
      availableLanguages,
      speak,
      stop,
      pause,
      resume,
      unlock,
      getAudioLevel,
    }),
    [status, error, isSupported, availableLanguages, speak, stop, pause, resume, unlock, getAudioLevel],
  );
}

/** Resolve a raw language code to BCP-47. Used by AnswerPanel. */
export function resolveBcp47(languageCode?: string): string {
  if (!languageCode) return 'en-US';
  return LANG_MAP[languageCode] ?? (languageCode.includes('-') ? languageCode : 'en-US');
}
